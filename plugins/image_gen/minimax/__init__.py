"""MiniMax Image-01 image generation backend for Hermes Agent.

Exposes MiniMax's ``image-01`` model as an :class:`ImageGenProvider`
implementation.

Features:
- Text-to-image (T2I) generation
- Image-to-image (I2I) via subject_reference (character consistency)
- Multiple aspect ratios (1:1, 16:9, 9:16, 4:3, 3:2, 2:3, 3:4, 21:9)
- Custom dimensions (width/height)
- Prompt optimizer
- AIGC watermark embedding
- Base64 or URL output

API docs: https://platform.minimax.io/docs/guides/image-generation

Selection precedence for model (first hit wins):
1. ``MINIMAX_IMAGE_MODEL`` env var (escape hatch for scripts / tests)
2. ``image_gen.minimax.model`` in ``config.yaml``
3. ``image_gen.model`` in ``config.yaml`` (when it matches our model IDs)
4. :data:`DEFAULT_MODEL` — ``image-01``
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    normalize_reference_images,
    resolve_aspect_ratio,
    save_b64_image,
    save_url_image,
    success_response,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL_GLOBAL = "https://api.minimax.io"
BASE_URL_CHINA = "https://api.minimaxi.com"

# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------

_MODELS: Dict[str, Dict[str, Any]] = {
    "image-01": {
        "display": "Image-01",
        "speed": "~10-30s",
        "strengths": "General-purpose text-to-image & image-to-image with subject reference support",
        "supports_custom_size": True,
    },
}

DEFAULT_MODEL = "image-01"

# Hermes uses 3 abstract aspect ratios → MiniMax aspect ratios.
# MiniMax supports more (4:3, 3:2, 2:3, 3:4, 21:9) but Hermes only
# exposes landscape/square/portrait. Users can pass custom ratios
# via kwargs.
_ASPECT_MAP: Dict[str, str] = {
    "landscape": "16:9",
    "square": "1:1",
    "portrait": "9:16",
}

# Valid MiniMax aspect ratios
_VALID_ASPECT_RATIOS = frozenset({
    "16:9", "1:1", "9:16", "4:3", "3:2", "2:3", "3:4", "21:9",
})

# Output format: prefer base64 so we can cache locally
DEFAULT_RESPONSE_FORMAT = "base64"

# Default image count
DEFAULT_IMAGE_COUNT = 1

# API timeout
_REQUEST_TIMEOUT = 120

# Prompt max length per MiniMax docs
_MAX_PROMPT_LENGTH = 1500

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_minimax_config() -> Dict[str, Any]:
    """Read ``image_gen.minimax`` (with fallthrough to ``image_gen``) from config.yaml."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    """Decide which model to use and return ``(model_id, meta)``."""
    env_override = os.environ.get("MINIMAX_IMAGE_MODEL")
    if env_override and env_override in _MODELS:
        return env_override, _MODELS[env_override]

    cfg = _load_minimax_config()
    minimax_cfg = cfg.get("minimax") if isinstance(cfg.get("minimax"), dict) else {}
    candidate: Optional[str] = None
    if isinstance(minimax_cfg, dict):
        value = minimax_cfg.get("model")
        if isinstance(value, str) and value in _MODELS:
            candidate = value
    if candidate is None:
        top = cfg.get("model")
        if isinstance(top, str) and top in _MODELS:
            candidate = top

    if candidate is not None:
        return candidate, _MODELS[candidate]

    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


def _resolve_api_key() -> Tuple[Optional[str], Optional[str]]:
    """Resolve API key and base URL.

    Returns ``(api_key, base_url)``.

    Precedence for base URL:
    1. ``image_gen.minimax.base_url`` in config
    2. ``MINIMAX_BASE_URL`` env var
    3. China endpoint if ``MINIMAX_CN_API_KEY`` is set
    4. Global endpoint (default)

    Precedence for API key:
    1. ``MINIMAX_API_KEY`` env var
    2. ``MINIMAX_CN_API_KEY`` env var
    """
    api_key = os.environ.get("MINIMAX_API_KEY") or os.environ.get("MINIMAX_CN_API_KEY")
    if not api_key:
        return None, None

    # Resolve base URL from config first, then env, then heuristics.
    cfg = _load_minimax_config()
    minimax_cfg = cfg.get("minimax") if isinstance(cfg.get("minimax"), dict) else {}

    base_url = None
    if isinstance(minimax_cfg, dict):
        cfg_url = minimax_cfg.get("base_url")
        if isinstance(cfg_url, str) and cfg_url.strip():
            base_url = cfg_url.strip().rstrip("/")

    if not base_url:
        env_url = os.environ.get("MINIMAX_BASE_URL")
        if env_url and env_url.strip():
            base_url = env_url.strip().rstrip("/")

    if not base_url:
        # Use China endpoint if CN key is set, otherwise global.
        if os.environ.get("MINIMAX_CN_API_KEY"):
            base_url = BASE_URL_CHINA
        else:
            base_url = BASE_URL_GLOBAL

    # Strip /anthropic suffix if someone copied their chat base URL.
    if base_url.endswith("/anthropic"):
        base_url = base_url[: -len("/anthropic")]

    return api_key, base_url


def _resolve_aspect_ratio(aspect_ratio: str) -> str:
    """Map Hermes aspect ratio to MiniMax aspect ratio string.

    Accepts Hermes abstract names (landscape/square/portrait) or
    explicit MiniMax ratios (16:9, 1:1, etc.).
    """
    resolved = resolve_aspect_ratio(aspect_ratio)
    minimax_ar = _ASPECT_MAP.get(resolved)
    if minimax_ar:
        return minimax_ar
    # If it's already a valid MiniMax ratio, pass through.
    if resolved in _VALID_ASPECT_RATIOS:
        return resolved
    return "1:1"


def _extract_error_message(response: requests.Response) -> str:
    """Extract a human-readable error message from a MiniMax API response."""
    try:
        body = response.json()
    except Exception:
        return response.text[:300] if response.text else f"HTTP {response.status_code}"

    # MiniMax business errors: base_resp.status_msg
    base_resp = body.get("base_resp")
    if isinstance(base_resp, dict):
        msg = base_resp.get("status_msg")
        if msg:
            return str(msg)

    # Fallback to generic message field.
    return body.get("message") or body.get("error") or response.text[:300]


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class MiniMaxImageGenProvider(ImageGenProvider):
    """MiniMax ``image-01`` image generation backend."""

    @property
    def name(self) -> str:
        return "minimax"

    @property
    def display_name(self) -> str:
        return "MiniMax"

    def is_available(self) -> bool:
        api_key = os.environ.get("MINIMAX_API_KEY")
        cn_api_key = os.environ.get("MINIMAX_CN_API_KEY")
        return bool(api_key or cn_api_key)

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
            }
            for model_id, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "MiniMax",
            "badge": "paid",
            "tag": "MiniMax Image-01 — text-to-image & image editing with subject reference for character consistency",
            "env_vars": [
                {
                    "key": "MINIMAX_API_KEY",
                    "prompt": "MiniMax API key (global endpoint)",
                    "url": "https://platform.minimax.io/user-center/basic-information/interface-key",
                },
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        # MiniMax supports image-to-image via subject_reference.
        return {"modalities": ["text", "image"], "max_reference_images": 10}

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate an image via MiniMax Image-01.

        Text-to-image: prompt → image.
        Image-to-image: prompt + subject_reference → image with character
        consistency (uses ``subject_reference`` with ``type: "character"``).
        """
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="minimax",
                aspect_ratio=aspect,
            )

        # Truncate prompt if too long (MiniMax limit: 1500 chars).
        if len(prompt) > _MAX_PROMPT_LENGTH:
            logger.warning(
                "Prompt truncated from %d to %d characters",
                len(prompt),
                _MAX_PROMPT_LENGTH,
            )
            prompt = prompt[:_MAX_PROMPT_LENGTH]

        # Resolve credentials.
        api_key, base_url = _resolve_api_key()
        if not api_key:
            return error_response(
                error=(
                    "No MiniMax API key found. Set MINIMAX_API_KEY (global) "
                    "or MINIMAX_CN_API_KEY (China) in ~/.hermes/.env, "
                    "or configure via `hermes tools` → Image Generation → MiniMax."
                ),
                error_type="auth_required",
                provider="minimax",
                prompt=prompt,
                aspect_ratio=aspect,
            )

        model_id, meta = _resolve_model()
        minimax_ar = _resolve_aspect_ratio(aspect)

        # Determine if this is an image-to-image call.
        # MiniMax supports subject_reference for character consistency.
        sources: List[str] = []
        if isinstance(image_url, str) and image_url.strip():
            sources.append(image_url.strip())
        for ref in (normalize_reference_images(reference_image_urls) or []):
            sources.append(ref)
        is_i2i = bool(sources)
        modality = "image" if is_i2i else "text"

        # Build the request payload.
        # Always request base64 so we can cache locally (like xAI does).
        payload: Dict[str, Any] = {
            "model": model_id,
            "prompt": prompt,
            "response_format": "base64",
        }

        # Aspect ratio.
        payload["aspect_ratio"] = minimax_ar

        # Custom dimensions (image-01 only). Overridden by aspect_ratio
        # on MiniMax's side, so only set when no aspect_ratio is explicitly
        # given via kwargs (we always set aspect_ratio above, so width/height
        # are only useful if the user explicitly passes them as override).
        width = kwargs.get("width")
        height = kwargs.get("height")
        if isinstance(width, int) and isinstance(height, int):
            if 512 <= width <= 2048 and 512 <= height <= 2048:
                if width % 8 == 0 and height % 8 == 0:
                    payload["width"] = width
                    payload["height"] = height

        # Image count.
        n = kwargs.get("n", DEFAULT_IMAGE_COUNT)
        if isinstance(n, int) and 1 <= n <= 9:
            payload["n"] = n

        # Prompt optimizer.
        prompt_optimizer = kwargs.get("prompt_optimizer")
        if prompt_optimizer is not None:
            payload["prompt_optimizer"] = bool(prompt_optimizer)
        else:
            # Check config for default.
            cfg = _load_minimax_config()
            minimax_cfg = cfg.get("minimax") if isinstance(cfg.get("minimax"), dict) else {}
            if isinstance(minimax_cfg, dict) and minimax_cfg.get("prompt_optimizer"):
                payload["prompt_optimizer"] = True

        # AIGC watermark.
        aigc_watermark = kwargs.get("aigc_watermark")
        if aigc_watermark is not None:
            payload["aigc_watermark"] = bool(aigc_watermark)

        # Seed for reproducibility.
        seed = kwargs.get("seed")
        if isinstance(seed, int):
            payload["seed"] = seed

        # Subject reference for image-to-image (character consistency).
        if is_i2i:
            # Build subject_reference entries. For each source image,
            # use the URL or local file path.
            ref_entries: List[Dict[str, str]] = []
            for src in sources:
                src = src.strip()
                lower = src.lower()
                if lower.startswith(("http://", "https://", "data:")):
                    # Public URL or data URI — pass directly.
                    ref_entries.append({
                        "type": "character",
                        "image_file": src,
                    })
                elif os.path.isfile(src):
                    # Local file — read and encode as data URI.
                    import base64

                    try:
                        with open(src, "rb") as fh:
                            raw = fh.read()
                    except OSError as exc:
                        logger.warning("Could not read source image %s: %s", src, exc)
                        continue
                    ext = os.path.splitext(src)[1].lstrip(".").lower() or "png"
                    if ext == "jpg":
                        ext = "jpeg"
                    b64 = base64.b64encode(raw).decode("utf-8")
                    data_uri = f"data:image/{ext};base64,{b64}"
                    ref_entries.append({
                        "type": "character",
                        "image_file": data_uri,
                    })
                else:
                    logger.warning(
                        "Skipping unrecognised source image ref (not a URL, data URI, or existing file): %s",
                        src,
                    )

            if ref_entries:
                # MiniMax caps subject_reference at some number; 10 is
                # reasonable and matches our capabilities declaration.
                payload["subject_reference"] = ref_entries[:10]

        # ---------------------------------------------------------------
        # Make the API call
        # ---------------------------------------------------------------
        endpoint_url = f"{base_url}/v1/image_generation"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(
                endpoint_url,
                headers=headers,
                json=payload,
                timeout=_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
        except requests.HTTPError as exc:
            resp_obj = exc.response
            status = resp_obj.status_code if resp_obj is not None else 0
            err_msg = _extract_error_message(resp_obj) if resp_obj is not None else str(exc)
            logger.error("MiniMax image gen failed (%d): %s", status, err_msg)

            # Map MiniMax business error codes to user-friendly messages.
            if resp_obj is not None:
                try:
                    body = resp_obj.json()
                    base_resp = body.get("base_resp") if isinstance(body, dict) else None
                    if isinstance(base_resp, dict):
                        code = base_resp.get("status_code")
                        if code == 1004 or code == 2049:
                            return error_response(
                                error="MiniMax API authentication failed. Check your MINIMAX_API_KEY.",
                                error_type="auth_required",
                                provider="minimax",
                                model=model_id,
                                prompt=prompt,
                                aspect_ratio=aspect,
                            )
                        if code == 1008:
                            return error_response(
                                error="MiniMax account balance insufficient. Top up at https://platform.minimax.io.",
                                error_type="billing_error",
                                provider="minimax",
                                model=model_id,
                                prompt=prompt,
                                aspect_ratio=aspect,
                            )
                        if code == 1026:
                            return error_response(
                                error="Sensitive content detected in prompt. Please modify your prompt and try again.",
                                error_type="content_policy",
                                provider="minimax",
                                model=model_id,
                                prompt=prompt,
                                aspect_ratio=aspect,
                            )
                        if code == 1002:
                            return error_response(
                                error="MiniMax API rate limit reached. Please try again later.",
                                error_type="rate_limit",
                                provider="minimax",
                                model=model_id,
                                prompt=prompt,
                                aspect_ratio=aspect,
                            )
                except Exception:
                    pass

            return error_response(
                error=f"MiniMax image generation failed ({status}): {err_msg}",
                error_type="api_error",
                provider="minimax",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.Timeout:
            return error_response(
                error=f"MiniMax image generation timed out ({_REQUEST_TIMEOUT}s)",
                error_type="timeout",
                provider="minimax",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.ConnectionError as exc:
            return error_response(
                error=f"MiniMax connection error: {exc}",
                error_type="connection_error",
                provider="minimax",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # ---------------------------------------------------------------
        # Parse response
        # ---------------------------------------------------------------
        try:
            result = resp.json()
        except Exception as exc:
            return error_response(
                error=f"MiniMax returned invalid JSON: {exc}",
                error_type="invalid_response",
                provider="minimax",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # Check MiniMax business-level status.
        base_resp = result.get("base_resp") if isinstance(result, dict) else None
        if isinstance(base_resp, dict):
            status_code = base_resp.get("status_code")
            if status_code != 0:
                status_msg = base_resp.get("status_msg", "Unknown error")
                logger.error(
                    "MiniMax API business error (%s): %s",
                    status_code,
                    status_msg,
                )
                return error_response(
                    error=f"MiniMax API error ({status_code}): {status_msg}",
                    error_type="api_error",
                    provider="minimax",
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

        # Extract image data.
        data = result.get("data") if isinstance(result, dict) else None
        if not isinstance(data, dict):
            return error_response(
                error="MiniMax response missing 'data' field",
                error_type="empty_response",
                provider="minimax",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # MiniMax returns either image_base64[] or image_urls[] depending
        # on response_format. We requested base64, so expect that first.
        b64_images: List[str] = data.get("image_base64") or []
        url_images: List[str] = data.get("image_urls") or []

        if b64_images:
            # Save base64 image to local cache. If multiple images were
            # requested (n > 1), return the first one and note the count.
            try:
                saved_path = save_b64_image(
                    b64_images[0],
                    prefix=f"minimax_{model_id}",
                )
            except Exception as exc:
                return error_response(
                    error=f"Could not save MiniMax image to cache: {exc}",
                    error_type="io_error",
                    provider="minimax",
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            image_ref = str(saved_path)
        elif url_images:
            # Fallback: materialise URL to local cache (same rationale as
            # xAI — ephemeral signed URLs may expire).
            try:
                saved_path = save_url_image(
                    url_images[0],
                    prefix=f"minimax_{model_id}",
                )
            except Exception as exc:
                logger.warning(
                    "MiniMax image URL %s could not be cached (%s); falling back to bare URL.",
                    url_images[0],
                    exc,
                )
                image_ref = url_images[0]
            else:
                image_ref = str(saved_path)
        else:
            return error_response(
                error="MiniMax response contained neither image_base64 nor image_urls",
                error_type="empty_response",
                provider="minimax",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # Build extra metadata.
        extra: Dict[str, Any] = {
            "aspect_ratio": minimax_ar,
        }
        if is_i2i:
            extra["modality"] = "image-to-image"
        if b64_images and len(b64_images) > 1:
            extra["total_generated"] = len(b64_images)
        if seed is not None:
            extra["seed"] = seed

        return success_response(
            image=image_ref,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="minimax",
            modality=modality,
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire ``MiniMaxImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(MiniMaxImageGenProvider())
