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

Selection precedence (first hit wins):
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

# International endpoint (also supports China via MINIMAX_BASE_URL config)
BASE_URL = "https://api.minimax.io"

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

# ---------------------------------------------------------------------------
# Config
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
        # Also check MINIMAX_CN_API_KEY for China endpoint users
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
                    "url": "https://platform.minimax.io",
                },
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        # MiniMax supports image-to-image via subject_reference
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
        # TODO: Implement the actual MiniMax API call
        #
        # Steps:
        # 1. Resolve API key (MINIMAX_API_KEY or MINIMAX_CN_API_KEY)
        # 2. Resolve base URL (api.minimax.io or api.minimaxi.com via config)
        # 3. Build payload:
        #    {
        #        "model": resolved_model_id,
        #        "prompt": prompt,
        #        "aspect_ratio": mapped_aspect_ratio,
        #        "response_format": "base64",
        #    }
        # 4. If image_url or reference_image_urls provided, add
        #    subject_reference for image-to-image
        # 5. POST to {base_url}/v1/image_generation
        # 6. Parse response: data.image_base64[0] or data.image_url[0]
        # 7. Save via save_b64_image() or save_url_image()
        # 8. Return success_response() or error_response()
        #
        # See plugins/image_gen/xai/__init__.py for reference.
        # See https://platform.minimax.io/docs/guides/image-generation
        # for API docs.
        return error_response(
            error="MiniMax image generation plugin is not yet implemented",
            error_type="not_implemented",
            provider="minimax",
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire ``MiniMaxImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(MiniMaxImageGenProvider())
