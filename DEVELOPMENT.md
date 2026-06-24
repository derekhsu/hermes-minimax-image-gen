# Hermes MiniMax Image Gen — Development Guide

## Architecture

This is a **Hermes Agent backend plugin** that implements the `ImageGenProvider` ABC.

```
~/.hermes/plugins/image_gen/minimax/
├── plugin.yaml          # Plugin manifest
└── __init__.py          # ImageGenProvider implementation + register()
```

### How it plugs in

1. Hermes discovers plugins under `~/.hermes/plugins/image_gen/<name>/`
2. Loads `plugin.yaml` (must have `kind: backend`)
3. Calls `register(ctx)` from `__init__.py`
4. `ctx.register_image_gen_provider(provider)` registers the provider in the image gen registry
5. User selects it via `image_gen.provider` in `config.yaml`

### Key contract

The provider class must implement `agent.image_gen_provider.ImageGenProvider`:

| Method | Required | Description |
|--------|----------|-------------|
| `name` (property) | ✅ | Stable ID, matches `image_gen.provider` config value |
| `display_name` (property) | — | Label shown in `hermes tools` |
| `is_available()` | — | Gate for missing creds/deps |
| `list_models()` | — | Catalog for model picker |
| `default_model()` | — | Fallback when no model configured |
| `get_setup_schema()` | — | Picker metadata + env-var prompts |
| `generate(prompt, aspect_ratio, **kwargs)` | ✅ | The actual image generation call |

### Reference plugins

Study these bundled plugins for patterns:

- `plugins/image_gen/xai/__init__.py` — simplest REST API plugin (closest to MiniMax)
- `plugins/image_gen/openai/__init__.py` — tiered models, image editing support
- `plugins/image_gen/krea/__init__.py` — async job polling pattern

## MiniMax API

**Endpoint:** `POST https://api.minimax.io/v1/image_generation`
**China endpoint:** `POST https://api.minimaxi.com/v1/image_generation`
**Model:** `image-01`

### Text-to-Image

```json
{
  "model": "image-01",
  "prompt": "a cyberpunk city at night",
  "aspect_ratio": "16:9",
  "response_format": "base64"
}
```

Response:
```json
{
  "data": {
    "image_base64": ["..."],
    "image_url": ["..."]
  }
}
```

### Image-to-Image (subject reference)

```json
{
  "model": "image-01",
  "prompt": "the same character in a library",
  "aspect_ratio": "16:9",
  "subject_reference": [
    {"type": "character", "image_file": "https://example.com/ref.png"}
  ],
  "response_format": "base64"
}
```

### Parameters

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `model` | string | ✅ | `image-01` |
| `prompt` | string | ✅ | Text description |
| `aspect_ratio` | string | — | `16:9`, `1:1`, `9:16`, `4:3`, `3:2`, `2:3`, `3:4`, `21:9` |
| `response_format` | string | — | `base64` or `url` |
| `subject_reference` | array | — | Reference images for character consistency |
| `width` / `height` | int | — | Custom dimensions (image-01 only) |
| `prompt_optimizer` | bool | — | Enable prompt enhancement |
| `aigc_watermark` | bool | — | Add AIGC watermark |

## TODO

- [ ] Implement `generate()` for text-to-image
- [ ] Implement image-to-image via `subject_reference`
- [ ] Handle aspect ratio mapping (Hermes: landscape/square/portrait → MiniMax)
- [ ] Config model selection
- [ ] Test with real MiniMax API key
- [ ] Publish as installable plugin
