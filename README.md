# Hermes MiniMax Image Gen

A [Hermes Agent](https://hermes-agent.nousresearch.com/) **image generation provider plugin** that adds MiniMax Image-01 as an `image_generate` backend — text-to-image and image-to-image with subject reference (character consistency).

## Status

🚧 **Work in progress** — Plugin skeleton, not yet functional.

## What this does

Registers a MiniMax backend with Hermes Agent's pluggable `image_gen` system, so you can select it via `hermes tools` → Image Generation → MiniMax and use it through the standard `image_generate` tool — just like OpenAI, FAL, xAI, or Krea.

## Prerequisites

- Hermes Agent (built-in `image_gen_provider` ABC)
- A [MiniMax](https://www.minimax.io) account with API key (`MINIMAX_API_KEY`)
- `MINIMAX_API_KEY` in `~/.hermes/.env` (you probably already have this if you use MiniMax as your chat model provider)

## Installation

```bash
# Clone the repo
git clone https://github.com/derekhsu/hermes-minimax-image-gen.git
cd hermes-minimax-image-gen

# Symlink or copy to Hermes user plugins
mkdir -p ~/.hermes/plugins/image_gen
ln -s "$PWD"/plugins/image_gen/minimax ~/.hermes/plugins/image_gen/minimax

# Enable the plugin
hermes plugins enable minimax

# Configure as active image gen provider
hermes config set image_gen.provider minimax
```

Or configure manually in `config.yaml`:

```yaml
image_gen:
  provider: minimax
  minimax:
    model: image-01
```

## Usage

Once installed and configured:

```
> generate an image of a cyberpunk city at night
```

Hermes will route the request to MiniMax Image-01 via this plugin.

## Development

See [`DEVELOPMENT.md`](DEVELOPMENT.md) for the plugin architecture and how to contribute.

## API Reference

- [MiniMax Image Generation Guide](https://platform.minimax.io/docs/guides/image-generation)
- [Text to Image API](https://platform.minimax.io/docs/api-reference/image-generation-t2i)
- [Image to Image API](https://platform.minimax.io/docs/api-reference/image-generation-i2i)

## License

MIT
