# Chaptergen - YouTube Chapter Generator

CLI tool that generates YouTube chapters from video transcripts using LLMs. Works with local models via [Ollama](https://ollama.com/) by default, with optional cloud provider support.

## Installation

```bash
pip install -e .
```

With OpenAI support:

```bash
pip install -e ".[openai]"
```

With dev/test dependencies:

```bash
pip install -e ".[all]"
```

## Quick Start

### Using Ollama (local, default)

1. Install and start Ollama: <https://ollama.com/>
2. Pull a model:

```bash
ollama pull llama3.1
```

3. Generate chapters:

```bash
chaptergen generate --input transcript.srt --format youtube
```

### Using OpenAI

```bash
export OPENAI_API_KEY=sk-...
chaptergen generate --input transcript.srt --provider openai --model gpt-4o-mini --format youtube
```

## Supported Input Formats

| Format | Extensions | Notes |
|--------|-----------|-------|
| Plain text | `.txt`, `.md` | Timestamps optional: `MM:SS` or `HH:MM:SS` at the start of a line, or on a line of their own (as copied from YouTube's "Show transcript" panel) |
| SubRip | `.srt` | Standard subtitle format |
| WebVTT | `.vtt` | Web subtitle format |

## Usage

### Generate Chapters

```bash
chaptergen generate --input <file> [options]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--input`, `-i` | *(required)* | Path to transcript file |
| `--provider`, `-p` | `ollama` | LLM provider (`ollama`, `openai`) |
| `--model`, `-m` | `llama3.1` / `gpt-4o-mini` | Model name |
| `--format`, `-f` | `chapters` | Output format (`chapters`, `youtube`, `json`) |
| `--output`, `-o` | stdout | Write to file |
| `--api-key` | — | API key (prefer `--api-key-env`) |
| `--api-key-env` | — | Env var name holding API key |
| `--base-url` | — | Override provider URL |
| `--temperature` | `0.0` | Sampling temperature |
| `--max-chapters` | — | Suggest max chapter count to LLM |
| `--min-gap` | `30` | Minimum seconds between chapters |

### Check Provider

Verify that your provider is reachable and the model is available:

```bash
chaptergen check --provider ollama --model llama3.1
```

## Output Formats

### chapters (default)

```
00:00 Introduction
02:15 Setting Up the Project
05:30 Writing Core Logic
```

### youtube

Ready to paste into a YouTube video description:

```
Chapters:
00:00 Introduction
02:15 Setting Up the Project
05:30 Writing Core Logic
```

### json

```json
{
  "provider": "ollama",
  "model": "llama3.1",
  "chapters": [
    {"timestamp": "00:00", "start_seconds": 0, "title": "Introduction"},
    {"timestamp": "02:15", "start_seconds": 135, "title": "Setting Up the Project"}
  ]
}
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `CHAPTERGEN_PROVIDER` | Default provider |
| `CHAPTERGEN_MODEL` | Default model |
| `OLLAMA_HOST` | Ollama server address, same format as Ollama's own, e.g. `gpu-box:11434` or `https://ollama.example.com` (default: `http://localhost:11434`) |
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENAI_BASE_URL` | OpenAI-compatible base URL |

## Model Recommendations

### Ollama (local)

- **llama3.1** — Good balance of speed and quality
- **llama3.1:70b** — Higher quality, needs more VRAM
- **mistral** — Fast, decent results

### OpenAI

- **gpt-5-mini** — Cost-effective, good quality
- **gpt-5.3** — Best quality

## Development

```bash
pip install -e ".[all]"
pytest
```

## Troubleshooting

**"Cannot connect to Ollama"** — Make sure Ollama is running (`ollama serve`) and accessible at the expected URL.

**"Model not found"** — Pull the model first: `ollama pull llama3.1`

**"No valid chapters found"** — The LLM failed to return structured output. Try a different model or re-run (local models can be inconsistent). Adding `--temperature 0` helps with determinism.

**Chapters look wrong** — Adjust `--min-gap` to control spacing, or use `--max-chapters` to limit count.
