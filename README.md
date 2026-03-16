<img src="assets/logo.svg" alt="AI_devs 4: Builders" width="200">

## Requirements

This project runs on [Node.js](https://nodejs.org/) (version 24 or later), a JavaScript runtime. Node.js ships with **npm**, a package manager used to install dependencies and run the examples.

### Installing Node.js

```bash
# macOS (Homebrew — https://brew.sh)
brew install node

# Windows (winget — https://learn.microsoft.com/en-us/windows/package-manager/winget)
winget install OpenJS.NodeJS

# Linux / Ubuntu / Debian (https://deb.nodesource.com)
curl -fsSL https://deb.nodesource.com/setup_24.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify installation
node -v
npm -v
```

Alternatively, download the installer directly from [nodejs.org/en/download](https://nodejs.org/en/download).

## Setup

Copy `env.example` to `.env`.

Set one Responses API key. You can choose between **OpenAI** and **OpenRouter**:

**[OpenRouter](https://openrouter.ai/settings/keys)** (recommended) — create an account and generate an API key. No additional verification required.

```bash
OPENROUTER_API_KEY=your_api_key_here
```

**[OpenAI](https://platform.openai.com/api-keys)** — create an account and generate an API key. Note that OpenAI requires [organization verification](https://help.openai.com/en/articles/10910291-api-organization-verification) before API access is granted, which may take additional time.

```bash
OPENAI_API_KEY=your_api_key_here
```

If both keys are present, provider defaults to OpenAI. Override with `AI_PROVIDER=openrouter`.

Some Lesson 04 examples also require:

```bash
GEMINI_API_KEY=your_gemini_key_here
REPLICATE_API_TOKEN=your_replicate_token_here
```

For image-generation examples, `OPENROUTER_API_KEY` can be used as the image backend with `google/gemini-3.1-flash-image-preview`. `GEMINI_API_KEY` remains the native fallback, and some media examples still need it for native Gemini-only features.

Some Lesson 05 examples also require:

```bash
RESEND_API_KEY=re_...
RESEND_FROM=noreply@yourdomain.com
SEED_API_KEY=your_optional_seed_token
```

## Lesson 01

| Example | Run | Description |
|---------|-----|-------------|
| `01_01_interaction` | `npm run lesson1:interaction` | Multi-turn conversation via input history |
| `01_01_structured` | `npm run lesson1:structured` | Structured JSON output with schema validation |
| `01_01_grounding` | `npm run lesson1:grounding` | Fact-checked HTML from markdown notes |

Install dependencies:

```bash
npm run lesson1:install
```

## Lesson 02

| Example | Run | Description |
|---------|-----|-------------|
| `01_02_tool_use` | `npm run lesson2:tool_use` | Function calling with sandboxed filesystem tools |
| `01_02_tools` | `bun run lesson2:minimal` | Minimal Responses API function-calling demo with a single `get_weather` tool |

Install dependencies:

```bash
npm run lesson2:install
```

## Lesson 03

| Example | Run | Description |
|---------|-----|-------------|
| `01_03_mcp_core` | `npm run lesson3:mcp_core` | Core MCP capabilities via stdio transport |
| `01_03_mcp_native` | `npm run lesson3:mcp_native` | One agent using MCP and native tools together |
| `01_03_mcp_translator` | `npm run lesson3:mcp_translator` | File translation agent over `files-mcp` |
| `01_03_upload_mcp` | `npm run lesson3:upload_mcp` | Upload workspace files through MCP servers |

Install dependencies:

```bash
npm run lesson3:install
```

## Lesson 04

| Example | Run | Description |
|---------|-----|-------------|
| `01_04_audio` | `npm run lesson4:audio` | Audio transcription, analysis, and TTS with Gemini |
| `01_04_video` | `npm run lesson4:video` | Video analysis, transcription, and extraction with Gemini |
| `01_04_generation` | `npm run lesson4:generation` | Interactive video-processing example with Gemini and MCP tools |
| `01_04_video_generation` | `npm run lesson4:video_generation` | Frame-based video generation with Gemini and Kling |
| `01_04_reports` | `npm run lesson4:reports` | PDF reports from HTML, local assets, and generated images |
| `01_04_image_guidance` | `npm run lesson4:image_guidance` | Pose-guided image generation from JSON templates |
| `01_04_json_image` | `npm run lesson4:json_image` | Token-efficient image generation from JSON prompts |
| `01_04_image_editing` | `npm run lesson4:image_editing` | Iterative image generation and editing with quality checks |
| `01_04_sprites` | `npm run lesson4:sprites` | Parallel isometric sprite-set generation from JSON templates |
| `01_04_image_recognition` | `npm run lesson4:image_recognition` | Vision-based image classification with MCP file operations |

Install dependencies:

```bash
npm run lesson4:install
```

## Lesson 05

| Example | Run | Description |
|---------|-----|-------------|
| `01_05_confirmation` | `npm run lesson5:confirmation` | Email-sending agent with human-in-the-loop confirmation UI |
| `01_05_agent` | `npm run lesson5:agent` | API server for agent orchestration, tool execution, and MCP integration |

Install dependencies:

```bash
npm run lesson5:install
```

`01_05_agent` requires a one-time database setup before the first run:

```bash
npm run lesson5:agent:db:push
npm run lesson5:agent:db:seed
```

The agent server starts on `http://127.0.0.1:3000` by default. The seed script creates a bearer token `0f47acce-3aa7-4b58-9389-21b2940ecc70` for authentication. You can override the port and other settings via `01_05_agent/.env` (local keys take priority over the root `.env`).

## Lesson 06

| Example | Run | Description |
|---------|-----|-------------|
| `02_01_agentic_rag` | `npm run lesson6:agentic_rag` | Agentic RAG with multi-step retrieval and conversation history |

Install dependencies:

```bash
npm run lesson6:install
```

## Lesson 07

| Example | Run | Description |
|---------|-----|-------------|
| `02_02_chunking` | `npm run lesson7:chunking` | Four text chunking strategies compared side-by-side |
| `02_02_embedding` | `npm run lesson7:embedding` | Interactive embedding demo with a pairwise similarity matrix |
| `02_02_hybrid_rag` | `npm run lesson7:hybrid_rag` | Hybrid RAG agent with SQLite FTS5 full-text search and sqlite-vec vector similarity |

Install dependencies:

```bash
npm run lesson7:install
```
