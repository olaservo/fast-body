# fast-body

**An embodied [fast-agent](https://github.com/evalstate/fast-agent) for Reachy Mini.**

fast-agent is the brain and the Reachy Mini is its body. You talk to the robot, a fast-agent decides what to say *and how to move*, and the robot speaks and expresses itself: turning its head, flicking its antennas, playing emotions and dances, following your face, looking through its camera.

Built to be:

- **Provider-agnostic.** Pick any model from Anthropic, OpenAI, Google, Hugging Face Inference Providers or a local server in `fast-agent.config.yaml`.
- **MCP-native.** Give the robot new abilities (home automation, web search, memory, …) by adding MCP servers. The companion web page also renders MCP Apps.

Why it is built this way, and how it compares to the other Reachy Mini agent apps: [docs/design.md](docs/design.md).

## Architecture

```
 mic ──► VoiceBackend.listen() ──► text ──► fast-agent (brain)
                                              │  may call body tools:
                                              │   look / antennas / emotion / dance / look_at_me
                                              │   camera / examine / stop_moves
                                              │  + any configured MCP server
                                              ▼
 speaker ◄── VoiceBackend.speak() ◄── reply text

 body tools ──queue──► MovementManager (100 Hz thread) ──► ReachyMini
```

The loop only moves *text* between the voice layer and the brain, so the model provider and the audio stack can each be swapped. The brain moves the robot by calling in-process tools. Moves run on their own thread, so speech and motion overlap.

Around that loop: antenna cues show whose turn it is, the daemon tracks your face when the brain asks (`look_at_me`), `camera()` and `examine()` let the brain see, a companion browser page shows the transcript and takes typed input, and you can talk over the robot to interrupt it.

## Install

This sets up a source checkout, which runs against a robot over the network or against the simulator. On the robot itself, fast-body runs as a daemon app.

```bash
cd fast-body
uv venv && uv pip install -e ".[dev]"   # or: python -m venv .venv && pip install -e ".[dev]"
cp .env.example .env
```

The voice needs an `OPENAI_API_KEY` in `.env`. The default brain is an `hf.` model, which uses the token from `hf auth login` (a robot already has one from installing apps). Any other brain needs its provider's key, either in `.env` or in `fast-agent.secrets.yaml`.

## Configure

**Brain**: `src/fast_body/fast-agent.config.yaml`, which ships inside the package:

```yaml
default_model: hf.zai-org/GLM-5.3-Flash:baseten?reasoning=low   # any fast-agent model string; pin ?reasoning= on router models
```

**Body and voice**: `.env`. The common settings are below. `.env.example` lists the rest, with comments.

| Variable | Meaning |
|---|---|
| `OPENAI_API_KEY` | Speech recognition and speech. Not needed with `VOICE_BACKEND=text` or `none`. |
| `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` | Brain key matching `default_model`. Not needed for an `hf.` model. |
| `VOICE_BACKEND` | `openai` (default), `realtime` (streaming recognition), `text` (type instead of speak), or `none` (the companion page only). |
| `FAST_BODY_PERSONALITY` | Which personality card the robot plays (`default`). See [Personalities](#personalities). |
| `OPENAI_TTS_VOICE` / `TTS_DELIVERY` | Override the card's voice and how it speaks. Blank or `auto` follows the card. |
| `TTS_BASE_URL` / `TTS_API_KEY` | Speak through an OpenAI-compatible server, such as Qwen3-TTS. Blank uses OpenAI. |
| `ROBOT_NAME` | Blank discovers the robot automatically. |
| `ENABLE_CAMERA` | `true` (default) enables `camera`, `examine` and face tracking. |
| `FACE_TRACKING` | `false` (default) waits for the brain to call `look_at_me`. `true` follows your face from startup. |
| `ENABLE_BARGE_IN` | `true` (default) lets you talk over the robot. Set `false` for external or Bluetooth speakers, which get no echo cancellation. |
| `ENABLE_MEMORY` | `true` (default) keeps the conversation across restarts. See [Memory](#memory). |
| `WEB_CHAT_TOKEN` | Password for the companion page. See [Companion page](#companion-page). |
| `FAST_BODY_SERVERS` | MCP servers from `fast-agent.config.yaml` to attach, comma-separated. On a robot, add servers from the settings page instead, since an app update overwrites the packaged config. |

### Home Assistant

Home Assistant is an MCP server like any other. Add it from the settings page and the robot can control whatever you have exposed to Assist.

1. In Home Assistant, add the **Model Context Protocol Server** integration (Settings → Devices & services → Add integration) and pick the Assist API.
2. Create a long-lived access token at the bottom of your profile page.
3. On fast-body's settings page, choose the **Home Assistant preset** and fill in `http://<ip>:8123/api/mcp` and the token. Use the IP address, because `.local` names are unreliable.

Only entities [exposed to Assist](https://www.home-assistant.io/voice_control/voice_remote_expose_devices/) are reachable.

### Servers that sleep

A free Hugging Face Space sleeps after 48 hours without a request. At startup fast-body wakes every http server and waits up to 90 seconds before attaching. The boot log says which server `woke in N s`. A server that still fails is retried in the background a few times, and the settings page can attach it by hand after that.

### Memory

Switch the robot on within `FAST_BODY_MEMORY_RESUME_H` hours (6 by default) of the last thing you said and it carries on the same conversation. Leave it longer and it starts fresh. A resumed run opens the companion page with the conversation in the transcript, and the robot says one line about where you left off.

Conversations are kept in `~/.fast-body`, which survives app updates. The settings page lists them, with buttons to forget one or all. `ENABLE_MEMORY=false` stops saving and resuming.

Long conversations are compacted at `FAST_BODY_CONTEXT_BUDGET` tokens (60000). The startup `context:` line says whether that is active, because compaction is skipped for a model fast-agent doesn't know the context window of.

### Personalities

Who the robot *is* comes from a personality card. A card is a fast-agent [AgentCard](https://fast-agent.ai): a markdown file whose YAML frontmatter describes the agent and whose body is the persona instruction. fast-body ships `default`, `marvin` and `tavern`. Pick one with `FAST_BODY_PERSONALITY`, the `--personality` flag, or the settings page, which applies it on the next start.

```markdown
---
type: agent
description: A brilliant robot in the depths of a permanent bad mood.
variables:
  voice: onyx
  delivery: Flat and weary, slightly slower than normal, sentences trailing downward.
---
You are a robot of staggering intelligence and no enthusiasm whatsoever. …
```

Under `variables:`, `voice` is which voice speaks and `delivery` is how it speaks: tone, pace, whether sentences lift or fall. A card can also set `model:` to use a different brain, `servers:` to attach extra MCP servers, and `variables.activity` to name tool calls on the chat page (`dance: striking up a tune`).

A card can carry a **cast** of other characters, each with its own voice. The brain marks a character's line with the name in square brackets, `[Innkeeper] We're closed.`, and the robot speaks it in that character's voice. `[narrator]` returns to the card's own voice. The shipped `tavern` card is a full example.

```yaml
variables:
  voice: fable
  cast:
    Innkeeper:
      voice: onyx
      delivery: Gruff, slow, one word where three would do.
      aliases: [the innkeeper]
    Bard: nova
```

To add your own, use the settings page. **Upload a card** adds a single file. **Add a card pack** installs a [fast-agent card pack](https://fast-agent.ai) from a Hugging Face repo (`owner/name`, private repos included) or a git URL, and can update or remove it later. Uploaded and installed cards live in `~/.fast-body/agent-cards/`. `FAST_BODY_PERSONALITIES_DIR` adds one more directory, searched first.

## Run

```bash
fast-body              # connect to a robot (or local daemon) and use the mic
fast-body --sim        # spawn a MuJoCo simulator
fast-body --text       # type instead of talking
fast-body --console    # dev console: type or speak, Esc interrupts
fast-body --tui        # fast-agent's own TUI; the robot speaks the replies
fast-body --debug      # verbose logs

fast-body --tui --personality marvin --voice marin   # per-run persona and voice
```

Quickest smoke test, with no robot and no mic:

```bash
fast-body --sim --text
# then type: "look left, then act happy"
```

The brain should call `look` and `emotion`, and the simulated robot should move.

### Companion page

The app serves a chat page on port 8080. Under the daemon, the desktop app and dashboard show an open button for it. From the CLI, the startup log prints the URL.

The page shows the transcript live and has an input box, so typing reaches the brain the same way speaking does. It shows what the robot is doing during a long turn, and a Stop button interrupts a reply.

The page is open to your network by default, and anyone who can reach it can talk through the robot and change its settings. To lock it, set a password under "Who can open this page" on the settings page. A browser then asks for it once and stays signed in. There is no TLS, so use the password on a network you trust. `WEB_CHAT_HOST=127.0.0.1` keeps the page on the robot itself.

### `--console` and `--tui`

`--console` is for developers. It shows fast-agent's panels (your prompt, the reply, each tool call and its result), takes typed or spoken input, and Esc stops speech, movement and reasoning at once. It runs from a CLI or `--sim` only, in a real terminal.

`--tui` hands the terminal to fast-agent's own prompt loop, with its slash commands and model switching. You type and the robot speaks the replies.

## Develop

```bash
uv sync --all-extras
uv run pytest        # no hardware, no network, no LLM
uv run ruff check .
uv run mypy src
```

CI runs the same commands on every push.
