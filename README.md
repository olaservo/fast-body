# fast-body

**An embodied [fast-agent](https://github.com/evalstate/fast-agent) for Reachy Mini.**

fast-agent is the brain; the Reachy Mini is its body. You talk to the robot, a fast-agent reasons about what to say *and how to move*, and the robot speaks and expresses itself physically — turning its head, flicking its antennas, playing emotions and dances, following your face, looking through its camera.

Built to be:

- **Provider-agnostic** — pick any model and a variety of providers (Anthropic, OpenAI, Google, Hugging Face providers, local) in `fast-agent.config.yaml`.
- **MCP-native** — give the robot new abilities (home automation, web search, memory, …) by adding MCP servers to connect skills and tools.  The companion web UI also renders MCP Apps.

Why it is built this way, and how it compares to the other Reachy Mini agent apps: [docs/design.md](docs/design.md).

## Architecture

```
 mic ──► VoiceBackend.listen() ──► text ──► fast-agent (brain)
                                              │  may call body tools:
                                              │   look / antennas / emotion / dance / look_at_me / camera / stop_moves
                                              │  + any configured MCP server
                                              ▼
 speaker ◄── VoiceBackend.speak() ◄── reply text

 body tools ──queue──► MovementManager (100 Hz thread) ──► ReachyMini
```

The conversation loop only ever moves *text* between the voice layer and the brain — that's what keeps both the model provider and the audio stack swappable. The brain decides when to move by calling in-process tools; moves run on their own thread so speech and motion overlap.

Around that loop: antenna cues show whose turn it is, the daemon tracks your face when the brain asks (`look_at_me`), `camera()` and `examine()` let the brain see, a companion browser page carries the transcript and takes typed input, and you can talk over the robot to interrupt it on its built-in speaker. Each of these is described in [docs/design.md](docs/design.md#conversational-polish).

## Install

```bash
cd fast-body
uv venv && uv pip install -e ".[dev]"   # or: python -m venv .venv && pip install -e ".[dev]"
cp .env.example .env                     # then fill in OPENAI_API_KEY (a brain key only if default_model is not an hf. model)
```

`reachy-mini` (the SDK) and `fast-agent-mcp` (the brain) install as normal dependencies.

## Configure

**Brain** — `src/fast_body/fast-agent.config.yaml` (inside the package, so it ships with an installed app):

```yaml
default_model: hf.zai-org/GLM-5.3-Flash:baseten?reasoning=low   # any fast-agent model string; pin ?reasoning= on router models
mcp:
  servers:                       # add MCP servers here, then list them in FAST_BODY_SERVERS
    # none shipped: Home Assistant and the like are added from the settings page
```

**Body / voice** — `.env` (see `.env.example`):

| Variable | Meaning |
|---|---|
| `OPENAI_API_KEY` | Voice STT + TTS. Not needed with `VOICE_BACKEND=text` or `none`. |
| `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` | Brain key matching `default_model`. Not needed for an `hf.` model — fast-agent uses the token `hf auth login` stores, which a robot already has. |
| `VOICE_BACKEND` | `openai` (default), `realtime` (streaming recognition, same TTS), `text` (type instead of speak), or `none` (no audio; the companion page is the only way in and out). |
| `TTS_DELIVERY` | How to speak — tone, pace, emotion — overriding the card's `delivery`. Blank follows the card. Not accepted by `tts-1` / `tts-1-hd`. |
| `FACE_TRACKING` | Follow the person's face from startup (off). Shares the camera with `camera()` and moves the head while a still is taken, so it waits to be asked — the brain can still turn it on with `look_at_me`. Settable from the settings page. |
| `TTS_BASE_URL` / `TTS_API_KEY` | Speak through an OpenAI-compatible server instead of OpenAI (a Qwen3-TTS server exposes `/v1/audio/speech`). Blank = OpenAI; the key defaults to `OPENAI_API_KEY`. |
| `TTS_LEAD_IN_S` | Speech buffered before playback starts (0.6). The reply streams in, so too small a lead-in drains the player mid-word; too large delays the first sound. |
| `TTS_WARMUP` | `true` (default) — one short synthesis at startup, discarded, run while the brain attaches its servers. The first reply after a boot took 19–20 s to sound on the robot and 1–2 s from then on; this pays that before anyone is waiting. |
| `BRAIN_TIMEOUT_S` | Ceiling on one brain turn, tool calls included (90). A bound on a stalled provider, not a latency budget — a turn that hits it is dropped with an apology rather than holding the conversation open. |
| `FAST_BODY_SERVERS` | Comma-separated MCP servers defined in `fast-agent.config.yaml` to attach. For a source checkout; on a robot, add servers from the settings page, since an app update overwrites the packaged config. |
| `FAST_BODY_PERSONALITY` | Which personality card the robot plays (`default`). See [Personalities](#personalities). |
| `FAST_BODY_PERSONALITIES_DIR` | Extra directory of your own cards, searched before `~/.fast-body/personalities/` and the built-ins; same-named cards override. |
| `OPENAI_TTS_VOICE` | `auto` (default) — the personality card's voice, falling back to `cedar`. Set a voice name to pin it. |
| `ROBOT_NAME` | Blank = auto-discover. |
| `ENABLE_CAMERA` | `true` (default) — enables the `camera` tool and `look_at_me` face tracking. |
| `FAST_BODY_VISION_MODEL` | The OpenAI model behind `examine(question)` (`gpt-5-mini`). Uses `OPENAI_API_KEY`; without one the tool says it is not configured. |
| `ENABLE_WEB_CHAT` | `true` (default) — serve the companion browser page. Don't disable it for a daemon-installed app: the dashboard sees the advertised page never answer and stops the app after 60 s. |
| `WEB_CHAT_PORT` / `WEB_CHAT_HOST` | `8080` / `0.0.0.0`. Use `127.0.0.1` to keep the page on the robot. Changing the port breaks the dashboard's open button, whose URL is a static literal in `main.py`. |
| `WEB_CHAT_TOKEN` | Blank (default) — the page is open to your network. Set a value (8+ characters), or set it from the settings page, and the page asks for it once per browser and keeps the browser signed in with a cookie, so the dashboard's open button still works. |
| `HEAD_TRACKING_WEIGHT` | `1.0` (default) — how strongly face tracking owns the head. |
| `HEAD_TRACKING_SPEAKING_WEIGHT` | `0.3` (default) — the same, while the robot is speaking. |
| `ENABLE_WOBBLE` | `true` (default) — audio-reactive head sway while speaking. |
| `ENABLE_BARGE_IN` | `true` (default) — talk over the robot to interrupt it, mic live while it speaks. Set `false` for external or Bluetooth speakers, which get no echo cancellation. |
| `SPEAK_BETWEEN_TOOLS` | `true` (default) — speak a turn's text as it arrives, ahead of each tool call, so narration written before a tool call is heard before it. `false` holds all of it until after the last tool call. |
| `ENABLE_MEMORY` | `true` (default) — keep the conversation across restarts. See [Memory](#memory). Settable from the settings page. |
| `FAST_BODY_MEMORY_RESUME_H` | `6` (default) — how stale the last conversation may be and still be resumed, in hours. `0` saves without ever resuming. |
| `FAST_BODY_MEMORY_DIR` | `~/.fast-body` (default) — where saved conversations, skills and cards live. |
| `FAST_BODY_MEMORY_GREETING` | `true` (default) — on a resumed run, say a line referring to where you left off instead of carrying on mid-thought. |
| `FAST_BODY_CONTEXT_BUDGET` | `60000` (default) — tokens of conversation replayed to the brain before it is compacted. `0` uses fast-agent's own threshold. |
| `FAST_BODY_SKILLS` | Extra skill directories, separated by the platform's path separator. Skills served by attached MCP servers are installed under `~/.fast-body/skills/` without this. |
| `CHOICE_TIMEOUT_S` | `300` (default) — how long a server's question (MCP elicitation) stays open for an answer by voice or on the page before it is cancelled. |

Speech-gating and realtime-backend tuning (`STT_*`, `VAD_*`, `REALTIME_*`) are documented in `.env.example`.

Brain provider keys can also live in `fast-agent.secrets.yaml` (see the `.example`); the voice still reads `OPENAI_API_KEY` from the environment.

### Home Assistant

Home Assistant is an MCP server like any other: add it from the settings page and the robot can control the house — lights, scenes, whatever you have exposed to Assist. The page's **Home Assistant preset** fills the form: name `home-assistant`, transport http, URL `http://<ip>:8123/api/mcp`, and a long-lived access token as the bearer. It is stored in `~/.fast-body/mcp_servers.yaml` with the other servers, which an app update leaves alone, and attaches once the brain is up.

In Home Assistant, add the **Model Context Protocol Server** integration (Settings → Devices & services → Add integration) and pick the LLM API it exposes, normally Assist. Only entities [exposed to Assist](https://www.home-assistant.io/voice_control/voice_remote_expose_devices/) are reachable. Create the token at the bottom of your HA profile page. Use HA's IP rather than a `.local` name — mDNS is unreliable enough to cost you a startup. The integration serves Streamable HTTP at `<url>/api/mcp`; behind a proxy, enter whatever the full endpoint is.

### Servers that sleep

A free Hugging Face Space sleeps after 48 hours without a request and takes tens of seconds to come back. At startup fast-body first sends one request to every enabled http server at once, which starts a sleeping Space, and waits up to 90 seconds for the slowest before attaching; the boot line then says which server `woke in N s`. A server that still fails to attach is retried in the background at growing intervals (about 20, 40 and 80 seconds later), with the skills sync run again once it makes it, and the log says when it gives up. The settings page can attach it by hand after that. Why it works this way: [docs/design.md](docs/design.md#servers-that-sleep).

### Memory

The robot keeps what was said. fast-agent saves the conversation after every turn and can hydrate one back; fast-body pins where that goes and decides when to pick it up.

Switch the robot on within `FAST_BODY_MEMORY_RESUME_H` hours of the last thing you said and it carries on the same conversation — it still knows the gate code you gave it this morning. Leave it longer and it starts fresh. On a resumed run the companion page opens with the conversation already in the transcript, and the robot speaks one line referring to where you left off before it starts listening; `FAST_BODY_MEMORY_GREETING=false` keeps the page seeding and drops the spoken line.

The store is `~/.fast-body`, not the installed package, so an app update or remove leaves it alone. The same directory holds skills and personality cards. What was written down is listed on the settings page, with a button to forget one conversation or all of them. `ENABLE_MEMORY=false` stops the saving and the resuming, and leaves earlier recordings listed so you can still delete them.

Every turn replays the conversation so far, and `FAST_BODY_CONTEXT_BUDGET` is the absolute token count at which it is compacted. A model fast-agent's `ModelDatabase` doesn't know reports no window at all, and compaction is then skipped entirely — without saying so. The startup `context:` line reports it either way, so check that before assuming a long conversation is being managed. The resume policy, retention, and why the budget is a token count: [docs/design.md](docs/design.md#memory).

### Personalities

Who the robot *is* — as opposed to what its body can do — comes from a personality card. A card is a fast-agent [AgentCard](https://fast-agent.ai): a markdown file whose YAML frontmatter describes the agent and whose body is the persona instruction. `src/fast_body/personalities/` ships `default`, `marvin` and `tavern`; pick one with `FAST_BODY_PERSONALITY`, the `--personality` CLI flag, or the dropdown on the settings page (which applies on next app start).

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

A card may also set `model:` (overriding `default_model` for that persona), `servers:` (extra MCP servers from `fast-agent.config.yaml`), and a speaking voice under `variables:` — a field the AgentCard schema reserves for app-defined data. A card sets two separate things about the sound: `voice` is *which* voice, and `delivery` is *how* to speak — tone, pace, whether sentences lift or fall. Delivery is passed to the TTS model as `instructions`, so a card carries character in the sound and not only in the words; the older `tts-1` models reject the field, so it is only sent when a card sets one. Both resolve the same way — `OPENAI_TTS_VOICE` / `TTS_DELIVERY` override → card → default — and the settings page has pickers for the voice and the persona (with `auto` meaning "follow the card").

A card can also say what the chat page may call a tool call while it runs, under `variables.activity`: a map from tool name to a short label, `dance: striking up a tune`.

A card can also carry a **cast**: other characters the robot voices, each with a voice and a delivery, for a persona that narrates a scene. The brain marks a character's line with the name in square brackets, `[Innkeeper] We're closed.`, and the voice layer speaks each stretch of the reply in the right voice, in order; `[narrator]` (or the card's own name) returns to the card's voice. Names match loosely and can have aliases; an unknown name is spoken in the card's voice with a logged warning. The rule is added to the instruction automatically when a card has a cast, so the card itself only lists the characters.

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

The shipped `tavern` card is that example in full: a storyteller who voices an innkeeper and a bard, and a starting point for a card of your own.

Write your own and add it from the settings page: **Upload a card** stores the file in `~/.fast-body/agent-cards/` on the robot, and **Add a card pack** installs a [fast-agent card pack](https://fast-agent.ai) — a repository holding `card-pack.yaml` directories under `packs/`, listed by a `marketplace.json` — through fast-agent's own pack manager, so the same repo installs on a laptop with `fast-agent cards add`. Write a Hugging Face repo as `owner/name` (read with the robot's own Hugging Face login, so a private repo works the way a private Space does for an MCP server), or give a git repository or registry URL; Look up shows what it offers, Install brings a pack's cards in, Update pulls a new revision without overwriting a card edited on the robot unless asked, Remove takes the pack and its cards out. A pack may only bring cards: one that also wants to install tool cards or files is refused. Cards that should not ship with the app (such as a persona tied to private content or to one household's servers) belong in a repository of their own, installed as a pack.

## Run

```bash
fast-body              # connect to a robot (or local daemon) and use the mic
fast-body --sim        # spawn a MuJoCo simulator
fast-body --text       # type instead of talking — no audio or robot mic needed
fast-body --console    # interactive dev console (see below)
fast-body --tui        # fast-agent's own TUI; the robot speaks the replies
fast-body --debug      # verbose logs

fast-body --tui --personality marvin --voice marin   # per-run persona/voice overrides
```

### Companion page

The app serves a chat page on `WEB_CHAT_PORT` (8080) in both run modes. Under the daemon, the desktop app and dashboard show an open button for it while the app is running (the daemon scrapes the URL from the `custom_app_url` literal in `main.py`). From the CLI, the startup log prints the URL:

```
chat page: http://localhost:8080/
```

You get the transcript live — what the mic heard and what the robot replied — and an input box that reaches the brain on the same path as speech, plus a Stop button that interrupts a reply the way Escape does in the console. Between the two, an activity line says the robot is thinking, for how long, and which step it is on (named only when the card allows it; see [Personalities](#personalities)), so a turn that is six tool calls and a minute long reads as working rather than stuck. On the robot, swap `localhost` for its address. `VOICE_BACKEND=none` turns off audio entirely and makes the page the only way in and out, which is the setup for a robot with no usable mic.

The page is open to your network by default, like the daemon on port 8000. Anyone who can reach it can talk through the robot and change its settings, including which MCP servers it uses. To lock it, set a password under "Who can open this page" on the settings page (or `WEB_CHAT_TOKEN` in the `.env`). From then on a browser opening the page is asked for the password once and stays signed in, so the dashboard's open button keeps working; a URL with `?token=` (the one the CLI logs) also signs the browser in. Changing the password signs every browser out, and the settings page can remove it or sign out the browser you are on.

There's no TLS and no user accounts, so the password crosses your network in the clear once per browser. Treat it as a lock for a network you trust, not for the internet; `WEB_CHAT_HOST=127.0.0.1` keeps the page on the robot itself.

### fast-agent's TUI (`--tui`)

Hands the terminal to fast-agent's own prompt loop instead of running our turn loop, against a real connected robot. The body tools are registered on the agent either way, so anything you type can move the robot, and you get fast-agent's slash commands, model switching and tool trace for free.

You type; **the robot speaks the replies aloud**, with the antenna cues it uses in a spoken turn. The prompt returns once it has finished talking, so the transcript and the audio stay in step.

Speech needs `OPENAI_API_KEY` like the normal loop. Without one the TUI runs silent rather than failing, so with an `hf.` model it still works with no keys at all — you just read the replies instead of hearing them.

### Interactive dev console (`--console`)

A developer front-end for watching and driving the brain from a shell (CLI/`--sim` only — the headless daemon never enables it). It does three things at once:

- **Un-blinds the brain.** fast-agent renders its own panels — your prompt, the assistant's reply, every tool call + arguments, and tool results — instead of staying silent.
- **Type *or* speak.** A persistent `you>` prompt accepts typed text, while the mic stays live; whichever comes first is sent. Typed input rides the exact same path as speech, and the robot still speaks and moves the reply aloud.
- **Escape = interrupt.** Press <kbd>Esc</kbd> any time to cut the robot off — speech, movement, *and* in-flight reasoning all stop and you're back at the prompt. (Keyboard, not acoustic: deterministic and needs no echo cancellation, unlike `ENABLE_BARGE_IN`.)

```bash
fast-body --sim --console
# type or say "do a little dance", then hit Esc partway through to cut it off
```

Needs `OPENAI_API_KEY` (voice). The default `hf.` model needs no other key; a different `default_model` needs its provider's key (e.g. `ANTHROPIC_API_KEY` for `sonnet`). `--console` implies the mic+TTS backend, so it takes precedence over `--text`.

Run it from a real interactive terminal (Windows Terminal, PowerShell, a TTY since prompt_toolkit needs a real TTY. fast-body's own INFO log lines are quieted in console mode so they don't step on the prompt; add `--debug` if you want them back. There is a brief (under about 0.5 s) delay before Esc registers, while prompt_toolkit disambiguates it from arrow and Alt sequences, and an Esc at an idle prompt is a no-op. Each listen turn races the mic against the prompt for up to about 30 s of mic idle, then re-listens.

### Quickest smoke test (no robot, no mic)

```bash
fast-body --sim --text
# then type: "look left, then act happy"
```

You should see the brain call the `look` and `emotion` tools and the simulated robot move.

## Develop

```bash
uv sync --all-extras
uv run pytest        # no hardware, no network, no LLM
uv run ruff check .
uv run mypy src
```

CI runs the same three commands on every push.
