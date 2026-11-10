# fast-body

**An embodied [fast-agent](https://github.com/evalstate/fast-agent) for Reachy Mini.**

fast-agent is the brain and the Reachy Mini is its body. You talk to the robot, a fast-agent decides what to say *and how to move*, and the robot speaks and expresses itself: turning its head, flicking its antennas, playing emotions and dances, following your face, looking through its camera.

**[Watch the demo](https://www.youtube.com/watch?v=_3PSoIxU2Zo)** to see it in action.

Built to be:

- **Provider-agnostic.** Pick any model from Anthropic, OpenAI, Google, Hugging Face Inference Providers or a local server in `fast-agent.config.yaml`.
- **MCP-native.** Give the robot new abilities (home automation, web search, memory, …) by adding MCP servers. The companion web page also renders MCP Apps.
- **One brain.** The body controls are tools on the agent itself, so no second model decides what to say or drifts out of character, and nothing runs beside the robot.

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

Around that loop: antenna cues show whose turn it is, a companion browser page shows the transcript and takes typed input, and you can talk over the robot to interrupt it.

`look_at_me` switches on the daemon's face tracker, which blends the aim into whatever pose the brain commands, so eye contact holds through an emotion or a dance.

`camera()` hands the frame to the brain. `examine(question)` has a separate vision model answer in words, so the image never enters the conversation.

## Install

This sets up a source checkout, which runs against a robot over the network or against the simulator. On the robot itself, fast-body runs as a daemon app.

```bash
cd fast-body
uv venv && uv pip install -e ".[dev]"   # or: python -m venv .venv && pip install -e ".[dev]"
cp .env.example .env
```

The voice needs an `OPENAI_API_KEY` in `.env`. The default brain is an `hf.` model, which uses the token from `hf auth login` (a robot already has one from installing apps). Any other brain needs its provider's key, either in `.env` or in `fast-agent.secrets.yaml`.

### Deploy to the robot

The robot's daemon installs apps from Hugging Face Spaces, so a deploy is a Space holding this repo. A private Space works.

1. Create a Space named `fast_body`, the same as the app's entry point, with the `static` SDK, and push this repo to it with this frontmatter at the top of `README.md`:

   ```yaml
   ---
   title: fast-body
   sdk: static
   tags:
     - reachy_mini
     - reachy_mini_python_app
   ---
   ```

2. Leave `tests/` and `scripts/` out of the Space, or its secret scanner may refuse the push.

3. Install it from the Reachy Mini app, which lists private Spaces too once the robot is logged in to Hugging Face.

4. Set `OPENAI_API_KEY` and any brain key on the settings page. They are kept across app updates.

Later versions go out with `POST /api/apps/update/fast_body`, or the dashboard's update button.

## Configure

**Brain**: `src/fast_body/fast-agent.config.yaml`, which ships inside the package:

```yaml
default_model: hf.zai-org/GLM-5.3-Flash:baseten?reasoning=low   # any fast-agent model string; pin ?reasoning= on router models
```

**Body and voice**: `.env`. The common settings are below. `.env.example` lists the rest, with comments.

| Variable | Meaning |
|---|---|
| `OPENAI_API_KEY` | Speech recognition and speech. Not needed with `VOICE_BACKEND=text` or `none`. |
| `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` | Brain key matching `default_model`. Not needed for an `hf.` model, which uses the robot's Hugging Face login: the dashboard sign-in, or `HF_TOKEN` if set. |
| `VOICE_BACKEND` | `openai` (default), `realtime` (streaming recognition), `text` (type instead of speak), or `none` (the companion page only). |
| `FAST_BODY_PERSONALITY` | Which personality card the robot plays (`default`). See [Personalities](#personalities). |
| `OPENAI_TTS_VOICE` / `TTS_DELIVERY` | Override the card's voice and how it speaks. Blank or `auto` follows the card. |
| `TTS_BASE_URL` / `TTS_API_KEY` | Speak through an OpenAI-compatible server, such as Qwen3-TTS. Blank uses OpenAI. |
| `ROBOT_NAME` | Blank discovers the robot automatically. |
| `ENABLE_CAMERA` | `true` (default) enables `camera`, `examine` and face tracking. |
| `FACE_TRACKING` | `false` (default) waits for the brain to call `look_at_me`. `true` follows your face from startup. |
| `ENABLE_BARGE_IN` | `true` (default) lets you talk over the robot. Set `false` for external or Bluetooth speakers, which get no echo cancellation. |
| `LISTEN_MODE` | `always_on` (default) keeps the mic live. `push_to_talk` closes it until you hold an antenna or the companion page's Talk button, and letting go sends what you said. Needs `VOICE_BACKEND=realtime`. |
| `ENABLE_MEMORY` | `true` (default) keeps the conversation across restarts. See [Memory](#memory). |
| `WEB_CHAT_TOKEN` | Password for the companion page. See [Companion page](#companion-page). |
| `FAST_BODY_SERVERS` | MCP servers from `fast-agent.config.yaml` to attach, comma-separated. On a robot, add servers from the settings page instead, since an app update overwrites the packaged config. |

### MCP servers

Add servers from the settings page, which has a preset for Home Assistant's Model Context Protocol Server integration. Servers added there survive app updates.

### Memory

Switch the robot on within `FAST_BODY_MEMORY_RESUME_H` hours (6 by default) of the last thing you said and it carries on the same conversation, repeating its last words to show where it left off. Leave it longer and it starts fresh.

Conversations are kept in `~/.fast-body`, which survives app updates. The settings page lists them, with buttons to forget one or all. `ENABLE_MEMORY=false` stops saving and resuming.

Long conversations are compacted at `FAST_BODY_CONTEXT_BUDGET` tokens (60000). The startup `context:` line says whether that is active.

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

The page is also an [MCP Apps](https://modelcontextprotocol.io/docs/extensions/apps) host. When the brain calls a tool that an MCP server pairs with a `ui://` resource, the page draws that resource's HTML, opening it as the call starts and filling it with the result. A widget can call tools itself, and the text of those results reaches the conversation so the brain hears about it. A widget can also send a message to the brain as if you had typed it, and open a web link in a new tab.

### Push-to-talk

By default the mic is always live and the server decides when you have finished a sentence. `LISTEN_MODE=push_to_talk` (also on the settings page) closes the mic until you ask for it: hold either antenna and speak, and letting go sends what you said. The companion page gets a Talk button that does the same, and a short tap on it keeps the mic open until the next tap. Opening the mic also cuts off a reply in progress. A hold with no speech in it, or shorter than a tenth of a second, sends nothing.

It needs `VOICE_BACKEND=realtime`, which commits the audio on release; with the default `openai` backend the setting is ignored with a warning. The antenna hold is detected on the difference between where the robot put its antennas and where they are, so its own breathing and gestures do not count; `PTT_ANTENNA_PRESS_RAD` and `PTT_ANTENNA_RELEASE_RAD` tune it, and `PTT_ANTENNA=false` leaves only the button.

The page is open to your network by default, and anyone who can reach it can talk through the robot and change its settings. To lock it, set a password under "Who can open this page" on the settings page. A browser then asks for it once and stays signed in. There is no TLS, so use the password on a network you trust. `WEB_CHAT_HOST=127.0.0.1` keeps the page on the robot itself.

### Drive it from a JS app

Everything the companion page does goes over one JSON-RPC 2.0 WebSocket at `/rpc` on port 8080, the socket the Reachy Mini daemon relays to the running app. A JS app using the Reachy Mini SDK reaches it from a phone or browser over WebRTC with `reachy.rpcCall(method, params)` and `reachy.onNotification(method, cb)`. On the LAN you can also open `ws://<robot>:8080/rpc` directly. The method names follow the conversation app's where the meaning is the same, so the Conversation panel in Pollen's `sdk-js-demo-app` drives fast-body once its `CONV_APP_NAME` is set to `fast_body`.

Methods:

| Method | Params | Result |
|---|---|---|
| `conversation.status` | | `ready`, `backend_connected`, `backend_error`, `errors`, `locked`, `personality`, `voice`, `listen_mode` |
| `conversation.say` | `text` | sent as your next turn, as if typed |
| `conversation.interrupt` | | stops the reply in progress |
| `conversation.history` | | `events`: the last 60 transcript entries, as the notifications that carried them |
| `conversation.mic` | `muted?` | `muted`; setting it needs push-to-talk |
| `listen.state`, `listen.set` | `held?`, `latched?`, `toggle?` | the Talk button's state; `null` without push-to-talk |
| `personalities.list`, `personalities.apply` | `name` | `apply` saves the card and returns `restart_required` |
| `personalities.upload`, `.delete`, `.packs.lookup`, `.packs.install`, `.packs.update`, `.packs.remove` | as the settings page | |
| `voices.list`, `voices.current`, `voices.apply` | `voice` | `apply` pins the voice and returns `restart_required` |
| `servers.list`, `.save`, `.toggle`, `.delete` | as the settings page | MCP servers, attached live when the brain is up |
| `memory.list`, `memory.forget` | `name` or `all` | |
| `choices.answer` | `id`, `option`, `elaboration?` | answers a question a server asked |
| `widgets.resource`, `widgets.call` | `server`, `uri` / `server`, `name`, `arguments` | an MCP App's HTML, and its own tool calls |

Notifications: `conversation.transcript` (`role` is `user` or `assistant`), `conversation.turn` (`listening`, `thinking`, `speaking`, `ready`), `conversation.activity` (a tool step, with the label the card allows), `conversation.history_cleared`, `widgets.call` and `widgets.result`, `choices.ask` and `choices.answered`, `listen.state`. Each carries a `seq`, so a client that fetched `conversation.history` can drop the notifications it already covered. A refused request has a stable string in `error.data.reason`, such as `invalid_params`, `not_found` or `not_running`.

Personality and voice are read when the conversation starts, so the `apply` methods save the setting for the next start. The daemon handles the `apps.*` namespace itself (start, stop, status) and never forwards it, so fast-body has no methods there.

Access follows the page's password. The daemon's relay connects from the robot itself with no token and no `Origin` header, and is let in because the daemon has already authenticated the remote end. Any other client needs the password as `?token=` or the page's cookie, and a browser must come from the page's own origin.

The relay carries every JS app the robot's owner opens, so it may not call the methods that change what runs on the robot: `servers.save`, `servers.toggle`, `servers.delete`, `personalities.upload`, `personalities.delete` and `personalities.packs.install` / `update` / `remove`. Those answer with reason `forbidden` and stay available on the chat page and to clients on the network.

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
