# Design notes

Why fast-body is built the way it is. The [README](../README.md) says how to install, configure and run it; this is the reasoning behind the parts that are not obvious from the config table.

## Conversational polish

Borrowed from `reachy_mini_conversation_app`, adapted to the turn-based loop:

- **State cues** — antennas show the turn: forward/attentive while *listening* (idle breathing pauses so the robot holds still), curious while *thinking*, neutral while *speaking*.
- **Face tracking** — the brain calls `look_at_me(true)` to follow your face with its head. The *daemon* runs the detector and blends the look-at aim into whatever pose we command (reachy-mini 1.9.0+), so this process spends nothing per frame and eye contact survives *through* emotions and dances. `HEAD_TRACKING_WEIGHT` sets how strongly it tracks (1.0 = the head is fully on you, lower lets the active move show through); while the robot speaks it eases to `HEAD_TRACKING_SPEAKING_WEIGHT` so its own expression reads. Gated on `ENABLE_CAMERA`; a no-op if the robot has no daemon-side tracking.
- **Audio-reactive wobble** — `enable_wobbling()` lets the daemon sway the head in time with the robot's own speech (composed on top of our poses). On-robot LOCAL audio backend only; a harmless no-op elsewhere.
- **Sight** — `camera()` hands the frame straight to the brain as image content, so the model looks at it and says what it sees in its own voice. Needs a vision-capable model; the default is one. `examine(question)` is the other way to look: a small vision model reads the frame and answers one question in words, so the picture never enters the brain's history. Use it for a fact from the scene.
- **Companion browser page** — the app serves a live transcript at `http://<robot>:8080/` that you can also type into, so a typed line and a spoken one reach the brain on the same path. This is the only way to talk to the robot while the daemon runs it headless: the daemon gives apps no terminal. Works the same from the CLI. See [Companion page](../README.md#companion-page) for the access controls.
- **Turn-latency logging** — each turn logs `user done → speaking` ms, with the turn's model calls and tokens (prompt, cached, completion, tool calls), so a slow turn says where it went.
- **Working, visibly** — while a reply is on its way the chat page shows that the robot is thinking, for how long, and a step per tool call, and one antenna flicks per call. Activity, not content: a step is named only when the card allows it (see [Personalities](../README.md#personalities)); arguments and results stay on the robot.
- **Barge-in** *(on by default)* — talk over the robot and it stops (flushes playback via `clear_player`), and the mic stays live while it speaks so you don't have to wait for it to finish. This works because the SDK builds `webrtcdsp` + `webrtcechoprobe` into the robot's LOCAL audio path, so the open mic doesn't hear the robot itself. That cancellation does **not** follow you to external or Bluetooth speakers — the echo probe's far-end reference stops matching what is played — so set `ENABLE_BARGE_IN=false` there. The mic is then muted for each reply instead, which means it can't hear you until the robot stops talking.
- **Streaming recognition** *(`VOICE_BACKEND=realtime`)* — listening moves to an OpenAI transcription session, so the server endpoints the speech and the local VAD, the length and level gates and the phrase list all drop out. There is no model in that session: it carries audio up and transcripts down, and the brain stays fast-agent's. Speaking is unchanged. `REALTIME_NOISE_REDUCTION` filters the mic ahead of the server's VAD, which is the lever the discrete path never had — it could only reject a clip after the fact. The model is `gpt-transcribe`; `gpt-live-transcribe` streams its transcript sooner but rejects turn detection, which would put endpointing back on us.

First-token audio still belongs to a speech-to-speech session, which would make the voice model the talker. That is the trade fast-body declines.

## How this compares

There are a couple of dozen Reachy Mini apps that give a body to an agent that already exists somewhere else. Nearly all share one shape: a realtime voice API runs the conversation and the real agent is reached through a single tool call, on a machine beside the robot. fast-body differs on five points.

- **One brain talks and moves.** Body controls are tools registered in-process on the agent, not a bridge to a second model. Nothing else is deciding what to say while the agent is only consulted, and there is no shell model to drift out of character.
- **New abilities are MCP servers.** Home automation, search, memory: add a server from the settings page, not bridge code. Every comparable app is wired to one agent product and extends through that product.
- **Any model.** The brain is `default_model` in a config file; the body doesn't know which provider is reasoning. The others are welded to one vendor's agent or one vendor's realtime API.
- **No companion machine, no keys.** It installs as a daemon app on the robot and runs against Hugging Face Inference using the token the robot already holds for installing apps. The comparable apps assume a laptop running a CLI, a gateway daemon, or Docker beside the robot.
- **Face tracking is the daemon's job.** `look_at_me` switches on daemon-side tracking, which blends the aim into whatever pose we command. Eye contact survives *through* an emotion or a dance, and this process spends nothing per frame. Apps that run their own detector pay for it continuously and lose the gaze whenever a move takes the head.

Where it's behind is in the [roadmap](../README.md#roadmap): first-token audio, which the realtime-API apps get from a speech-to-speech session.

## Servers that sleep

A free Hugging Face Space sleeps after 48 hours without a request and takes tens of seconds to come back, longer than the 15 seconds an attach waits. While the app runs, its open MCP session keeps a Space awake; the gap is a start after a long stop. So at startup fast-body first sends one request to every enabled http server at once, which starts a sleeping Space, and waits up to 90 seconds for the slowest before attaching. What the boot line and the retries look like is in the README's [Servers that sleep](../README.md#servers-that-sleep).

## Memory

The settings are in the README's [Memory](../README.md#memory); the policy behind them is here.

Switch the robot on within `FAST_BODY_MEMORY_RESUME_H` hours of the last thing you said and it carries on the same conversation — it still knows the gate code you gave it this morning. Leave it longer and it starts fresh. Six hours is a default chosen for how a robot is actually used: it carries a morning into an afternoon and lets last week go.

This policy is fast-body's, not fast-agent's. Upstream never resumes unless you pass `--resume`, and then takes the most recent session with no age check — which works because a person is at a terminal deciding per run. A robot has nobody to ask, so it needs a standing rule. Retention *is* fast-agent's: the session manager keeps a rolling 20 (`session_history_window`) and drops the rest as new ones are created, so the store does not grow without bound and the forget buttons are for deleting on purpose.

The store is `~/.fast-body`, not the installed package. An app *remove* wipes the package directory, and memory that dies on an update is worse than none because it looks like it works. The same directory holds skills and personality cards.

What was written down is listed on the settings page, with a button to forget one conversation or all of them. This is a microphone in a room writing down what it hears; it should not take an SSH session to see or delete that. `ENABLE_MEMORY=false` stops the saving and the resuming, and leaves earlier recordings listed so you can still delete them.

**Picking a conversation back up.** fast-agent's CLI prints the last assistant message when you pass `--resume`, so you can see what you are rejoining. A robot has no terminal, so it does the same two things in its own way: the companion page opens with the resumed conversation already in the transcript, and the robot speaks one line referring to where you left off before it starts listening. Without either, a resumed run looks exactly like a forgetful one and you have to take the memory on faith.

That line quotes the robot's own last words rather than asking the brain to compose a greeting. A generated one would cost a turn before anyone had spoken, and `brain.send()` takes a *user* message — so producing it would write a line you never said into the very history it is recalling. `FAST_BODY_MEMORY_GREETING=false` keeps the page seeding and drops the spoken line.

**Bounding the context.** Every turn replays the conversation so far, so an unbounded transcript gets slow and expensive before it gets anywhere near overflowing. fast-agent compacts automatically, but on a *fraction* of the model's context window — the wrong unit here, since the router models report 1M tokens and the 0.85 default would first compact at ~891k. `FAST_BODY_CONTEXT_BUDGET` is an absolute token count instead, converted to that fraction at startup once the window is known, so swapping the brain doesn't silently change what you set.

A model fast-agent's `ModelDatabase` doesn't know reports no window at all, and compaction is then skipped entirely — without saying so. The startup `context:` line reports it either way, so check that before assuming a long conversation is being managed.
