# Design notes

Why fast-body is built the way it is. The [README](../README.md) covers installing, configuring and running it.

## How it differs

There are a couple of dozen Reachy Mini apps that give a body to an agent that lives somewhere else. Most of them run the conversation on a realtime voice API and reach the real agent through one tool call, from a machine beside the robot. fast-body makes different choices:

- **One brain talks and moves.** The body controls are tools on the agent itself. No second model decides what to say or drifts out of character.
- **New abilities are MCP servers.** Home automation, search and memory are servers you add from the settings page. The comparable apps extend through the one agent product they are wired to.
- **Any model.** The brain is `default_model` in a config file, and the body doesn't know which provider is reasoning.
- **Nothing beside the robot.** It installs as a daemon app and runs the brain on Hugging Face Inference, using the token the robot already has for installing apps. The voice still needs an OpenAI key.
- **Face tracking is the daemon's job.** `look_at_me` switches on the daemon's tracker, which blends the aim into whatever pose we command. Eye contact holds through an emotion or a dance, and the app spends nothing per frame.

What it gives up is first-token audio. The realtime-API apps get that from a speech-to-speech session, which makes the voice model the talker. fast-body keeps the agent as the talker.

## Talking over the robot

Barge-in is on by default, so the mic stays live while the robot speaks. This works because the SDK cancels the robot's own echo on its built-in speaker. The cancellation does not follow the audio to an external or Bluetooth speaker, so set `ENABLE_BARGE_IN=false` there. The mic is then muted during each reply.

## Two ways to look

`camera()` hands the frame to the brain, which describes it in its own voice. `examine(question)` has a separate vision model read the frame and answer in words. The image never enters the conversation, which keeps the history small when the brain only needs a fact, like the number on a die.

## Memory

The robot picks up the last conversation if it was within `FAST_BODY_MEMORY_RESUME_H` hours (6 by default). That is long enough to carry a morning into an afternoon, and short enough that last week is forgotten. fast-agent itself only resumes when told to with `--resume`, which suits a person at a terminal. A robot has nobody to ask, so it needs a standing rule.

The store lives in `~/.fast-body` so that removing or updating the app leaves it alone. Memory that disappears on an update is worse than none, because it looks like it works.

On a resumed run the robot repeats its own last words to show where it left off. Asking the brain for a greeting would cost a turn, and it would write a user message you never said into the history.

`FAST_BODY_CONTEXT_BUDGET` is a token count. fast-agent compacts at a fraction of the model's context window, and the router models report windows of about 1M tokens, so its default would not compact until about 850k.
