"""Embodiment — the robot's body, exposed to the fast-agent brain.

- `moves`   : the 100 Hz movement system (head/antennas, recorded emotions & dances, idle breathing)
- `context` : a process-wide handle the in-process tools use to reach the live robot
- `tools`   : `@fast.tool` functions the brain calls to move the body and look
- `cues`    : conversation state (listening / thinking / speaking) as antenna gestures
- `gaze`    : daemon-side face tracking, on and off, and how strongly it owns the head
- `vision`  : `examine()`: one question about the camera frame to a small vision model
"""
