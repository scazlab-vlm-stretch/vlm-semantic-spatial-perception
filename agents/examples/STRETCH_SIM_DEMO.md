## Stretch Sim Demo

- `examples/run_stretch_sim_demo.py` uses `TaskOrchestrator` as the top-level coordinator for perception, task analysis, PDDL generation, planning, and state persistence in the Stretch PyBullet demo.
- Keep the `GEMINI_MODEL` runtime default aligned with the docstring and with currently available Gemini models. A stale `gemini-2.0-flash` default caused object detection to fail for new users with a `404 NOT_FOUND`.
