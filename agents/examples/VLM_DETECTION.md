## VLM Detection Example

- `examples/test_vlm_detection.py` is the minimal orchestrator-backed VLM detection path: it mirrors `run_stretch_sim_demo.py` up through continuous detection, but removes sim, PDDL generation, planning, and execution.
- Keep this example on the `ObjectTracker` / `ContinuousObjectTracker` path. Do not silently switch it to GSAM2; the point of the example is to exercise the VLM detector itself.
- Use a static image frame provider to keep the example dependency-light and focused on the detection loop structure rather than camera setup.
