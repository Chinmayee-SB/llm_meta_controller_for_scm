# Live demo

Runs the frozen checkpoints live on a laptop CPU and animates the real simulation.
Nothing on screen is pre-rendered: each light packet is a shipment from the running model.

## Start

```bash
./run_demo.sh
```

Open http://localhost:8765 and press `F` for full screen. Wait about 30 seconds after starting
so the suite-wide numbers can finish computing in the background.

## LLM key

Copy `.env.example` to `.env` in this repository's root and set `OPENAI_API_KEY`, then restart.
`.env` is ignored by git. The status at the top right shows whether a test call succeeded.
Without a working key, section 03 uses the logged decisions and says so on screen.

## Sections

| Key | Section | What it shows |
|---|---|---|
| `1` | Recovery | Base and curriculum policies on the same shock |
| `2` | Cooperation Dial | The same frozen weights respond to the alpha input |
| `3` | Zero-Shot Supervisor | The LLM reads telemetry every 20 periods and assigns alpha |
| `4` | Evidence | All 280 traces; select one to rerun it live |

Other keys: `Space` play or pause, `R` restart, left and right arrows step one period.
In section 03, pressing pause during a decision holds the run on it.

## Check the numbers

```bash
.venv/bin/python demo/verify.py
```

Reruns all 280 traces for all four arms and fails if any differs from `benchmark_results`.

## Files

- `engine.py`: period-by-period simulation, same loop as `regime_aware/evaluate_benchmark.py`
- `server.py`: local API, LLM calls and static hosting
- `static/`: front end, with no build step and local fonts, so it runs offline
