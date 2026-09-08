<div align="center">

# MAPLE

**Memory-Augmented Planning with Language and Evolution**

Build an optimization problem, revise it in conversation, and continue from the accepted state.

[Quick start](#quick-start) · [Runtime](#runtime) · [Harness application](https://anonymous.4open.science/r/MAPLE-harness-core) · [Online demo](https://anonymous.4open.science/w/MAPLE-demo-review-F98B/)

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![License ISC](https://img.shields.io/badge/License-ISC-39857D)
![Solvers](https://img.shields.io/badge/Solvers-LP%20%2F%20MILP%20%2F%20GA%20%2F%20NSGA--II-334B60)

</div>

MAPLE translates natural-language requests and public tables into an executable Workbench. Each accepted update retains the current problem, decisions, event history, and numerical candidates for the next request.

## Quick start

Requires Python 3.10 or newer. Download the source archive from [this anonymous repository](https://anonymous.4open.science/r/MAPLE00) and extract it into a directory named `MAPLE`.

```bash
cd MAPLE
python -m venv .venv
source .venv/bin/activate
python -m pip install -e . numpy scipy
python scripts/llm_tests/run_liveopt_dynamic_scaffold_smoke.py --output-dir /tmp/maple-smoke
```

The smoke check uses a local scripted client and runs without an API key. Autonomous model calls require provider credentials; configure them using [.env.example](.env.example).

For the browser workspace, install [MAPLE Harness](https://anonymous.4open.science/r/MAPLE-harness-core#quick-start).

## Runtime

| Stage | Behavior |
|---|---|
| Construct | Declare typed decisions and evaluation functions in a Workbench. |
| Revise | Identify and apply the data or function changes needed by a new request. |
| Resolve history | Look up stored events and previously accepted decisions. |
| Search | Use LP/MILP, GA, or NSGA-II with fresh or reusable candidates. |
| Commit | Validate the output and retain the accepted state for subsequent requests. |

A dispatcher can tighten a delivery window and then ask to preserve an earlier customer assignment. MAPLE resolves that historical reference and searches under the revised constraints.

## Source map

| Directory | Contents |
|---|---|
| [evo2/](evo2/) | Agents, typed operators, numerical solvers, memory, and evaluation |
| [demo/](demo/) | Local browser application and scripted client |
| [data/](data/) | Benchmark inputs and numerical references |
| [scripts/](scripts/) | Dataset preparation, execution, and scoring utilities |
| [tests/](tests/) | Runtime, provider-adapter, data, and evaluation checks |

Package and command names retain `liveopt` for compatibility with existing integrations.

## Development

```bash
python -m pip install -e . numpy scipy pytest
python -m pytest tests -q
```

Application runs consume public requests, tables, and execution feedback. Held-out evaluators and reference fronts are used separately when scoring results.

## License

Released under the [ISC License](LICENSE). Third-party data and dependencies retain their own notices.
