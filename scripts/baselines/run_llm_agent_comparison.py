#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.llm_tests.run_emnlp_baseline_comparison import summarize_file


def run_cmd(cmd: list[str]) -> int:
    print("RUN", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, text=True, capture_output=True)
    print(p.stdout)
    if p.returncode != 0:
        print(p.stderr, file=sys.stderr)
    return p.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LLM-based agents on long natural-language dynamic episodes.")
    parser.add_argument("--episodes", default="data/evo2_dynoptbench/episodes_llm_memory/llm_memory_sample_6episodes.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--out-dir", default="logs/llm_tests/llm_agent_comparison")
    parser.add_argument(
        "--modes",
        default="evo2,stagewise_full_restart,react_tools,optimai",
    )
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    summaries = []
    for mode in modes:
        output = out / f"{mode}_limit{args.limit}.jsonl"
        cmd = [
            sys.executable,
            "-m",
            "evo2.experiments.run_episode_suite",
            "--episodes",
            args.episodes,
            "--limit",
            str(args.limit),
            "--agent-mode",
            mode,
            "--model",
            args.model,
            "--output",
            str(output),
            "--memory-root",
            str(out / f"memory_{mode}"),
        ]
        code = run_cmd(cmd)
        if code == 0 and output.exists():
            summaries.append(summarize_file(mode, output))
        else:
            summaries.append({"mode": mode, "error": f"exit={code}", "output": str(output)})
    summary_path = out / f"summary_limit{args.limit}.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "results": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
