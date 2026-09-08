#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


HEADER = r"""\subsection{Complete Public NLDO Requests and Updates}
\label{app:complete-nldo-public-cases}

This section gives operator-facing public request and t01--t12 update summaries without omitting any time step. To avoid repeating the shared quality block for every update, each item lists the business-language update before the public evaluation reminder; Appendix~\ref{app:prompts-contracts} specifies the quality and output contract. The update-type badges (\textsf{memory}, \textsf{large}, \textsf{maintenance}, \textsf{objective}, \textsf{simple}) are analysis annotations added only for this appendix: they are never included in agent prompts or public update strings, and \textsf{large} marks exactly the deliberately disruptive t11--t12 replacements. Exact reproduction additionally requires the machine-readable prompt strings, public tables, and checksums. Hidden deltas, references, and evaluator implementations are never included here.

\begingroup
% Keep the complete reproducibility text at the dense-appendix level; the
% compact update list below avoids a nearly empty spill page.
\footnotesize
\setlength{\parskip}{0pt}
\sloppy
"""


LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "_": r"\_",
    "&": r"\&",
    "%": r"\%",
    "#": r"\#",
    "$": r"\$",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(text: str) -> str:
    return "".join(LATEX_ESCAPES.get(char, char) for char in text)


def business_text(text: str) -> str:
    for marker in ("\n\nScoring after this update:", "\n\nScoring rules:"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return " ".join(text.split())


def update_label(update: dict[str, Any]) -> str:
    if update.get("requires_memory"):
        return "memory"
    return str(update.get("difficulty") or "update").replace("_", "-")


def render_episode(episode: dict[str, Any]) -> str:
    episode_id = str(episode["episode_id"])
    domain = latex_escape(str(episode.get("domain") or "unknown"))
    lines = [
        f"\\NLDOEpisodeHead{{{latex_escape(episode_id)} (NLDO; {domain})}}",
        "\\noindent\\textbf{Initial public request.} "
        + latex_escape(business_text(str(episode.get("public_initial_problem") or ""))),
        "",
        "\\noindent\\textbf{Public updates.}",
        r"\begin{NLDOUpdateList}",
    ]
    for stage_index, update in enumerate(episode.get("update_stream") or [], start=1):
        label = latex_escape(update_label(update))
        text = latex_escape(business_text(str(update.get("public_update") or "")))
        lines.append(f"\\item[\\NLDOUpdateLabel{{t{stage_index:02d}}}{{{label}}}] {text}")
    lines.extend([r"\end{NLDOUpdateList}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the complete public NLDO appendix from its JSONL source.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/evo2_dynoptbench/public_csv/nldo_15episodes_12updates_csv.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=Path("paper/sections/nldo_full_cases.tex"))
    args = parser.parse_args()

    episodes = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    body = HEADER + "\n".join(render_episode(episode) for episode in episodes) + "\\endgroup\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(body, encoding="utf-8")
    print(json.dumps({"input": str(args.input), "output": str(args.output), "episodes": len(episodes)}))


if __name__ == "__main__":
    main()
