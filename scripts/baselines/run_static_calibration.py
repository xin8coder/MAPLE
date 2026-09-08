#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evo2.agents.deepseek_client import DeepSeekDebugClient


DEFAULT_METHODS = "liveopt_static,optimai_2025,or_llm_agent_2025"
PROMPT_VERSION = "static_solving_v5_executable_code_accuracy_schema_hint"
STATIC_CODE_TIMEOUT_SECONDS = 20
OFFICIAL_STATIC_METHODS = {"orlm", "or_llm_agent_2025", "optimus"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run cache-first static NL optimization calibration rows for LiveOpt and selected baselines."
    )
    parser.add_argument("--manifest", required=True, help="Path to static_calibration_manifest.json.")
    parser.add_argument("--out-dir", default="logs/static_calibration/run_latest")
    parser.add_argument("--methods", default="")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--case-offset", type=int, default=0)
    parser.add_argument("--case-stride", type=int, default=1)
    parser.add_argument("--case-ids", default="", help="Optional comma-separated case ids to run.")
    parser.add_argument("--stop-after-new", type=int, default=0)
    parser.add_argument("--use-response-cache", dest="use_response_cache", action="store_true", default=True)
    parser.add_argument("--no-response-cache", dest="use_response_cache", action="store_false")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--cache-dir", default="outputs/deepseek_cache")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-cache-misses", action="store_true")
    parser.add_argument("--retry-statuses", default="", help="Comma-separated statuses to reopen on resume, e.g. failed,schema_failed.")
    parser.add_argument("--no-json-mode", action="store_true", help="Do not request provider-enforced JSON mode; parse JSON from normal text instead.")
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument(
        "--code-repair-attempts",
        type=int,
        default=int(os.environ.get("DEEPSEEK_CODE_REPAIR_ATTEMPTS", "3") or 3),
        help=(
            "Equal public code-output repair budget for non-official code-solving rows. "
            "Repairs only compile/runtime/stdout-format failures; never uses reference or hidden evaluator feedback."
        ),
    )
    parser.add_argument(
        "--official-repair-attempts",
        type=int,
        default=int(os.environ.get("DEEPSEEK_MAX_ATTEMPTS", "3") or 3),
    )
    parser.add_argument(
        "--response-protocol",
        choices=("json", "code_block", "official"),
        default="json",
        help="Expected model response protocol. official uses source-repository prompt/flow when available.",
    )
    args = parser.parse_args()

    configure_cache_env(args)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases_path = resolve_manifest_case_file(manifest_path, manifest.get("case_file") or "")
    cases = load_jsonl(cases_path)
    cases = select_cases(cases, args)
    if args.limit > 0:
        cases = cases[: args.limit]
    methods = [item.strip() for item in (args.methods or ",".join(manifest.get("methods") or []) or DEFAULT_METHODS).split(",") if item.strip()]
    unknown = [method for method in methods if method not in METHOD_REQUIRED_FIELDS]
    if unknown:
        raise SystemExit(f"Unsupported static calibration methods: {unknown}")

    out_dir = Path(args.out_dir)
    trace_dir = out_dir / "traces"
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "static_calibration_rows.jsonl"
    summary_path = out_dir / "static_calibration_summary.json"

    existing = load_jsonl(rows_path) if args.resume and rows_path.exists() else []
    latest = {row_key(row): row for row in existing}
    mode = "a" if args.resume and rows_path.exists() else "w"
    new_rows = 0
    skipped_existing = 0
    client = DeepSeekDebugClient(model=args.model)
    with rows_path.open(mode, encoding="utf-8") as handle:
        for case in cases:
            for method in methods:
                key = make_key(case, method)
                old = latest.get(key)
                if old and not should_retry(old, args):
                    skipped_existing += 1
                    continue
                row = run_case_method(case, method, client, args, trace_dir)
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                latest[key] = row
                new_rows += 1
                if args.stop_after_new and new_rows >= args.stop_after_new:
                    break
            if args.stop_after_new and new_rows >= args.stop_after_new:
                break

    latest_rows = [latest[key] for key in sorted(latest)]
    summary = summarize(latest_rows, args, manifest_path, rows_path, skipped_existing, new_rows)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def configure_cache_env(args: argparse.Namespace) -> None:
    os.environ["DEEPSEEK_CACHE"] = "1" if args.use_response_cache else "0"
    if args.cache_dir:
        os.environ["DEEPSEEK_CACHE_DIR"] = args.cache_dir
    if args.cache_only:
        os.environ["DEEPSEEK_CACHE_ONLY"] = "1"
        os.environ.setdefault("DEEPSEEK_API_KEY", "cache-only-dummy-key")
    else:
        os.environ.pop("DEEPSEEK_CACHE_ONLY", None)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def resolve_manifest_case_file(manifest_path: Path, raw_case_file: str) -> Path:
    path = Path(raw_case_file)
    if path.is_absolute() or path.exists():
        return path
    manifest_relative = manifest_path.parent / path
    if manifest_relative.exists():
        return manifest_relative
    sibling = manifest_path.parent / path.name
    if sibling.exists():
        return sibling
    return manifest_relative


def select_cases(cases: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = list(cases)
    if args.case_ids.strip():
        wanted = {item.strip() for item in args.case_ids.split(",") if item.strip()}
        selected = [case for case in selected if str(case.get("case_id")) in wanted or str(case.get("source_id")) in wanted]
    stride = max(1, int(args.case_stride or 1))
    offset = max(0, int(args.case_offset or 0))
    if stride > 1 or offset:
        selected = [case for index, case in enumerate(selected) if index % stride == offset % stride]
    return selected


def run_case_method(
    case: dict[str, Any],
    method: str,
    client: DeepSeekDebugClient,
    args: argparse.Namespace,
    trace_dir: Path,
) -> dict[str, Any]:
    if args.response_protocol == "official":
        if method in OFFICIAL_STATIC_METHODS:
            return run_case_method_official(case, method, client, args, trace_dir)
        return run_case_method_official_unsupported(case, method, args, trace_dir)
    started = time.perf_counter()
    messages = build_messages(case, method, response_protocol=args.response_protocol)
    raw_response = ""
    parsed: dict[str, Any] = {}
    usage: dict[str, Any] = {}
    status = "failed"
    failure_reason = ""
    validation_feedback: list[str] = []
    code_result: dict[str, Any] = {}
    repair_trace: list[dict[str, Any]] = []
    try:
        response = client.chat(messages, temperature=0.0, max_tokens=args.max_tokens, json_mode=not args.no_json_mode)
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        raw_response = response.get("choices", [{}])[0].get("message", {}).get("content", "") if isinstance(response, dict) else ""
        parsed = parse_model_response(raw_response, method, protocol=args.response_protocol)
        validation_feedback = validate_parsed_output(method, parsed)
        parsed, code_result, repair_trace = execute_and_repair_solver_code(
            case=case,
            method=method,
            parsed=parsed,
            client=client,
            args=args,
            base_messages=messages,
            initial_validation_feedback=validation_feedback,
        )
        if repair_trace:
            usage = sum_usage([usage, *(attempt.get("token_usage", {}) for attempt in repair_trace)])
        validation_feedback = validate_parsed_output(method, parsed)
        status = "completed" if not validation_feedback else "schema_failed"
        failure_reason = "; ".join(validation_feedback)
        if not code_result.get("compile_success"):
            status = "compile_failed"
            failure_reason = str(code_result.get("compile_error") or "compile_failed")
        elif not code_result.get("runtime_success"):
            status = "runtime_failed"
            failure_reason = str(code_result.get("runtime_error") or "runtime_failed")
    except Exception as exc:  # noqa: BLE001
        failure_reason = str(exc)
        status = "cache_miss_blocked" if "DEEPSEEK_CACHE_ONLY" in failure_reason or "local response cache missed" in failure_reason else "failed"
        validation_feedback = [failure_reason]
    latency = time.perf_counter() - started
    trace_path = write_trace(
        trace_dir,
        case,
        method,
        messages,
        raw_response,
        parsed,
        usage,
        status,
        failure_reason,
        extra={
            "code_execution": code_result,
            "code_output_repair": {
                "policy": "equal_public_code_output_repair_v1",
                "max_attempts": int(getattr(args, "code_repair_attempts", 0) or 0),
                "attempts_used": len(repair_trace),
                "attempts": repair_trace,
                "hidden_feedback_used": False,
                "reference_feedback_used": False,
            },
        },
    )
    return {
        "schema_version": "liveopt_static_calibration_row_v1",
        "prompt_version": PROMPT_VERSION,
        "case_id": case.get("case_id"),
        "source": case.get("source"),
        "source_id": case.get("source_id"),
        "method": method,
        "model": args.model,
        "status": status,
        "problem_text": case.get("problem_text", ""),
        "reference_available": bool(case.get("reference")),
        "reference_keys": sorted((case.get("reference") or {}).keys()) if isinstance(case.get("reference"), dict) else [],
        "messages": messages,
        "selected_stage_prompt": messages[-1]["content"] if messages else "",
        "raw_response": raw_response,
        "parsed_output": parsed,
        "validation_feedback": validation_feedback,
        "reference_evaluation": evaluate_reference(case, parsed),
        "official_evaluation": evaluate_official_static(case, parsed),
        "code_execution": code_result,
        "code_output_repair": {
            "policy": "equal_public_code_output_repair_v1",
            "max_attempts": int(getattr(args, "code_repair_attempts", 0) or 0),
            "attempts_used": len(repair_trace),
            "attempts": repair_trace,
            "hidden_feedback_used": False,
            "reference_feedback_used": False,
        },
        "token_usage": normalize_usage(usage),
        "latency_seconds": latency,
        "failure_reason": failure_reason,
        "trace_path": str(trace_path),
    }


def build_messages(case: dict[str, Any], method: str, response_protocol: str = "json") -> list[dict[str, str]]:
    source = str(case.get("source") or "").lower()
    compact_code_first = source == "bwor"
    code_block_protocol = response_protocol == "code_block"
    if code_block_protocol:
        return build_code_block_messages(case, method)
    rules = [
        "Use only the problem text and public_context in this request.",
        "Do not assume hidden benchmark labels, hidden objective implementations, or dataset-specific templates.",
        "Return exactly one compact JSON object.",
        "Prefer explicit variables, objective, constraints, solver path, and answer format.",
        "Include executable Python solver_code. The code must read parameters.json when public_context contains parameters, solve the optimization problem, and print exactly one JSON object to stdout.",
        "The printed JSON object must contain objective_value and solution. Use standard library and scipy.optimize only; do not import gurobipy, pandas, networkx, requests, evo2, or private benchmark modules.",
        "When giving final_answer, evaluate the objective on the stated solution.",
        "Substitute the stated solution into every hard constraint before finalizing.",
        "For small integer LP/MILP problems, compare at least one lower-objective or nearby candidate and explain why it is infeasible or worse.",
        "Do not treat a feasible corner as optimal unless the objective direction and better-looking alternatives have been checked.",
        "When using scipy linprog/milp, encode maximization by negating the objective internally, then print the original objective value with the correct sign.",
        "Preserve the units and scale used by the public text. If the code rescales percentages, costs, or capacities internally, convert objective_value and solution fields back before printing.",
        "The printed objective_value must be recomputed from the printed solution, not copied from a derivation.",
        "For ratio, at-least, at-most, and no-more-than constraints, restate the inequality direction before coding it.",
        "Return only the fields inside required_output_schema; do not echo case_id, source, problem_text, prompt_version, public_reference_fields, required_output_schema, rules, or task.",
        "Keep derivation strings short and avoid unescaped quotation marks inside JSON string values.",
    ]
    if compact_code_first:
        rules = [
            "Use only the public problem text in this request.",
            "Return exactly one compact JSON object and no markdown.",
            "Keep all reasoning fields short. Do not write long derivations.",
            "Include executable Python solver_code using only standard library and scipy.optimize.",
            "Do not import gurobipy, pandas, networkx, requests, evo2, or private benchmark modules.",
            "The code must print exactly one JSON object with objective_value, solution, and optional solution_status.",
            "For no-optimal cases, print solution_status as infeasible, unbounded, or no_optimal.",
            "For LP/MILP/IP cases, prefer scipy.optimize.linprog or scipy.optimize.milp when applicable.",
            "If maximizing, negate the scipy objective internally and print the original objective value with the correct sign.",
            "Preserve public units and scale in objective_value and solution fields.",
            "Compute objective_value from the printed solution before printing JSON.",
            "Return only the fields inside required_output_schema.",
        ]
    task = "Solve this static natural-language optimization problem from public text and public parameters."
    if source == "nldo-dynamic-public":
        task = "Solve the current dynamic optimization stage from public text, public parameters, and allowed public state."
    payload = {
        "prompt_version": PROMPT_VERSION,
        "case_id": case.get("case_id"),
        "source": case.get("source"),
        "task": task,
        "problem_text": case.get("problem_text", ""),
        "public_context": compact_public_context(case.get("public_context")),
        "public_reference_fields": sorted((case.get("reference") or {}).keys()) if isinstance(case.get("reference"), dict) else [],
        "public_solution_schema_hint": solution_schema_hint_from_reference(case.get("reference")),
        "rules": rules,
        "required_output_schema": schema_for_method(method, compact=compact_code_first),
    }
    system = SYSTEM_PROMPTS[method]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


def build_code_block_messages(case: dict[str, Any], method: str) -> list[dict[str, str]]:
    method_instruction = {
        "optimai_2025": (
            "Follow the OptimAI workflow internally: formulate the model, compare solver plans, choose a solver, "
            "and validate the code. Output only the final executable code block."
        ),
        "or_llm_agent_2025": (
            "Follow the OR-LLM-Agent workflow internally: build the math model, generate solver code, self-check, "
            "and repair obvious errors. Output only the final executable code block."
        ),
        "liveopt_static": (
            "Follow LiveOpt static mode internally: choose a compact encoding/solver path and verify the result. "
            "Output only the final executable code block."
        ),
        "react_tools": (
            "Follow a ReAct+Tools workflow internally: reason about which public solver/tool action would solve the "
            "static request, then emit the final executable solver code. Output only the final executable code block."
        ),
        "persistent_react": (
            "Follow a Persistent ReAct+Tools workflow internally: reuse your carried solver code and accepted plan "
            "when supplied, apply the smallest necessary edits for the new public update, and emit the complete "
            "updated executable solver code (never a diff). Output only the final executable code block."
        ),
        "react_transcript_state": (
            "Follow a ReAct+Tools workflow with an explicit public transcript and public accepted-state summary. "
            "Do not use LiveOpt memory or restart APIs. Output only the final executable solver code."
        ),
        "react_generic_workbench": (
            "Follow a ReAct+Tools workflow with a generic public executable workbench surface and fixed public tools. "
            "Do not use LiveOpt LSM, TSS patch state, search archives, or adaptive restart. Output only the final executable code block."
        ),
        "react_public_delta_oracle": (
            "Follow a ReAct+Tools workflow with a public delta summary derived only from visible text. "
            "Do not use hidden deltas or LiveOpt memory. Output only the final executable code block."
        ),
    }.get(method, "Build an optimization model and output only the final executable code block.")
    system = (
        "You are an operations research modeling code generator. "
        "Produce one complete Python code block for the public problem. "
        "Use standard library and scipy.optimize only."
    )
    public_context = compact_public_context(case.get("public_context"))
    public_parameters = public_context.get("parameters") if isinstance(public_context.get("parameters"), dict) else {}
    solution_schema_hint = solution_schema_hint_from_reference(case.get("reference"))
    public_context_text = json.dumps(public_context, ensure_ascii=False, sort_keys=True)
    public_parameters_text = json.dumps(public_parameters, ensure_ascii=False, sort_keys=True)
    solution_schema_hint_text = json.dumps(solution_schema_hint, ensure_ascii=False, sort_keys=True)
    user = f"""Problem source: {case.get('source')}
Problem id: {case.get('source_id') or case.get('case_id')}

Natural-language optimization problem:
{case.get('problem_text', '')}

Public context JSON:
{public_context_text}

Public parameters JSON:
{public_parameters_text}

Expected public solution schema hint (field names/types only, no answer values):
{solution_schema_hint_text}

Instructions:
- {method_instruction}
- Do not use gurobipy, pandas, networkx, requests, evo2, hidden labels, or private benchmark modules.
- The code must solve the problem from the text and public parameters above and print exactly one JSON object to stdout.
- At runtime, the same public parameters are available in `parameters.json`; use the exact parameter keys shown above instead of inventing placeholder names.
- The printed JSON object must include `objective_value` for optimal cases and `solution`.
- When the schema hint is nonempty, use those exact solution field names and compatible nested shapes in `solution`; do not copy placeholder type strings as values.
- If the public problem has no optimal solution, print `solution_status` as `infeasible`, `unbounded`, or `no_optimal`.
- Prefer scipy.optimize.linprog for LPs and scipy.optimize.milp for integer or mixed-integer cases when applicable.
- When using scipy.optimize.milp, import and pass `Bounds(lb, ub)` rather than a list of tuple bounds; pass constraints as `LinearConstraint` objects.
- If the user asks to "formulate" a model but an answer metric is available, build and solve the public model, then print the optimal objective and solution.
- Use the public variable meanings from the prompt when naming solution fields; avoid generic x1/x2 when semantic variable names are clear.
- Output only:
```python
# complete executable code here
```
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


ORLM_Q2MC_TEMPLATE = """Below is an operations research question. Build a mathematical model and corresponding python code using `coptpy` that appropriately addresses the question.

# Question:
{question}

# Response:"""

OR_LLM_AGENT_MATH_MODEL_SYSTEM_PROMPT = (
    "你是一个运筹优化专家。请根据用户提供的运筹优化问题构建数学模型，以数学（线性规划）模型对原问题进行有效建模。"
    "尽量关注获得一个正确的数学模型表达式，无需太关注解释。"
    "该模型后续用作指导生成gurobi代码，这一步主要用作生成有效的线性规模表达式。"
)

OR_LLM_AGENT_CODE_GENERATION_SYSTEM_PROMPT = (
    "你是一个运筹优化专家。请根据用户提供的运筹优化问题构建数学模型，并写出完整、可靠的 Python 代码，使用 Gurobi 求解该运筹优化问题。"
    "代码中请包含必要的模型构建、变量定义、约束添加、目标函数设定以及求解和结果输出。"
    "以 ```python\n{code}\n``` 形式输出，无需输出代码解释。"
)

OR_LLM_AGENT_REQUEST_GUROBI_CODE_WITH_MATH_PROMPT = (
    "请基于以上的数学模型，写出完整、可靠的 Python 代码，使用 Gurobi 求解该运筹优化问题。"
    "代码中请包含必要的模型构建、变量定义、约束添加、目标函数设定以及求解和结果输出。"
    "以 ```python\n{code}\n``` 形式输出，无需输出代码解释。"
)

OR_LLM_AGENT_ERROR_FIX_PROMPT_TEMPLATE = "代码执行出现错误，错误信息如下:\n{error_msg}\n请修复代码并重新提供完整的可执行代码。"

OR_LLM_AGENT_INFEASIBLE_SOLUTION_PROMPT = (
    "现有模型运行结果为*无可行解*，请认真仔细地检查数学模型和gurobi代码，是否存在错误，以致于造成无可行解"
    "检查完成后，最终请重新输出gurobi python代码以 ```python\n{code}\n``` 形式输出，无需输出代码解释。"
)

OPTIMUS_REFLEXION_PROMPT_TEMPLATE = """
You are an expert operations research analyst. Your task is to generate Gurobi code to solve the following optimization problem:

{problem_description}

The code should save the final optimal value in a file named 'ref_optimal_value.txt'.
First, reason about the problem and model it. Then, generate gurobipy code to solve it. Put the code between two '=====' lines, like this:

=====
import ...
...
=====

- The code should save the final optimal value in a file named 'ref_optimal_value.txt'.
- Generate the complete code, including the model definition, variables, constraints, objective function, and optimization. It must be runnable.
- Do not generate anything after the second '====='.
- Take a deep breath and think step by step.
"""

OPTIMUS_REFLEXION_REPAIR_TEMPLATE = """
You are an expert operations research analyst. You have been given the task to generate Gurobi code to solve an optimization problem. You have generated the following Gurobi code:

{generated_code}

You have been updating the code for these errors (the last one is the most recent one):

{feedback}

Based on this feedback, suggest improvements to the Gurobi code.
First, reason about the problem and model it. Then, generate gurobipy code to solve it. Put the code between two '=====' lines, like this:

=====
import ...
...
=====

- The code should save the final optimal value in a file named 'ref_optimal_value.txt'.
- Generate the complete code, including the model definition, variables, constraints, objective function, and optimization. It must be runnable.
- Do not generate anything after the second '====='.
- Take a deep breath and think step by step.
"""


def run_case_method_official(
    case: dict[str, Any],
    method: str,
    client: DeepSeekDebugClient,
    args: argparse.Namespace,
    trace_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    messages: list[dict[str, str]] = []
    llm_responses: list[dict[str, Any]] = []
    raw_response = ""
    parsed: dict[str, Any] = {}
    usage: dict[str, Any] = {}
    status = "failed"
    failure_reason = ""
    validation_feedback: list[str] = []
    code_result: dict[str, Any] = {}
    try:
        if method == "or_llm_agent_2025":
            parsed, messages, llm_responses, code_result = run_official_or_llm_agent(case, client, args)
        elif method == "optimus":
            parsed, messages, llm_responses, code_result = run_official_optimus(case, client, args)
        elif method == "orlm":
            parsed, messages, llm_responses, code_result = run_official_orlm(case, method, client, args)
        else:
            return run_case_method_official_unsupported(case, method, args, trace_dir)
        raw_response = llm_responses[-1].get("raw_response", "") if llm_responses else ""
        usage = sum_usage(call.get("usage", {}) for call in llm_responses)
        validation_feedback = validate_parsed_output(method, parsed)
        if not code_result.get("compile_success"):
            status = "compile_failed"
            failure_reason = str(code_result.get("compile_error") or "compile_failed")
        elif not code_result.get("runtime_success"):
            status = "runtime_failed"
            failure_reason = str(code_result.get("runtime_error") or "runtime_failed")
        elif validation_feedback:
            status = "schema_failed"
            failure_reason = "; ".join(validation_feedback)
        else:
            status = "completed"
    except Exception as exc:  # noqa: BLE001
        failure_reason = str(exc)
        status = "cache_miss_blocked" if "DEEPSEEK_CACHE_ONLY" in failure_reason or "local response cache missed" in failure_reason else "failed"
        validation_feedback = [failure_reason]
        if not messages:
            messages = build_official_initial_messages(case, method)
    latency = time.perf_counter() - started
    official_eval = evaluate_official_execution(case, parsed, code_result)
    trace_path = write_trace(
        trace_dir,
        case,
        method,
        messages,
        raw_response,
        parsed,
        usage,
        status,
        failure_reason,
        prompt_version=official_prompt_version(method),
        extra={
            "llm_responses": llm_responses,
            "official_flow": official_flow_name(method),
            "official_code_execution": code_result,
            "code_output_repair": {
                "policy": "source_aligned_native_repair_capped_v1",
                "max_attempts": int(getattr(args, "official_repair_attempts", 0) or 0),
                "attempts_used": count_official_repair_calls(llm_responses),
                "hidden_feedback_used": False,
                "reference_feedback_used": False,
            },
        },
    )
    return {
        "schema_version": "liveopt_static_calibration_row_v1",
        "prompt_version": official_prompt_version(method),
        "case_id": case.get("case_id"),
        "source": case.get("source"),
        "source_id": case.get("source_id"),
        "method": method,
        "model": args.model,
        "status": status,
        "problem_text": case.get("problem_text", ""),
        "reference_available": bool(case.get("reference")),
        "reference_keys": sorted((case.get("reference") or {}).keys()) if isinstance(case.get("reference"), dict) else [],
        "messages": messages,
        "selected_stage_prompt": messages[-1]["content"] if messages else "",
        "raw_response": raw_response,
        "llm_responses": llm_responses,
        "parsed_output": parsed,
        "validation_feedback": validation_feedback,
        "reference_evaluation": evaluate_reference(case, parsed),
        "official_evaluation": official_eval,
        "token_usage": normalize_usage(usage),
        "latency_seconds": latency,
        "failure_reason": failure_reason,
        "trace_path": str(trace_path),
        "official_flow": official_flow_name(method),
        "official_code_execution": code_result,
        "code_output_repair": {
            "policy": "source_aligned_native_repair_capped_v1",
            "max_attempts": int(getattr(args, "official_repair_attempts", 0) or 0),
            "attempts_used": count_official_repair_calls(llm_responses),
            "hidden_feedback_used": False,
            "reference_feedback_used": False,
        },
    }


def run_case_method_official_unsupported(
    case: dict[str, Any],
    method: str,
    args: argparse.Namespace,
    trace_dir: Path,
) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": "Official-protocol baseline guard.",
        },
        {
            "role": "user",
            "content": (
                f"Method `{method}` has no local source-repository prompt/flow mapping in this runner. "
                "Use `code_block` for paper-protocol reproductions or add an explicit official adapter."
            ),
        },
    ]
    failure_reason = (
        f"unsupported_official_protocol: {method} is not mapped to a checked official prompt/flow; "
        "not using LiveOpt or legacy JSON prompts as a substitute."
    )
    trace_path = write_trace(
        trace_dir,
        case,
        method,
        messages,
        "",
        {},
        {},
        "unsupported_official_protocol",
        failure_reason,
        prompt_version="official_protocol_guard_v1",
        extra={"official_flow": "unsupported"},
    )
    return {
        "schema_version": "liveopt_static_calibration_row_v1",
        "prompt_version": "official_protocol_guard_v1",
        "case_id": case.get("case_id"),
        "source": case.get("source"),
        "source_id": case.get("source_id"),
        "method": method,
        "model": args.model,
        "status": "unsupported_official_protocol",
        "problem_text": case.get("problem_text", ""),
        "reference_available": bool(case.get("reference")),
        "reference_keys": sorted((case.get("reference") or {}).keys()) if isinstance(case.get("reference"), dict) else [],
        "messages": messages,
        "selected_stage_prompt": messages[-1]["content"],
        "raw_response": "",
        "parsed_output": {},
        "validation_feedback": [failure_reason],
        "reference_evaluation": evaluate_reference(case, {}),
        "official_evaluation": {
            "schema_version": "static_solving_official_eval_v1",
            "status": "unsupported_official_protocol",
            "official_accuracy": False,
            "compile_success": False,
            "runtime_success": False,
            "objective_exact": None,
            "solution_exact": None,
        },
        "token_usage": normalize_usage({}),
        "latency_seconds": 0.0,
        "failure_reason": failure_reason,
        "trace_path": str(trace_path),
        "official_flow": "unsupported",
        "official_code_execution": {},
    }


def official_prompt_version(method: str) -> str:
    if method == "or_llm_agent_2025":
        return "or_llm_agent_official_math_code_debug_v1"
    if method == "optimus":
        return "optimus_official_reflexion_gurobi_v1"
    return "orlm_official_q2mc_coptpy_v1"


def official_flow_name(method: str) -> str:
    if method == "or_llm_agent_2025":
        return "math_model_then_gurobi_code_then_execute_debug"
    if method == "optimus":
        return "optimus_reflexion_gurobi_code_debug"
    return "q2mc_coptpy_single_prompt_then_execute"


def count_official_repair_calls(llm_responses: list[dict[str, Any]]) -> int:
    return sum(1 for item in llm_responses if "repair" in str(item.get("role") or ""))


def build_official_initial_messages(case: dict[str, Any], method: str) -> list[dict[str, str]]:
    question = official_question_text(case)
    if method == "or_llm_agent_2025":
        return [
            {"role": "system", "content": OR_LLM_AGENT_MATH_MODEL_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
    if method == "optimus":
        return [{"role": "user", "content": OPTIMUS_REFLEXION_PROMPT_TEMPLATE.format(problem_description=question).strip()}]
    return [{"role": "user", "content": ORLM_Q2MC_TEMPLATE.format(question=question).strip()}]


def run_official_orlm(
    case: dict[str, Any],
    method: str,
    client: DeepSeekDebugClient,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
    question = official_question_text(case)
    messages = [{"role": "user", "content": ORLM_Q2MC_TEMPLATE.format(question=question).strip()}]
    response = client.chat(messages, temperature=0.0, max_tokens=args.max_tokens, json_mode=False)
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    code = extract_official_code(content)
    code_result = execute_official_solver_code(code, solver="coptpy")
    objective = first_numeric(code_result.get("objective_value"))
    parsed = parsed_output_from_code_block(method, code)
    parsed["final_answer"] = {
        "objective_value": objective,
        "solution_status": "optimal" if objective is not None else "no_optimal",
        "solution": {},
        "derivation": "ORLM official q2mc prompt executed with COPT",
    }
    return parsed, messages, [{"role": "q2mc", "usage": response.get("usage", {}) or {}, "raw_response": content}], code_result


def run_official_optimus(
    case: dict[str, Any],
    client: DeepSeekDebugClient,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
    question = official_question_text(case)
    prompt = OPTIMUS_REFLEXION_PROMPT_TEMPLATE.format(problem_description=question).strip()
    messages = [{"role": "user", "content": prompt}]
    calls: list[dict[str, Any]] = []
    response = client.chat(messages, temperature=0.0, max_tokens=args.max_tokens, json_mode=False)
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    calls.append({"role": "initial_reflexion", "usage": response.get("usage", {}) or {}, "raw_response": content})
    code = extract_optimus_code(content)
    code_result = execute_official_solver_code(code, solver="gurobipy")
    feedback = ""
    attempts = 0
    max_attempts = max(0, int(getattr(args, "official_repair_attempts", 3) or 0))
    while attempts < max_attempts and not code_result.get("runtime_success"):
        attempts += 1
        feedback += "\n" + str(code_result.get("runtime_error") or code_result.get("compile_error") or "unknown execution error")
        repair_prompt = OPTIMUS_REFLEXION_REPAIR_TEMPLATE.format(generated_code=code, feedback=feedback).strip()
        repair_messages = [{"role": "user", "content": repair_prompt}]
        repair_response = client.chat(repair_messages, temperature=0.0, max_tokens=args.max_tokens, json_mode=False)
        repair_content = repair_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        calls.append({"role": f"reflexion_repair_{attempts}", "usage": repair_response.get("usage", {}) or {}, "raw_response": repair_content})
        messages.extend([{"role": "assistant", "content": content}, {"role": "user", "content": repair_prompt}])
        content = repair_content
        code = extract_optimus_code(repair_content)
        code_result = execute_official_solver_code(code, solver="gurobipy")
    objective = first_numeric(code_result.get("objective_value"))
    parsed = parsed_output_from_code_block("optimus", code)
    parsed["formulation"] = {"source": "OptiMUS official Reflexion prompt/code-debug flow"}
    parsed["debug_feedback"] = feedback
    parsed["final_answer"] = {
        "objective_value": objective,
        "solution_status": "optimal" if objective is not None else "no_optimal",
        "solution": {},
        "derivation": "OptiMUS official Reflexion flow",
    }
    return parsed, messages, calls, code_result


def run_official_or_llm_agent(
    case: dict[str, Any],
    client: DeepSeekDebugClient,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
    question = official_question_text(case)
    all_messages: list[dict[str, str]] = []
    calls: list[dict[str, Any]] = []

    math_messages = [
        {"role": "system", "content": OR_LLM_AGENT_MATH_MODEL_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    math_response = client.chat(math_messages, temperature=0.0, max_tokens=args.max_tokens, json_mode=False)
    math_model = math_response.get("choices", [{}])[0].get("message", {}).get("content", "")
    calls.append({"role": "math_model", "usage": math_response.get("usage", {}) or {}, "raw_response": math_model})
    all_messages.extend(math_messages)
    all_messages.append({"role": "assistant", "content": math_model})

    code_messages = [
        {"role": "system", "content": OR_LLM_AGENT_MATH_MODEL_SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": math_model},
        {"role": "user", "content": OR_LLM_AGENT_REQUEST_GUROBI_CODE_WITH_MATH_PROMPT},
    ]
    code_response = client.chat(code_messages, temperature=0.0, max_tokens=args.max_tokens, json_mode=False)
    code_content = code_response.get("choices", [{}])[0].get("message", {}).get("content", "")
    calls.append({"role": "code_generation", "usage": code_response.get("usage", {}) or {}, "raw_response": code_content})
    all_messages.extend([{"role": "user", "content": OR_LLM_AGENT_REQUEST_GUROBI_CODE_WITH_MATH_PROMPT}, {"role": "assistant", "content": code_content}])
    code = extract_official_code(code_content)
    code_result = execute_official_solver_code(code, solver="gurobipy")

    attempts = 0
    max_attempts = max(0, int(getattr(args, "official_repair_attempts", 3) or 0))
    repair_messages = list(code_messages)
    repair_messages.append({"role": "assistant", "content": code_content})
    while attempts < max_attempts and not code_result.get("runtime_success"):
        attempts += 1
        repair_prompt = OR_LLM_AGENT_ERROR_FIX_PROMPT_TEMPLATE.format(
            error_msg=code_result.get("runtime_error") or code_result.get("compile_error") or "unknown execution error"
        )
        repair_messages.append({"role": "user", "content": repair_prompt})
        repair_response = client.chat(repair_messages, temperature=0.0, max_tokens=args.max_tokens, json_mode=False)
        repair_content = repair_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        calls.append({"role": f"repair_{attempts}", "usage": repair_response.get("usage", {}) or {}, "raw_response": repair_content})
        all_messages.extend([{"role": "user", "content": repair_prompt}, {"role": "assistant", "content": repair_content}])
        repair_messages.append({"role": "assistant", "content": repair_content})
        code = extract_official_code(repair_content)
        code_result = execute_official_solver_code(code, solver="gurobipy")

    if code_result.get("runtime_success") and first_numeric(code_result.get("objective_value")) is None and attempts < max_attempts:
        attempts += 1
        infeasible_messages = list(repair_messages)
        infeasible_messages.append({"role": "user", "content": OR_LLM_AGENT_INFEASIBLE_SOLUTION_PROMPT})
        infeasible_response = client.chat(infeasible_messages, temperature=0.0, max_tokens=args.max_tokens, json_mode=False)
        infeasible_content = infeasible_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        calls.append({"role": "infeasible_repair", "usage": infeasible_response.get("usage", {}) or {}, "raw_response": infeasible_content})
        all_messages.extend([{"role": "user", "content": OR_LLM_AGENT_INFEASIBLE_SOLUTION_PROMPT}, {"role": "assistant", "content": infeasible_content}])
        candidate_code = extract_official_code(infeasible_content)
        candidate_result = execute_official_solver_code(candidate_code, solver="gurobipy")
        if candidate_result.get("runtime_success"):
            code = candidate_code
            code_result = candidate_result

    objective = first_numeric(code_result.get("objective_value"))
    parsed = parsed_output_from_code_block("or_llm_agent_2025", code)
    parsed["formulation"] = {"math_model": math_model}
    parsed["final_answer"] = {
        "objective_value": objective,
        "solution_status": "optimal" if objective is not None else "no_optimal",
        "solution": {},
        "derivation": "OR-LLM-Agent official math-code-debug flow",
    }
    return parsed, all_messages, calls, code_result


def official_question_text(case: dict[str, Any]) -> str:
    text = str(case.get("problem_text") or "").strip()
    public_context = compact_public_context(case.get("public_context"))
    params = public_context.get("parameters") if isinstance(public_context.get("parameters"), dict) else {}
    if params:
        text = text.rstrip() + "\n\nPublic numeric/data parameters:\n" + json.dumps(params, ensure_ascii=False, sort_keys=True)
    hint = solution_schema_hint_from_reference(case.get("reference"))
    if hint:
        text = text.rstrip() + "\n\nExpected answer schema field names only:\n" + json.dumps(hint, ensure_ascii=False, sort_keys=True)
    return text


def extract_official_code(content: str) -> str:
    code = extract_python_code_block(content)
    if code:
        return code
    text = content or ""
    if "=====" in text:
        start = text.find("=====")
        end = text.find("=====", start + 5)
        if end > start:
            return text[start + 5 : end].replace("```python", "").replace("```", "").strip()
    return text.strip()


def extract_optimus_code(content: str) -> str:
    text = content or ""
    start = text.find("=====")
    if start >= 0:
        end = text.find("=====", start + 5)
        if end > start:
            return text[start + 5 : end].replace("```python", "").replace("```", "").strip()
    return extract_official_code(text)


def execute_official_solver_code(code: str, *, solver: str) -> dict[str, Any]:
    if not code:
        return {"compile_success": False, "runtime_success": False, "compile_error": "missing official code", "runtime_error": ""}
    augmented = official_execution_prelude(solver) + code + official_objective_probe(solver)
    try:
        ast.parse(augmented)
        compile(augmented, "<official_solver_code>", "exec")
    except Exception as exc:  # noqa: BLE001
        return {"compile_success": False, "runtime_success": False, "compile_error": f"{type(exc).__name__}: {exc}", "runtime_error": ""}
    with tempfile.TemporaryDirectory(prefix="official_solver_") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "solver.py").write_text(augmented, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "solver.py"],
                cwd=tmp_path,
                text=True,
                capture_output=True,
                timeout=STATIC_CODE_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "compile_success": True,
                "runtime_success": False,
                "compile_error": "",
                "runtime_error": f"TimeoutExpired after {STATIC_CODE_TIMEOUT_SECONDS}s",
                "stdout_preview": (exc.stdout or "")[:1000] if isinstance(exc.stdout, str) else "",
                "stderr_preview": (exc.stderr or "")[:1000] if isinstance(exc.stderr, str) else "",
            }
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    objective = extract_official_objective(stdout)
    if objective is None:
        for filename in ("ref_optimal_value.txt", "output_solution.txt"):
            output_file = tmp_path / filename
            if output_file.exists():
                objective = first_numeric(output_file.read_text(encoding="utf-8", errors="ignore"))
                if objective is not None:
                    break
    return {
        "compile_success": True,
        "runtime_success": proc.returncode == 0,
        "compile_error": "",
        "runtime_error": "" if proc.returncode == 0 else f"returncode={proc.returncode}",
        "stdout_preview": stdout[:2000],
        "stderr_preview": stderr[:2000],
        "objective_value": objective,
        "output": {"objective_value": objective, "solution": {}},
    }


def official_objective_probe(solver: str) -> str:
    if solver == "coptpy":
        return """

try:
    _model = globals().get('model')
    if _model is None:
        for _candidate in list(globals().values()):
            if hasattr(_candidate, 'objval') and hasattr(_candidate, 'status'):
                _model = _candidate
                break
    if _model is not None:
        _obj = getattr(_model, 'objval', None)
        print(f"Just print the best solution: {_obj}" if _obj is not None else "No Best Solution")
except Exception:
    pass
"""
    return """

try:
    _model = globals().get('model') or globals().get('m')
    if _model is None:
        for _candidate in list(globals().values()):
            if hasattr(_candidate, 'objVal') and hasattr(_candidate, 'status'):
                _model = _candidate
                break
    if _model is not None:
        _obj = getattr(_model, 'objVal', None)
        print(f"Just print the best solution: {_obj}" if _obj is not None else "No Best Solution")
except Exception:
    pass
"""


def official_execution_prelude(solver: str) -> str:
    if solver != "coptpy":
        return ""
    return """
try:
    import coptpy as _liveopt_coptpy
    from coptpy import COPT as _LIVEOPT_COPT
    for _liveopt_name in ('MAXIMIZE', 'MINIMIZE', 'OPTIMAL', 'BINARY', 'INTEGER', 'CONTINUOUS'):
        if not hasattr(_liveopt_coptpy, _liveopt_name) and hasattr(_LIVEOPT_COPT, _liveopt_name):
            setattr(_liveopt_coptpy, _liveopt_name, getattr(_LIVEOPT_COPT, _liveopt_name))
    if not hasattr(_liveopt_coptpy, 'sum_'):
        setattr(_liveopt_coptpy, 'sum_', getattr(_liveopt_coptpy, 'quicksum', sum))
except Exception:
    pass

"""


def extract_official_objective(stdout: str) -> float | None:
    text = stdout or ""
    marker = "Just print the best solution:"
    if marker in text:
        tail = text.rsplit(marker, 1)[-1].strip().splitlines()[0]
        value = first_numeric(tail)
        if value is not None:
            return value
    for label in ["Optimal Objective Value:", "Best solution", "Optimal objective", "Total profit:", "objective:", "Objective:"]:
        if label in text:
            tail = text.rsplit(label, 1)[-1].strip().splitlines()[0]
            value = first_numeric(tail)
            if value is not None:
                return value
    numbers = re.findall(r"[-+]?(?:\\d+\\.\\d+|\\d+)(?:[eE][-+]?\\d+)?", text)
    return float(numbers[-1]) if numbers else None


def evaluate_official_execution(case: dict[str, Any], parsed: dict[str, Any], code_result: dict[str, Any]) -> dict[str, Any]:
    reference = case.get("reference") if isinstance(case.get("reference"), dict) else {}
    reference_solution = parse_reference_solution(reference)
    reference_objective = reference_solution.get("objective_value")
    if reference_objective is None:
        for key in ("en_answer", "answer", "objective", "objective_value", "solution"):
            reference_objective = first_numeric(reference.get(key))
            if reference_objective is not None:
                break
    predicted_objective = first_numeric(code_result.get("objective_value"))
    tolerance_abs, tolerance_rel = official_numeric_tolerance(case)
    objective = compare_numeric(reference_objective, predicted_objective, tolerance_abs=tolerance_abs, tolerance_rel=tolerance_rel)
    return {
        "schema_version": "static_solving_official_eval_v1",
        "status": "official_compared" if reference_objective is not None else "no_reference_objective",
        "reference_status": normalize_status(reference.get("solution_status") or reference.get("status")),
        "predicted_status": "optimal" if predicted_objective is not None else None,
        "status_exact": None,
        "code_available": bool(extract_solver_code(parsed)),
        "compile_success": bool(code_result.get("compile_success")),
        "runtime_success": bool(code_result.get("runtime_success")),
        "compile_error": code_result.get("compile_error", ""),
        "runtime_error": code_result.get("runtime_error", ""),
        "stdout_preview": code_result.get("stdout_preview", ""),
        "stderr_preview": code_result.get("stderr_preview", ""),
        "reference_objective": reference_objective,
        "predicted_objective": predicted_objective,
        "objective_exact": objective.get("match"),
        "objective_absolute_error": objective.get("absolute_error"),
        "objective_relative_error": objective.get("relative_error"),
        "objective_tolerance": objective.get("tolerance"),
        "reference_solution_available": False,
        "reference_solution_required": False,
        "predicted_solution_available": False,
        "solution_exact": None,
        "solution_mismatches": [],
        "official_accuracy": bool(code_result.get("compile_success")) and bool(code_result.get("runtime_success")) and objective.get("match") is True,
    }


def sum_usage(usages) -> dict[str, int]:
    total = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 0,
        "local_cache_hits": 0,
        "local_cache_saved_tokens": 0,
    }
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        normalized = normalize_usage(usage)
        for key in total:
            total[key] += int(normalized.get(key, 0) or 0)
    return total


SYSTEM_PROMPTS = {
    "liveopt_static": (
        "You are LiveOpt in static mode. Convert the public natural-language optimization task into a "
        "compact workbench plan: data assumptions, decision variables, objectives, constraints, encoding, "
        "solver choice, verification, and output format. Return JSON only."
    ),
    "optimai_2025": (
        "You are an OptimAI-style optimization agent. Formulate the task, propose multiple solver-code plans, "
        "select a plan, and include a debug/validation plan. Return JSON only."
    ),
    "or_llm_agent_2025": (
        "You are an OR-LLM-Agent-style agent. Produce reasoning/modeling, formulation, code-generation, "
        "self-verification, and self-repair artifacts for the public optimization task. Return JSON only."
    ),
    "nl4opt": (
        "You are an NL4Opt-style formulator. Extract entities and produce a linear or integer optimization "
        "form when supported by the public text. Return JSON only."
    ),
    "react_tools": (
        "You are a ReAct+Tools optimization agent. Use a Thought -> public tool-plan -> validation pattern, "
        "then return executable solver code and the final answer. Return JSON only."
    ),
    "persistent_react": (
        "You are a Persistent ReAct+Tools optimization agent. Use a Thought -> public tool-plan -> validation pattern, "
        "then return executable solver code and the final answer. "
        "You keep a persistent code workspace across stages: when your previous stage solver code and your previous "
        "accepted final answer are supplied, treat them as your own carried state. Apply the smallest necessary edits "
        "for the new public update while keeping parts that are still valid; when an update refers to earlier "
        "decisions, rely on the carried accepted plan. Always return the COMPLETE updated solver code, never a diff. "
        "You have no TSS workbench slots, no candidate population or archive, no restart interface, and no LiveOpt "
        "artifacts. Return JSON only."
    ),
    "react_transcript_state": (
        "You are a ReAct+Tools optimization agent with an explicit public transcript and accepted-state summary. "
        "Use only public state, public tables, public feedback, and public solver/verifier tools. Return JSON only."
    ),
    "react_generic_workbench": (
        "You are a ReAct+Tools optimization agent with a generic public Workbench interface. "
        "When a previous accepted solver artifact is supplied, modify that executable code locally and emit the complete revised code. "
        "Use fixed public solver tools, but no LiveOpt typed segments, private memory, search archives, or adaptive restart. Return JSON only."
    ),
    "react_public_delta_oracle": (
        "You are a ReAct+Tools optimization agent with a public delta summary. "
        "Use only deltas derived from visible user text and public tables; do not assume hidden evaluator state. Return JSON only."
    ),
    "optimai": (
        "You are an optimization modeling agent. Choose a solver path and provide a compact public formulation. "
        "Return JSON only."
    ),
    "orlm": (
        "You are an ORLM-style optimization modeler. Produce a mathematical model and COPT solver code. "
        "Return JSON only."
    ),
    "optimus": (
        "You are an OptiMUS-style optimization code generator. Produce Gurobi solver code and debug it. "
        "Return JSON only."
    ),
}


METHOD_REQUIRED_FIELDS = {
    "liveopt_static": [
        "problem_understanding",
        "data_assumptions",
        "decision_variables",
        "objectives",
        "constraints",
        "encoding_plan",
        "solver_plan",
        "verification_plan",
        "output_solution_format",
        "solver_code",
        "final_answer",
    ],
    "optimai_2025": [
        "formulation",
        "candidate_solver_plans",
        "selected_plan",
        "debug_validation_plan",
        "solver_code",
        "final_answer",
    ],
    "or_llm_agent_2025": [
        "reasoning_modeling_trace",
        "formulation",
        "code_generation_plan",
        "self_verification",
        "self_repair",
        "solver_code",
        "final_answer",
    ],
    "nl4opt": [
        "semantic_entities",
        "optimization_form",
        "solver_routing",
        "solver_code",
        "final_answer",
    ],
    "react_tools": [
        "thought",
        "react_trace",
        "tool_plan",
        "validation_plan",
        "solver_code",
        "final_answer",
    ],
    "persistent_react": [
        "thought",
        "react_trace",
        "tool_plan",
        "validation_plan",
        "solver_code",
        "final_answer",
    ],
    "react_transcript_state": [
        "thought",
        "public_state_reconstruction",
        "react_trace",
        "tool_plan",
        "validation_plan",
        "solver_code",
        "final_answer",
    ],
    "react_generic_workbench": [
        "thought",
        "generic_workbench_plan",
        "react_trace",
        "tool_plan",
        "validation_plan",
        "solver_code",
        "final_answer",
    ],
    "react_public_delta_oracle": [
        "thought",
        "public_delta_use",
        "react_trace",
        "tool_plan",
        "validation_plan",
        "solver_code",
        "final_answer",
    ],
    "optimai": [
        "formulation",
        "solver_choice",
        "solver_plan",
        "validation_plan",
        "solver_code",
        "final_answer",
    ],
    "orlm": [
        "formulation",
        "solver_choice",
        "solver_plan",
        "validation_plan",
        "solver_code",
        "final_answer",
    ],
    "optimus": [
        "formulation",
        "solver_choice",
        "solver_plan",
        "validation_plan",
        "solver_code",
        "final_answer",
    ],
}


def schema_for_method(method: str, compact: bool = False) -> dict[str, Any]:
    if compact and method == "optimai_2025":
        return {
            "formulation": "one short object with variables, objective, constraints",
            "candidate_solver_plans": [{"plan_id": "p1", "solver": "scipy.optimize", "implementation_notes": ["short"]}],
            "selected_plan": {"plan_id": "p1", "reason": "short"},
            "debug_validation_plan": ["short public checks"],
            "solver_code": {"language": "python", "code": "complete code; print JSON with objective_value and solution"},
            "final_answer": {"objective_value": "number|null", "solution_status": "optimal|infeasible|unbounded|no_optimal", "solution": {}},
        }
    if compact and method == "or_llm_agent_2025":
        return {
            "reasoning_modeling_trace": "short modeling summary",
            "formulation": "one short object with variables, objective, constraints",
            "code_generation_plan": "short scipy code plan",
            "self_verification": "short public verification checklist",
            "self_repair": "short repair rule if code fails",
            "solver_code": {"language": "python", "code": "complete code; print JSON with objective_value and solution"},
            "final_answer": {"objective_value": "number|null", "solution_status": "optimal|infeasible|unbounded|no_optimal", "solution": {}},
        }
    if compact and method == "liveopt_static":
        return {
            "problem_understanding": "short string",
            "data_assumptions": [],
            "decision_variables": [{"name": "x", "type": "binary|integer|continuous", "meaning": "short"}],
            "objectives": [{"sense": "min|max", "expression": "public expression"}],
            "constraints": [{"expression": "public expression", "type": "hard"}],
            "encoding_plan": [{"variable": "x", "encoding_skill": "continuous_vector|integer_vector|binary_vector|assignment"}],
            "solver_plan": {"solver": "LP|MILP|exact", "reason": "short"},
            "verification_plan": ["short public checks"],
            "output_solution_format": {"objective_value": "number", "solution": "object"},
            "solver_code": {"language": "python", "code": "complete code; print JSON with objective_value and solution"},
            "final_answer": {"objective_value": "number|null", "solution_status": "optimal|infeasible|unbounded|no_optimal", "solution": {}},
        }
    if method == "liveopt_static":
        return {
            "problem_understanding": "short string",
            "data_assumptions": [],
            "decision_variables": [{"name": "x", "type": "binary|integer|continuous|permutation|assignment", "meaning": "short"}],
            "objectives": [{"sense": "min|max", "expression": "public expression"}],
            "constraints": [{"expression": "public expression", "type": "hard|soft"}],
            "encoding_plan": [{"variable": "x", "encoding_skill": "choice_vector|assignment|permutation|continuous_vector|integer_vector|binary_vector"}],
            "solver_plan": {"solver": "LP|MILP|GA|NSGA-II|exact|hybrid", "reason": "short"},
            "verification_plan": [],
            "output_solution_format": {},
            "solver_code": {
                "language": "python",
                "code": "read parameters.json if present; solve; print JSON with objective_value and solution",
            },
            "final_answer": {"objective_value": "number|null", "solution": {}, "derivation": "short"},
        }
    if method == "optimai_2025":
        return {
            "formulation": {"variables": [], "objective": [], "constraints": []},
            "candidate_solver_plans": [{"plan_id": "p1", "solver": "LP|MILP|GA|NSGA-II", "implementation_notes": []}],
            "selected_plan": {"plan_id": "p1", "reason": "short"},
            "debug_validation_plan": [],
            "solver_code": {
                "language": "python",
                "code": "read parameters.json if present; solve; print JSON with objective_value and solution",
            },
            "final_answer": {"objective_value": "number|null", "solution": {}, "derivation": "short"},
        }
    if method == "or_llm_agent_2025":
        return {
            "reasoning_modeling_trace": {},
            "formulation": {"variables": [], "objective": [], "constraints": []},
            "code_generation_plan": {},
            "self_verification": {},
            "self_repair": {},
            "solver_code": {
                "language": "python",
                "code": "read parameters.json if present; solve; print JSON with objective_value and solution",
            },
            "final_answer": {"objective_value": "number|null", "solution": {}, "derivation": "short"},
        }
    if method == "nl4opt":
        return {
            "semantic_entities": {"variables": [], "parameters": [], "sets": []},
            "optimization_form": {"objective": "", "constraints": []},
            "solver_routing": {"solver": "LP|MILP|unknown", "reason": "short"},
            "solver_code": {
                "language": "python",
                "code": "read parameters.json if present; solve; print JSON with objective_value and solution",
            },
            "final_answer": {"objective_value": "number|null", "solution": {}, "derivation": "short"},
        }
    if method == "react_tools":
        return {
            "thought": "short public reasoning summary",
            "react_trace": [
                {"thought": "inspect public text/parameters", "action": "choose scipy solver or direct computation", "observation": "public-only"}
            ],
            "tool_plan": [{"tool": "scipy.optimize|direct_python", "purpose": "solve the public static model"}],
            "validation_plan": ["compile", "run", "check objective and solution schema"],
            "solver_code": {
                "language": "python",
                "code": "read parameters.json if present; solve; print JSON with objective_value and solution",
            },
            "final_answer": {"objective_value": "number|null", "solution": {}, "derivation": "short"},
        }
    if method == "persistent_react":
        return {
            "thought": "short public reasoning summary covering what the new update changes and which carried state is reused",
            "react_trace": [
                {"thought": "inspect carried solver code, carried accepted plan, and the new public update", "action": "edit the previous solver code minimally, or rewrite only the parts invalidated by the update", "observation": "public-only"}
            ],
            "tool_plan": [{"tool": "scipy.optimize|direct_python", "purpose": "solve the current public stage"}],
            "validation_plan": ["compile", "run", "check objective and solution schema"],
            "solver_code": {
                "language": "python",
                "code": "complete updated code (never a diff); read parameters.json if present; solve; print JSON with objective_value and solution",
            },
            "final_answer": {"objective_value": "number|null", "solution": {}, "derivation": "short"},
        }
    if method == "react_transcript_state":
        return {
            "thought": "short public reasoning summary",
            "public_state_reconstruction": {
                "current_stage": "short",
                "previous_accepted_output_used": "yes|no",
                "active_public_requirements": [],
            },
            "react_trace": [
                {"thought": "reconstruct public state", "action": "choose scipy solver or direct computation", "observation": "public-only"}
            ],
            "tool_plan": [{"tool": "scipy.optimize|direct_python", "purpose": "solve the current public state"}],
            "validation_plan": ["compile", "run", "check objective and solution schema"],
            "solver_code": {
                "language": "python",
                "code": "read parameters.json if present; solve; print JSON with objective_value and solution",
            },
            "final_answer": {"objective_value": "number|null", "solution": {}, "derivation": "short"},
        }
    if method == "react_generic_workbench":
        return {
            "thought": "short public reasoning summary",
            "generic_workbench_plan": {
                "data_needed": [],
                "decision_segments": [],
                "solver_or_search": "LP|MILP|scalar_ea|pareto_ea|direct_python",
                "fixed_restart_policy": "full_restart_or_none",
            },
            "react_trace": [
                {"thought": "map request to public workbench", "action": "generate solver code", "observation": "public-only"}
            ],
            "tool_plan": [{"tool": "scipy.optimize|direct_python|public_ga", "purpose": "solve current public stage"}],
            "validation_plan": ["compile", "run", "check objective and solution schema"],
            "solver_code": {
                "language": "python",
                "code": "read parameters.json if present; solve; print JSON with objective_value, solution, and optional candidate_archive",
            },
            "final_answer": {"objective_value": "number|null", "solution": {}, "candidate_archive": [], "derivation": "short"},
        }
    if method == "react_public_delta_oracle":
        return {
            "thought": "short public reasoning summary",
            "public_delta_use": {
                "affected_public_text": "short",
                "affected_tables_or_fields": [],
                "affected_objectives_or_constraints": [],
            },
            "react_trace": [
                {"thought": "apply public delta summary", "action": "generate solver code", "observation": "public-only"}
            ],
            "tool_plan": [{"tool": "scipy.optimize|direct_python|public_ga", "purpose": "solve current public stage"}],
            "validation_plan": ["compile", "run", "check objective and solution schema"],
            "solver_code": {
                "language": "python",
                "code": "read parameters.json if present; solve; print JSON with objective_value, solution, and optional candidate_archive",
            },
            "final_answer": {"objective_value": "number|null", "solution": {}, "candidate_archive": [], "derivation": "short"},
        }
    return {
        "formulation": {"variables": [], "objective": [], "constraints": []},
        "solver_choice": "LP|MILP|GA|NSGA-II|exact|unknown",
        "solver_plan": {},
        "validation_plan": [],
        "solver_code": {
            "language": "python",
            "code": "read parameters.json if present; solve; print JSON with objective_value and solution",
        },
        "final_answer": {"objective_value": "number|null", "solution": {}, "derivation": "short"},
    }


def parse_json_object(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = extract_first_json_object(text)
    if not isinstance(value, dict):
        raise ValueError("response must be a JSON object")
    if isinstance(value.get("required_output_schema"), dict):
        nested = value["required_output_schema"]
        if any(key in nested for key in ("final_answer", "formulation", "decision_variables", "reasoning_modeling_trace")):
            return nested
    return value


def parse_model_response(content: str, method: str, protocol: str = "json") -> dict[str, Any]:
    if protocol == "code_block":
        stripped = (content or "").strip()
        if stripped.startswith("```python") or stripped.startswith("```py"):
            code = extract_python_code_block(content)
            if code:
                return parsed_output_from_code_block(method, code)
        if stripped.startswith("{") or stripped.startswith("```json"):
            try:
                return parse_json_object(content)
            except Exception:
                pass
        code = extract_python_code_block(content)
        if not code:
            raise ValueError("code_block response did not contain Python code or JSON")
        return parsed_output_from_code_block(method, code)
    return parse_json_object(content)


def extract_python_code_block(content: str) -> str:
    text = content or ""
    fence = "```"
    start = text.find(fence)
    while start >= 0:
        line_end = text.find("\n", start + len(fence))
        if line_end < 0:
            break
        language = text[start + len(fence) : line_end].strip().lower()
        end = text.find(fence, line_end + 1)
        if end < 0:
            break
        body = text[line_end + 1 : end].strip()
        if language in {"python", "py", ""} and body:
            return body
        start = text.find(fence, end + len(fence))
    stripped = text.strip()
    if stripped.startswith("import ") or "\nimport " in stripped or "print(" in stripped:
        return stripped
    return ""


def parsed_output_from_code_block(method: str, code: str) -> dict[str, Any]:
    common_solver_code = {"language": "python", "code": code}
    common_final_answer = {"objective_value": None, "solution_status": None, "solution": {}, "derivation": "code-block protocol"}
    if method == "optimai_2025":
        return {
            "formulation": {"source": "code_block_protocol"},
            "candidate_solver_plans": [{"plan_id": "p1", "solver": "scipy.optimize", "implementation_notes": ["generated code block"]}],
            "selected_plan": {"plan_id": "p1", "reason": "single official-style code-block plan"},
            "debug_validation_plan": ["compile generated Python", "run code", "compare objective"],
            "solver_code": common_solver_code,
            "final_answer": common_final_answer,
        }
    if method == "or_llm_agent_2025":
        return {
            "reasoning_modeling_trace": {"source": "code_block_protocol"},
            "formulation": {"source": "code_block_protocol"},
            "code_generation_plan": {"solver": "scipy.optimize"},
            "self_verification": {"checks": ["compile", "run", "compare objective"]},
            "self_repair": {"policy": "not invoked in single code-block response"},
            "solver_code": common_solver_code,
            "final_answer": common_final_answer,
        }
    if method == "liveopt_static":
        return {
            "problem_understanding": "code-block protocol",
            "data_assumptions": [],
            "decision_variables": [],
            "objectives": [],
            "constraints": [],
            "encoding_plan": [],
            "solver_plan": {"solver": "scipy.optimize", "reason": "single generated code block"},
            "verification_plan": ["compile", "run", "compare objective"],
            "output_solution_format": {"objective_value": "number", "solution": "object"},
            "solver_code": common_solver_code,
            "final_answer": common_final_answer,
        }
    if method == "react_tools":
        return {
            "thought": "code-block protocol",
            "react_trace": [
                {"thought": "solve the public static request", "action": "generated_solver_code", "observation": "compile/run/evaluate"}
            ],
            "tool_plan": [{"tool": "python_scipy_or_direct", "purpose": "solve public static optimization"}],
            "validation_plan": ["compile generated Python", "run code", "compare objective"],
            "solver_code": common_solver_code,
            "final_answer": common_final_answer,
        }
    if method == "persistent_react":
        return {
            "thought": "code-block protocol",
            "react_trace": [
                {"thought": "edit carried solver code for the current public stage", "action": "generated_solver_code", "observation": "compile/run/evaluate"}
            ],
            "tool_plan": [{"tool": "python_scipy_or_direct", "purpose": "solve current public stage"}],
            "validation_plan": ["compile generated Python", "run code", "compare objective"],
            "solver_code": common_solver_code,
            "final_answer": common_final_answer,
        }
    if method == "react_transcript_state":
        return {
            "thought": "code-block protocol",
            "public_state_reconstruction": {"source": "code_block_protocol"},
            "react_trace": [
                {"thought": "solve current public state", "action": "generated_solver_code", "observation": "compile/run/evaluate"}
            ],
            "tool_plan": [{"tool": "python_scipy_or_direct", "purpose": "solve current public optimization state"}],
            "validation_plan": ["compile generated Python", "run code", "compare objective"],
            "solver_code": common_solver_code,
            "final_answer": common_final_answer,
        }
    if method == "react_generic_workbench":
        return {
            "thought": "code-block protocol",
            "generic_workbench_plan": {"source": "code_block_protocol"},
            "react_trace": [
                {"thought": "use generic workbench", "action": "generated_solver_code", "observation": "compile/run/evaluate"}
            ],
            "tool_plan": [{"tool": "python_scipy_or_direct", "purpose": "solve current public stage"}],
            "validation_plan": ["compile generated Python", "run code", "compare objective"],
            "solver_code": common_solver_code,
            "final_answer": common_final_answer,
        }
    if method == "react_public_delta_oracle":
        return {
            "thought": "code-block protocol",
            "public_delta_use": {"source": "code_block_protocol"},
            "react_trace": [
                {"thought": "use public delta summary", "action": "generated_solver_code", "observation": "compile/run/evaluate"}
            ],
            "tool_plan": [{"tool": "python_scipy_or_direct", "purpose": "solve current public stage"}],
            "validation_plan": ["compile generated Python", "run code", "compare objective"],
            "solver_code": common_solver_code,
            "final_answer": common_final_answer,
        }
    return {
        "formulation": {"source": "code_block_protocol"},
        "solver_choice": "scipy.optimize",
        "solver_plan": {"source": "single generated code block"},
        "validation_plan": ["compile", "run", "compare objective"],
        "solver_code": common_solver_code,
        "final_answer": common_final_answer,
    }


def extract_first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("response did not contain a JSON object")


def validate_parsed_output(method: str, parsed: dict[str, Any]) -> list[str]:
    feedback: list[str] = []
    for field in METHOD_REQUIRED_FIELDS[method]:
        if field not in parsed:
            feedback.append(f"missing required field: {field}")
    if method == "optimai_2025" and not isinstance(parsed.get("candidate_solver_plans"), list):
        feedback.append("candidate_solver_plans must be a list")
    if method == "liveopt_static" and not isinstance(parsed.get("decision_variables"), list):
        feedback.append("decision_variables must be a list")
    if "final_answer" in parsed and not isinstance(parsed.get("final_answer"), dict):
        feedback.append("final_answer must be an object")
    solver_code = extract_solver_code(parsed)
    if "solver_code" in METHOD_REQUIRED_FIELDS[method] and not solver_code:
        feedback.append("solver_code.code must be a non-empty Python source string")
    return feedback


def evaluate_reference(case: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    reference = case.get("reference") if isinstance(case.get("reference"), dict) else {}
    reference_value = first_numeric(reference.get("en_answer"))
    if reference_value is None:
        reference_value = first_numeric(reference.get("answer"))
    if reference_value is None:
        reference_value = first_numeric(reference.get("objective"))
    if reference_value is None:
        reference_value = first_numeric(reference.get("objective_value"))
    if reference_value is None:
        reference_value = first_numeric(reference.get("solution"))
    prediction = parsed.get("final_answer") if isinstance(parsed.get("final_answer"), dict) else {}
    predicted_value = first_numeric(prediction.get("objective_value"))
    if predicted_value is None:
        predicted_value = first_numeric(prediction.get("answer"))
    if reference_value is None:
        return {"status": "no_numeric_reference"}
    if predicted_value is None:
        return {"status": "missing_numeric_prediction", "reference_value": reference_value}
    error = abs(predicted_value - reference_value)
    tolerance = max(1e-6, 1e-6 * abs(reference_value))
    return {
        "status": "numeric_compared",
        "reference_value": reference_value,
        "predicted_value": predicted_value,
        "absolute_error": error,
        "relative_error": error / max(1.0, abs(reference_value)),
        "numeric_match": error <= tolerance,
        "tolerance": tolerance,
    }


def evaluate_official_static(case: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    """Evaluate static solving rows with NLP4LP/OptiMUS-style executable criteria.

    Official-style accuracy is stricter than answer-only scoring: the submitted
    code must compile, run, print a JSON solution, match the reference objective,
    and match the reference variable assignment when a reference solution exists.
    """

    reference = case.get("reference") if isinstance(case.get("reference"), dict) else {}
    reference_status = normalize_status(reference.get("solution_status") or reference.get("status"))
    reference_solution = parse_reference_solution(reference)
    reference_objective = reference_solution.get("objective_value")
    reference_variables = reference_solution.get("solution")
    if reference_objective is None:
        reference_objective = first_numeric(reference.get("en_answer"))
    if reference_objective is None:
        reference_objective = first_numeric(reference.get("answer"))
    if reference_objective is None:
        reference_objective = first_numeric(reference.get("objective"))
    if reference_objective is None:
        reference_objective = first_numeric(reference.get("objective_value"))
    if reference_objective is None:
        reference_objective = first_numeric(reference.get("solution"))

    code = extract_solver_code(parsed)
    code_result = execute_static_solver_code(code, case)
    execution_prediction = code_result.get("output") if isinstance(code_result.get("output"), dict) else {}
    if not execution_prediction:
        execution_prediction = parsed.get("final_answer") if isinstance(parsed.get("final_answer"), dict) else {}

    predicted_status = normalize_status(execution_prediction.get("solution_status") or execution_prediction.get("status"))
    predicted_objective = first_numeric(execution_prediction.get("objective_value"))
    if predicted_objective is None:
        predicted_objective = first_numeric(execution_prediction.get("objective"))
    if predicted_objective is None:
        predicted_objective = first_numeric(execution_prediction.get("answer"))
    predicted_solution = extract_prediction_solution(execution_prediction)

    tolerance_abs, tolerance_rel = official_numeric_tolerance(case)
    objective = compare_numeric(reference_objective, predicted_objective, tolerance_abs=tolerance_abs, tolerance_rel=tolerance_rel)
    solution = compare_solution(reference_variables, predicted_solution)
    status = compare_solution_status(reference_status, predicted_status)
    has_reference_solution = reference_variables is not None
    requires_solution = bool((case.get("metadata") or {}).get("reference_solution_required", has_reference_solution))
    no_optimal_reference = reference_status in NO_OPTIMAL_STATUSES
    if no_optimal_reference:
        target_match = status.get("match") is True
    elif requires_solution:
        target_match = objective.get("match") is True and solution.get("match") is True
    else:
        target_match = objective.get("match") is True
    official_accuracy = code_result.get("compile_success") is True and code_result.get("runtime_success") is True and target_match
    return {
        "schema_version": "static_solving_official_eval_v1",
        "status": "official_compared" if reference_objective is not None else "no_reference_objective",
        "reference_status": reference_status,
        "predicted_status": predicted_status,
        "status_exact": status.get("match"),
        "code_available": bool(code),
        "compile_success": bool(code_result.get("compile_success")),
        "runtime_success": bool(code_result.get("runtime_success")),
        "compile_error": code_result.get("compile_error", ""),
        "runtime_error": code_result.get("runtime_error", ""),
        "stdout_preview": code_result.get("stdout_preview", ""),
        "stderr_preview": code_result.get("stderr_preview", ""),
        "reference_objective": reference_objective,
        "predicted_objective": predicted_objective,
        "objective_exact": objective.get("match"),
        "objective_absolute_error": objective.get("absolute_error"),
        "objective_relative_error": objective.get("relative_error"),
        "objective_tolerance": objective.get("tolerance"),
        "reference_solution_available": reference_variables is not None,
        "reference_solution_required": requires_solution,
        "predicted_solution_available": predicted_solution is not None,
        "solution_exact": solution.get("match") if has_reference_solution else None,
        "solution_mismatches": solution.get("mismatches", []),
        "official_accuracy": bool(official_accuracy),
    }


def extract_solver_code(parsed: dict[str, Any]) -> str:
    raw = parsed.get("solver_code")
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        for key in ("code", "source", "python_code", "solver_code"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for key in ("code", "python_code", "source_code"):
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def execute_and_repair_solver_code(
    *,
    case: dict[str, Any],
    method: str,
    parsed: dict[str, Any],
    client: DeepSeekDebugClient,
    args: argparse.Namespace,
    base_messages: list[dict[str, str]],
    initial_validation_feedback: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Run generated solver code and apply equal public code-output repair.

    The repair loop is deliberately narrow: it receives only public prompt
    material, the current code, schema feedback, and compile/runtime/stdout
    errors. It must not see reference answers or hidden evaluator feedback.
    """

    max_attempts = max(0, int(getattr(args, "code_repair_attempts", 0) or 0))
    current = dict(parsed) if isinstance(parsed, dict) else {}
    code = extract_solver_code(current)
    code_result = execute_static_solver_code(code, case)
    attempts: list[dict[str, Any]] = []
    feedback = code_output_repair_feedback(code_result, initial_validation_feedback or [])
    attempt_index = 0
    while attempt_index < max_attempts and feedback:
        attempt_index += 1
        repair_messages = build_code_output_repair_messages(
            case=case,
            method=method,
            base_messages=base_messages,
            parsed=current,
            code=code,
            code_result=code_result,
            validation_feedback=initial_validation_feedback or [],
            attempt_index=attempt_index,
            max_attempts=max_attempts,
        )
        started = time.perf_counter()
        response = client.chat(
            repair_messages,
            temperature=0.0,
            max_tokens=getattr(args, "max_tokens", 5000),
            json_mode=False,
        )
        latency = time.perf_counter() - started
        raw_response = response.get("choices", [{}])[0].get("message", {}).get("content", "") if isinstance(response, dict) else ""
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        repaired_code = ""
        parse_error = ""
        try:
            repaired = parse_model_response(raw_response, method, protocol="code_block")
            repaired_code = extract_solver_code(repaired)
        except Exception as exc:  # noqa: BLE001
            parse_error = f"{type(exc).__name__}: {exc}"
        if repaired_code:
            candidate = patch_parsed_solver_code(current, repaired_code)
            candidate_result = execute_static_solver_code(repaired_code, case)
        else:
            candidate = current
            candidate_result = {
                "compile_success": False,
                "runtime_success": False,
                "compile_error": parse_error or "repair response did not contain solver code",
                "runtime_error": "",
            }
        attempts.append(
            {
                "attempt": attempt_index,
                "policy": "equal_public_code_output_repair_v1",
                "messages": repair_messages,
                "raw_response": raw_response,
                "token_usage": normalize_usage(usage),
                "latency_seconds": latency,
                "code_available": bool(repaired_code),
                "parse_error": parse_error,
                "code_execution": compact_static_code_execution(candidate_result),
                "hidden_feedback_used": False,
                "reference_feedback_used": False,
            }
        )
        current = candidate
        code = extract_solver_code(current)
        code_result = candidate_result
        feedback = code_output_repair_feedback(code_result, validate_parsed_output(method, current))
    current = attach_execution_output_to_final_answer(current, code_result)
    return current, code_result, attempts


def code_output_repair_feedback(code_result: dict[str, Any], validation_feedback: list[str]) -> list[str]:
    feedback = [
        item
        for item in (validation_feedback or [])
        if "solver_code" in str(item) or "code" in str(item).lower()
    ]
    if not code_result.get("compile_success"):
        feedback.append(str(code_result.get("compile_error") or "compile_failed"))
    elif not code_result.get("runtime_success"):
        feedback.append(str(code_result.get("runtime_error") or "runtime_failed"))
        stdout = str(code_result.get("stdout_preview") or "").strip()
        stderr = str(code_result.get("stderr_preview") or "").strip()
        if stdout:
            feedback.append(f"stdout: {stdout[:600]}")
        if stderr:
            feedback.append(f"stderr: {stderr[:600]}")
    return [item for item in feedback if item]


def build_code_output_repair_messages(
    *,
    case: dict[str, Any],
    method: str,
    base_messages: list[dict[str, str]],
    parsed: dict[str, Any],
    code: str,
    code_result: dict[str, Any],
    validation_feedback: list[str],
    attempt_index: int,
    max_attempts: int,
) -> list[dict[str, str]]:
    public_context = compact_public_context(case.get("public_context"))
    payload = {
        "repair_policy": "equal_public_code_output_repair_v1",
        "attempt": attempt_index,
        "max_attempts": max_attempts,
        "method": method,
        "problem_text": case.get("problem_text", ""),
        "public_context": public_context,
        "current_solver_code": code,
        "current_public_fields": {
            "final_answer": parsed.get("final_answer") if isinstance(parsed, dict) else {},
            "solver_plan": parsed.get("solver_plan") if isinstance(parsed, dict) else {},
            "formulation": parsed.get("formulation") if isinstance(parsed, dict) else {},
        },
        "validation_feedback": validation_feedback,
        "code_execution": compact_static_code_execution(code_result),
        "rules": [
            "Repair only Python code and stdout JSON formatting.",
            "Do not change the public objective, constraints, data interpretation, or modeling assumptions unless needed to make the existing code executable.",
            "Do not use reference answers, hidden evaluator feedback, hidden deltas, benchmark ids, or private modules.",
            "Use only standard library and scipy.optimize.",
            "The code must print exactly one JSON object with objective_value and solution for optimal cases.",
            "If the public problem is infeasible or unbounded, print solution_status as infeasible, unbounded, or no_optimal.",
            "Return only one Python code block.",
        ],
    }
    original_system = base_messages[0]["content"] if base_messages else "You repair public optimization solver code."
    return [
        {
            "role": "system",
            "content": (
                original_system
                + "\nYou are now in a bounded public code-output repair step. "
                "You may repair code execution and JSON output only."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


def patch_parsed_solver_code(parsed: dict[str, Any], code: str) -> dict[str, Any]:
    out = dict(parsed) if isinstance(parsed, dict) else {}
    solver_code = out.get("solver_code")
    if isinstance(solver_code, dict):
        solver_code = dict(solver_code)
        solver_code["code"] = code
        solver_code.setdefault("language", "python")
        out["solver_code"] = solver_code
    else:
        out["solver_code"] = {"language": "python", "code": code}
    return out


def attach_execution_output_to_final_answer(parsed: dict[str, Any], code_result: dict[str, Any]) -> dict[str, Any]:
    out = dict(parsed) if isinstance(parsed, dict) else {}
    output = code_result.get("output") if isinstance(code_result.get("output"), dict) else {}
    if output:
        final_answer = out.get("final_answer") if isinstance(out.get("final_answer"), dict) else {}
        merged = dict(final_answer)
        for key in ("objective_value", "objective", "answer", "solution", "solution_status", "status", "candidate_archive"):
            if key in output:
                merged[key] = output[key]
        out["final_answer"] = merged
    return out


def compact_static_code_execution(value: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "compile_success",
        "runtime_success",
        "compile_error",
        "runtime_error",
        "stdout_preview",
        "stderr_preview",
        "output",
    ]
    return {key: value.get(key) for key in keys if key in value}


def execute_static_solver_code(code: str, case: dict[str, Any]) -> dict[str, Any]:
    if not code:
        return {
            "compile_success": False,
            "runtime_success": False,
            "compile_error": "missing solver_code",
            "runtime_error": "",
        }
    try:
        ast.parse(code)
        compile(code, "<static_solver_code>", "exec")
    except Exception as exc:  # noqa: BLE001
        return {
            "compile_success": False,
            "runtime_success": False,
            "compile_error": f"{type(exc).__name__}: {exc}",
            "runtime_error": "",
        }
    public_context = case.get("public_context") if isinstance(case.get("public_context"), dict) else {}
    parameters = public_context.get("parameters") if isinstance(public_context.get("parameters"), dict) else {}
    with tempfile.TemporaryDirectory(prefix="liveopt_static_solver_") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "solver.py").write_text(code, encoding="utf-8")
        (tmp_path / "parameters.json").write_text(json.dumps(parameters, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "solver.py"],
                cwd=tmp_path,
                text=True,
                capture_output=True,
                timeout=STATIC_CODE_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "compile_success": True,
                "runtime_success": False,
                "compile_error": "",
                "runtime_error": f"TimeoutExpired after {STATIC_CODE_TIMEOUT_SECONDS}s",
                "stdout_preview": (exc.stdout or "")[:1000] if isinstance(exc.stdout, str) else "",
                "stderr_preview": (exc.stderr or "")[:1000] if isinstance(exc.stderr, str) else "",
            }
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        return {
            "compile_success": True,
            "runtime_success": False,
            "compile_error": "",
            "runtime_error": f"returncode={proc.returncode}",
            "stdout_preview": stdout[:1000],
            "stderr_preview": stderr[:1000],
        }
    output = parse_execution_json(stdout)
    if not isinstance(output, dict):
        return {
            "compile_success": True,
            "runtime_success": False,
            "compile_error": "",
            "runtime_error": "stdout did not contain a JSON object",
            "stdout_preview": stdout[:1000],
            "stderr_preview": stderr[:1000],
        }
    return {
        "compile_success": True,
        "runtime_success": True,
        "compile_error": "",
        "runtime_error": "",
        "stdout_preview": stdout[:1000],
        "stderr_preview": stderr[:1000],
        "output": output,
    }


def parse_execution_json(stdout: str) -> dict[str, Any] | None:
    text = (stdout or "").strip()
    if not text:
        return None
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None
    return None


def parse_reference_solution(reference: dict[str, Any]) -> dict[str, Any]:
    raw = reference.get("solution")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = None
    if not isinstance(raw, dict):
        return {}
    objective = first_numeric(raw.get("objective"))
    if objective is None:
        objective = first_numeric(raw.get("objective_value"))
    variables = raw.get("variables")
    if variables is None:
        variables = raw.get("solution")
    return {
        "objective_value": objective,
        "solution": normalize_solution_shape(variables) if variables is not None else None,
    }


def solution_schema_hint_from_reference(reference: Any) -> Any:
    """Expose solution field names and shapes without exposing answer values."""
    if not isinstance(reference, dict):
        return {}
    parsed = parse_reference_solution(reference)
    solution = parsed.get("solution")
    if solution is None:
        return {}
    return value_schema_hint(solution)


def value_schema_hint(value: Any, *, depth: int = 0, max_items: int = 8) -> Any:
    if depth > 4:
        return "..."
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                out["..."] = f"{max(0, len(value) - max_items)} more fields"
                break
            out[str(key)] = value_schema_hint(item, depth=depth + 1, max_items=max_items)
        return out
    if isinstance(value, list):
        if not value:
            return []
        sample = value_schema_hint(value[0], depth=depth + 1, max_items=max_items)
        if all(value_schema_hint(item, depth=depth + 1, max_items=max_items) == sample for item in value[:max_items]):
            return [sample, f"... length {len(value)}"]
        return [value_schema_hint(item, depth=depth + 1, max_items=max_items) for item in value[:max_items]]
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if value is None:
        return "nullable"
    return "string"


def extract_prediction_solution(prediction: dict[str, Any]) -> Any:
    if not isinstance(prediction, dict):
        return None
    value = prediction.get("solution")
    if isinstance(value, dict) and "variables" in value:
        value = value.get("variables")
    if value is None:
        value = prediction.get("variables")
    if value is None:
        ignored = {"objective", "objective_value", "answer", "derivation", "status"}
        candidates = {key: val for key, val in prediction.items() if key not in ignored}
        value = candidates or None
    return normalize_solution_shape(value) if value is not None else None


def normalize_solution_shape(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text and text[0] in "{[":
            try:
                return normalize_solution_shape(json.loads(text))
            except Exception:
                return text
        return text
    if isinstance(value, dict):
        return {str(key): normalize_solution_shape(val) for key, val in value.items()}
    if isinstance(value, list):
        return [normalize_solution_shape(item) for item in value]
    if isinstance(value, (int, float)):
        return float(value)
    return value


NO_OPTIMAL_STATUSES = {
    "no_optimal",
    "nooptimum",
    "no_optimum",
    "nooptimal",
    "infeasible",
    "unbounded",
    "infeasibleorunbounded",
    "infeasible_or_unbounded",
}


def official_numeric_tolerance(case: dict[str, Any]) -> tuple[float | None, float | None]:
    metadata = case.get("metadata") if isinstance(case.get("metadata"), dict) else {}
    abs_value = first_numeric(metadata.get("official_tolerance_abs"))
    rel_value = first_numeric(metadata.get("official_tolerance_rel"))
    return abs_value, rel_value


def normalize_status(value: Any) -> str | None:
    if value is None:
        return None
    return "".join(ch for ch in str(value).lower() if ch.isalnum() or ch == "_").replace("__", "_")


def compare_solution_status(reference_status: str | None, predicted_status: str | None) -> dict[str, Any]:
    if reference_status is None:
        return {"status": "no_reference_status", "match": None}
    if reference_status in NO_OPTIMAL_STATUSES:
        return {
            "status": "compared",
            "match": predicted_status in NO_OPTIMAL_STATUSES,
        }
    if predicted_status is None:
        return {"status": "missing_prediction_status", "match": None}
    return {"status": "compared", "match": reference_status == predicted_status}


def compare_numeric(
    reference_value: float | None,
    predicted_value: float | None,
    *,
    tolerance_abs: float | None = None,
    tolerance_rel: float | None = None,
) -> dict[str, Any]:
    if reference_value is None:
        return {"status": "no_reference", "match": None}
    if predicted_value is None:
        return {"status": "missing_prediction", "match": False}
    error = abs(float(predicted_value) - float(reference_value))
    abs_tol = 1e-6 if tolerance_abs is None else max(0.0, float(tolerance_abs))
    rel_tol = 1e-6 if tolerance_rel is None else max(0.0, float(tolerance_rel))
    tolerance = max(abs_tol, rel_tol * abs(float(reference_value)))
    return {
        "status": "compared",
        "absolute_error": error,
        "relative_error": error / max(1.0, abs(float(reference_value))),
        "tolerance": tolerance,
        "match": error <= tolerance,
    }


def compare_solution(reference_solution: Any, predicted_solution: Any) -> dict[str, Any]:
    if reference_solution is None:
        return {"status": "no_reference_solution", "match": True, "mismatches": []}
    if predicted_solution is None:
        return {"status": "missing_prediction_solution", "match": False, "mismatches": ["missing solution"]}
    mismatches: list[str] = []
    compare_solution_value(reference_solution, predicted_solution, "solution", mismatches)
    return {"status": "compared", "match": not mismatches, "mismatches": mismatches[:20]}


def compare_solution_value(reference: Any, predicted: Any, path: str, mismatches: list[str]) -> None:
    if isinstance(reference, dict):
        pred_map = predicted if isinstance(predicted, dict) else list_to_index_map(predicted)
        if not isinstance(pred_map, dict):
            mismatches.append(f"{path}: expected object")
            return
        normalized_pred = {normalize_key(key): val for key, val in pred_map.items()}
        used_pred_keys: set[str] = set()
        missing: list[tuple[Any, Any]] = []
        for key, ref_val in reference.items():
            pred_key = normalize_key(key)
            matched_key = pred_key if pred_key in normalized_pred else find_alias_key(pred_key, normalized_pred, used_pred_keys)
            if matched_key is None:
                missing.append((key, ref_val))
                continue
            used_pred_keys.add(matched_key)
            compare_solution_value(ref_val, normalized_pred[matched_key], f"{path}.{key}", mismatches)
        if missing:
            ref_nums = [first_numeric(value) for _, value in missing]
            unused_nums = [first_numeric(value) for key, value in normalized_pred.items() if key not in used_pred_keys]
            if len(missing) == len(unused_nums) and numeric_multisets_match(ref_nums, unused_nums):
                return
            for key, _ in missing:
                mismatches.append(f"{path}.{key}: missing")
        return
    if isinstance(reference, list):
        if not isinstance(predicted, list):
            pred_map = predicted if isinstance(predicted, dict) else {}
            predicted = [pred_map.get(str(i)) for i in range(len(reference))] if isinstance(pred_map, dict) else predicted
        if not isinstance(predicted, list):
            mismatches.append(f"{path}: expected list")
            return
        if len(predicted) != len(reference):
            mismatches.append(f"{path}: length {len(predicted)} != {len(reference)}")
            return
        for index, (ref_item, pred_item) in enumerate(zip(reference, predicted)):
            compare_solution_value(ref_item, pred_item, f"{path}[{index}]", mismatches)
        return
    ref_num = first_numeric(reference)
    pred_num = first_numeric(predicted)
    if ref_num is not None or pred_num is not None:
        cmp = compare_numeric(ref_num, pred_num)
        if cmp.get("match") is not True:
            mismatches.append(f"{path}: {predicted} != {reference}")
        return
    if str(reference) != str(predicted):
        mismatches.append(f"{path}: {predicted!r} != {reference!r}")


def list_to_index_map(value: Any) -> dict[str, Any] | None:
    if isinstance(value, list):
        return {str(index): item for index, item in enumerate(value)}
    return None


def normalize_key(key: Any) -> str:
    return "".join(ch for ch in str(key).lower() if ch.isalnum())


def find_alias_key(reference_key: str, candidates: dict[str, Any], used: set[str]) -> str | None:
    for key in candidates:
        if key in used:
            continue
        if len(key) >= 3 and (reference_key.endswith(key) or key.endswith(reference_key) or key in reference_key or reference_key in key):
            return key
    return None


def numeric_multisets_match(left: list[float | None], right: list[float | None]) -> bool:
    if any(value is None for value in left) or any(value is None for value in right):
        return False
    if len(left) != len(right):
        return False
    left_sorted = sorted(float(value) for value in left if value is not None)
    right_sorted = sorted(float(value) for value in right if value is not None)
    return all(compare_numeric(a, b).get("match") is True for a, b in zip(left_sorted, right_sorted))


def first_numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            text = value.strip()
            if text and text[0] in "{[":
                try:
                    return first_numeric(json.loads(text))
                except Exception:
                    return None
            return None
    if isinstance(value, dict):
        for key in ("objective_value", "answer", "value", "optimum"):
            result = first_numeric(value.get(key))
            if result is not None:
                return result
        result = first_numeric(value.get("objective"))
        if result is not None:
            return result
    return None


def compact_public_context(value: Any) -> Any:
    if not isinstance(value, dict):
        return {}
    allowed = {}
    for key in ("parameters", "metadata", "keywords", "type"):
        if key in value:
            allowed[key] = value[key]
    return allowed


def write_trace(
    trace_dir: Path,
    case: dict[str, Any],
    method: str,
    messages: list[dict[str, str]],
    raw_response: str,
    parsed: dict[str, Any],
    usage: dict[str, Any],
    status: str,
    failure_reason: str,
    *,
    prompt_version: str = PROMPT_VERSION,
    extra: dict[str, Any] | None = None,
) -> Path:
    case_id = str(case.get("case_id") or "case").replace("/", "_")
    path = trace_dir / method / f"{case_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_id": case.get("case_id"),
        "source": case.get("source"),
        "method": method,
        "prompt_version": prompt_version,
        "messages": messages,
        "raw_response": raw_response,
        "parsed_output": parsed,
        "usage": normalize_usage(usage),
        "status": status,
        "failure_reason": failure_reason,
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def normalize_usage(usage: dict[str, Any]) -> dict[str, int]:
    if not isinstance(usage, dict):
        return {}
    keys = [
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
        "local_cache_hits",
        "local_cache_saved_tokens",
    ]
    return {key: int(usage.get(key, 0) or 0) for key in keys}


def make_key(case: dict[str, Any], method: str) -> str:
    return f"{case.get('case_id')}::{method}"


def row_key(row: dict[str, Any]) -> str:
    return f"{row.get('case_id')}::{row.get('method')}"


def should_retry(row: dict[str, Any], args: argparse.Namespace) -> bool:
    retry_statuses = {item.strip() for item in str(getattr(args, "retry_statuses", "") or "").split(",") if item.strip()}
    if retry_statuses and str(row.get("status") or "") in retry_statuses:
        return True
    return bool(args.retry_cache_misses and row.get("status") == "cache_miss_blocked")


def summarize(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    manifest_path: Path,
    rows_path: Path,
    skipped_existing: int,
    new_rows: int,
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    method_status_counts: dict[str, dict[str, int]] = {}
    method_reference_counts: dict[str, dict[str, Any]] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        method = str(row.get("method") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        method_status_counts.setdefault(method, {})
        method_status_counts[method][status] = method_status_counts[method].get(status, 0) + 1
        metrics = method_reference_counts.setdefault(
            method,
            {
                "numeric_total": 0,
                "numeric_matches": 0,
                "official_total": 0,
                "official_accuracy": 0,
                "compile_success": 0,
                "runtime_success": 0,
                "objective_exact": 0,
                "status_exact": 0,
                "solution_exact_total": 0,
                "solution_exact": 0,
                "compile_errors": 0,
                "runtime_errors": 0,
                "total_tokens": 0,
                "rows_with_tokens": 0,
                "total_latency_seconds": 0.0,
                "rows_with_latency": 0,
            },
        )
        evaluation = row.get("reference_evaluation") if isinstance(row.get("reference_evaluation"), dict) else {}
        if evaluation.get("status") == "numeric_compared":
            metrics["numeric_total"] += 1
            if evaluation.get("numeric_match") is True:
                metrics["numeric_matches"] += 1
        official = row.get("official_evaluation") if isinstance(row.get("official_evaluation"), dict) else {}
        if official:
            metrics["official_total"] += 1
            if official.get("official_accuracy") is True:
                metrics["official_accuracy"] += 1
            if official.get("compile_success") is True:
                metrics["compile_success"] += 1
            else:
                metrics["compile_errors"] += 1
            if official.get("runtime_success") is True:
                metrics["runtime_success"] += 1
            elif official.get("compile_success") is True:
                metrics["runtime_errors"] += 1
            if official.get("objective_exact") is True:
                metrics["objective_exact"] += 1
            if official.get("status_exact") is True:
                metrics["status_exact"] += 1
            if official.get("reference_solution_available") is True:
                metrics["solution_exact_total"] += 1
            if official.get("solution_exact") is True:
                metrics["solution_exact"] += 1
        token_usage = row.get("token_usage") if isinstance(row.get("token_usage"), dict) else {}
        if token_usage.get("total_tokens") is not None:
            metrics["total_tokens"] += int(token_usage.get("total_tokens") or 0)
            metrics["rows_with_tokens"] += 1
        if row.get("latency_seconds") is not None:
            metrics["total_latency_seconds"] += float(row.get("latency_seconds") or 0.0)
            metrics["rows_with_latency"] += 1
    for metrics in method_reference_counts.values():
        total = int(metrics.get("numeric_total") or 0)
        metrics["numeric_match_rate"] = (float(metrics.get("numeric_matches") or 0) / total) if total else None
        official_total = int(metrics.get("official_total") or 0)
        metrics["official_accuracy_rate"] = (float(metrics.get("official_accuracy") or 0) / official_total) if official_total else None
        metrics["compile_success_rate"] = (float(metrics.get("compile_success") or 0) / official_total) if official_total else None
        metrics["runtime_success_rate"] = (float(metrics.get("runtime_success") or 0) / official_total) if official_total else None
        metrics["objective_exact_rate"] = (float(metrics.get("objective_exact") or 0) / official_total) if official_total else None
        solution_total = int(metrics.get("solution_exact_total") or 0)
        metrics["solution_exact_rate"] = (float(metrics.get("solution_exact") or 0) / solution_total) if solution_total else None
        metrics["compile_error_rate"] = (float(metrics.get("compile_errors") or 0) / official_total) if official_total else None
        metrics["runtime_error_rate"] = (float(metrics.get("runtime_errors") or 0) / official_total) if official_total else None
        token_rows = int(metrics.get("rows_with_tokens") or 0)
        metrics["mean_tokens"] = (float(metrics.get("total_tokens") or 0) / token_rows) if token_rows else None
        latency_rows = int(metrics.get("rows_with_latency") or 0)
        metrics["mean_latency_seconds"] = (
            float(metrics.get("total_latency_seconds") or 0.0) / latency_rows
            if latency_rows
            else None
        )
    return {
        "schema_version": "liveopt_static_calibration_summary_v1",
        "manifest": str(manifest_path),
        "rows_path": str(rows_path),
        "model": args.model,
        "cache_only": bool(args.cache_only),
        "response_protocol": args.response_protocol,
        "code_repair_policy": "equal_public_code_output_repair_v1",
        "code_repair_attempts": int(getattr(args, "code_repair_attempts", 0) or 0),
        "official_repair_policy": "source_aligned_native_repair_capped_v1",
        "official_repair_attempts": int(getattr(args, "official_repair_attempts", 0) or 0),
        "case_selection": {
            "case_offset": int(args.case_offset or 0),
            "case_stride": int(args.case_stride or 1),
            "case_ids": args.case_ids,
            "limit": int(args.limit or 0),
        },
        "new_rows": new_rows,
        "skipped_existing": skipped_existing,
        "latest_rows": len(rows),
        "status_counts": status_counts,
        "method_status_counts": method_status_counts,
        "method_reference_counts": method_reference_counts,
    }


if __name__ == "__main__":
    raise SystemExit(main())
