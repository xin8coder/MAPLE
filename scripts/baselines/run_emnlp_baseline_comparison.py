#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


MAIN_MODES = {"evo2", "aevo", "stagewise_full_restart", "react", "react_tools", "optimai"}


def run_cmd(cmd, env=None):
    print('RUN', ' '.join(cmd), flush=True)
    p = subprocess.run(cmd, text=True, capture_output=True, env=env)
    print(p.stdout)
    if p.returncode != 0:
        print(p.stderr, file=sys.stderr)
    return p.returncode


def summarize_file(mode: str, path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    updates = [u for r in rows for u in r.get('update_results', [])]
    llm_trace_tokens = sum(int((u.get('llm_traces') or {}).get('total_tokens') or 0) for u in updates)
    token_usage_rows = [r.get('token_usage') or {} for r in rows]
    local_cache_hits = sum(int(t.get('local_cache_hits') or 0) for t in token_usage_rows)
    local_cache_saved_tokens = sum(int(t.get('local_cache_saved_tokens') or 0) for t in token_usage_rows)
    provider_cache_hit_tokens = sum(int(t.get('provider_prompt_cache_hit_tokens') or 0) for t in token_usage_rows)
    provider_cache_miss_tokens = sum(int(t.get('provider_prompt_cache_miss_tokens') or 0) for t in token_usage_rows)
    stage_traces = [s for r in rows for s in r.get('stage_traces', [])]
    hidden_scores = [
        (s.get('hidden_evaluation') or {}).get('normalized_score')
        for s in stage_traces
        if (s.get('hidden_evaluation') or {}).get('normalized_score') is not None
    ]
    hidden_hv = [
        (s.get('hidden_evaluation') or {}).get('normalized_hv')
        for s in stage_traces
        if (s.get('hidden_evaluation') or {}).get('normalized_hv') is not None
    ]
    hidden_igd = [
        (s.get('hidden_evaluation') or {}).get('igd')
        for s in stage_traces
        if (s.get('hidden_evaluation') or {}).get('igd') is not None
    ]
    return {
        'mode': mode,
        'episodes': len(rows),
        'updates': sum(r['episode_metrics'].get('update_count', 0) for r in rows),
        'all_updates_feasible': sum(r['episode_metrics']['all_updates_feasible'] for r in rows),
        'success_rate': sum(r['episode_metrics']['all_updates_feasible'] for r in rows) / len(rows) if rows else 0,
        'hidden_true_update_pass_ratio': _mean([
            r['episode_metrics'].get('hidden_true_update_pass_ratio')
            for r in rows
            if r['episode_metrics'].get('hidden_true_update_pass_ratio') is not None
        ]),
        'hidden_normalized_score_mean': _mean(hidden_scores),
        'hidden_hv_ratio_mean': _mean(hidden_hv),
        'hidden_igd_mean': _mean(hidden_igd),
        'cum_tokens': sum(r['episode_metrics']['cumulative_token_cost'] for r in rows),
        'llm_trace_tokens': llm_trace_tokens,
        'local_cache_hits': local_cache_hits,
        'local_cache_saved_tokens': local_cache_saved_tokens,
        'provider_prompt_cache_hit_tokens': provider_cache_hit_tokens,
        'provider_prompt_cache_miss_tokens': provider_cache_miss_tokens,
        'avg_tokens': sum(r['episode_metrics']['cumulative_token_cost'] for r in rows) / len(rows) if rows else 0,
        'cum_disruption': sum(r['episode_metrics']['cumulative_disruption'] for r in rows),
        'avg_final_objective': sum(r['episode_metrics']['final_objective'] for r in rows) / len(rows) if rows else 0,
        'avg_memory_reuse': sum(r['episode_metrics']['memory_reuse_ratio'] for r in rows) / len(rows) if rows else 0,
        'avg_latency': sum(r['episode_metrics']['cumulative_latency_seconds'] for r in rows) / len(rows) if rows else 0,
        'output': str(path),
    }


def _mean(values):
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', default='data/evo2_dynoptbench/episodes/fjsp_episodes.jsonl')
    ap.add_argument('--limit', type=int, default=1)
    ap.add_argument('--max-updates', type=int, default=10, help='maximum dynamic updates per episode; -1 means all updates; 0 means initial stage only')
    ap.add_argument('--model', default='deepseek-v4-pro')
    ap.add_argument('--out-dir', default='logs/llm_tests/emnlp_baselines')
    ap.add_argument('--nldo-source-id', default='', help='Optional NLDO source/profile id such as S01.')
    ap.add_argument('--episode-id-prefix', default='', help='Optional episode-id prefix such as NLDO-P010.')
    ap.add_argument('--checkpoint-dir', default='', help='Shared checkpoint root; defaults to <out-dir>/checkpoints.')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--use-response-cache', action='store_true', help='Force DEEPSEEK_CACHE=1 for every subprocess.')
    ap.add_argument('--cache-only', action='store_true', help='Set DEEPSEEK_CACHE_ONLY=1 so uncached LLM requests fail before provider calls.')
    ap.add_argument('--cache-dir', default='', help='Override DEEPSEEK_CACHE_DIR.')
    ap.add_argument('--trace-dir', default='', help='Write prompt/response JSONL traces under this directory.')
    ap.add_argument('--population-size', type=int, default=0)
    ap.add_argument('--generations', type=int, default=0)
    ap.add_argument('--optimai-num-plans', type=int, default=0)
    ap.add_argument('--optimai-debug-rounds', type=int, default=-1)
    ap.add_argument('--artifact-debug-rounds', type=int, default=0)
    ap.add_argument('--ga-stability-runs', type=int, default=0)
    ap.add_argument(
        '--modes',
        default='',
        help='Optional comma-separated override, e.g. evo2,stagewise_full_restart,react_tools,optimai',
    )
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    if args.modes.strip():
        modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    else:
        modes = ['evo2', 'stagewise_full_restart', 'react_tools', 'optimai']
    unknown = [m for m in modes if m not in MAIN_MODES]
    if unknown:
        raise SystemExit(f"Unknown or removed modes: {unknown}. Allowed modes: {sorted(MAIN_MODES)}")
    summaries = []
    for mode in modes:
        output = out / f'{mode}_limit{args.limit}.jsonl'
        checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else out / 'checkpoints'
        cmd = [sys.executable, '-m', 'evo2.experiments.run_episode_suite', '--episodes', args.episodes, '--limit', str(args.limit), '--agent-mode', mode, '--model', args.model, '--output', str(output), '--memory-root', str(out / f'memory_{mode}'), '--checkpoint-dir', str(checkpoint_dir / mode)]
        cmd.extend(['--max-updates', str(args.max_updates)])
        if args.nldo_source_id:
            cmd.extend(['--nldo-source-id', args.nldo_source_id])
        if args.episode_id_prefix:
            cmd.extend(['--episode-id-prefix', args.episode_id_prefix])
        if args.resume:
            cmd.append('--resume')
        if args.population_size > 0:
            cmd.extend(['--population-size', str(args.population_size)])
        if args.generations > 0:
            cmd.extend(['--generations', str(args.generations)])
        if args.optimai_num_plans > 0:
            cmd.extend(['--optimai-num-plans', str(args.optimai_num_plans)])
        if args.optimai_debug_rounds >= 0:
            cmd.extend(['--optimai-debug-rounds', str(args.optimai_debug_rounds)])
        if args.artifact_debug_rounds > 0:
            cmd.extend(['--artifact-debug-rounds', str(args.artifact_debug_rounds)])
        if args.ga_stability_runs > 0:
            cmd.extend(['--ga-stability-runs', str(args.ga_stability_runs)])
        env = os.environ.copy()
        if args.use_response_cache or args.cache_only:
            env['DEEPSEEK_CACHE'] = '1'
        if args.cache_only:
            env['DEEPSEEK_CACHE_ONLY'] = '1'
        if args.cache_dir:
            env['DEEPSEEK_CACHE_DIR'] = args.cache_dir
        if args.trace_dir:
            env['DEEPSEEK_TRACE_DIR'] = str(Path(args.trace_dir) / mode)
        code = run_cmd(cmd, env=env)
        if code == 0 and output.exists():
            summaries.append(summarize_file(mode, output))
        else:
            summaries.append({'mode': mode, 'error': f'exit={code}', 'output': str(output)})
    summary_path = out / f'summary_limit{args.limit}.json'
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'summary': str(summary_path), 'results': summaries}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
