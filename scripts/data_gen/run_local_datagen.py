#!/usr/bin/env python3
"""Deterministic local multiprocessing JSONL trajectory generator."""
import argparse
import concurrent.futures
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).with_name('generate_multisector_data.py')

def run_worker(task):
    idx, n, seed, id_offset, output, topology, prime = task
    cmd = [sys.executable, str(WORKER), '--topology', topology,
           '--n_scrambles', str(n), '--start_seed', str(seed), '--id_offset', str(id_offset),
           '--min_steps', '5', '--max_steps', '20', '--prime', str(prime),
           '--output', str(output)]
    started = time.monotonic()
    print(f'[datagen worker {idx+1}] generating {n} trajectories (seed={seed})', flush=True)
    subprocess.run(cmd, cwd=REPO, check=True, stdout=subprocess.DEVNULL)
    print(f'[datagen worker {idx+1}] complete in {time.monotonic()-started:.1f}s', flush=True)
    return idx, Path(output)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--topology', choices=['trianglebox','box1l'], default='box1l')
    ap.add_argument('--n_workers', type=int, default=4)
    ap.add_argument('--n_scrambles_per_worker', type=int, default=7500)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--prime', type=int, default=1009)
    ap.add_argument('--output', default='data/box1l_training_data.jsonl')
    args = ap.parse_args()
    if args.n_workers < 1 or args.n_scrambles_per_worker < 1:
        ap.error('worker and scramble counts must be positive')
    seeds = random.Random(args.seed).sample(range(1, 2**31), args.n_workers)
    out = (REPO / args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    # Place worker JSONLs beside the requested output (often on mounted Drive),
    # so in-progress parts are durable if a hosted runtime is interrupted.
    with tempfile.TemporaryDirectory(prefix='.sailir_datagen_', dir=out.parent) as td:
        tasks = [(i, args.n_scrambles_per_worker, seeds[i], i*args.n_scrambles_per_worker,
                  Path(td)/f'worker_{i:04d}.jsonl', args.topology, args.prime)
                 for i in range(args.n_workers)]
        started = time.monotonic()
        completed = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            futures = [pool.submit(run_worker, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                completed.append(future.result())
                done = len(completed)
                elapsed = max(time.monotonic()-started, 1e-6)
                remaining = elapsed * (args.n_workers-done) / args.n_workers
                print(f'[datagen] {done}/{args.n_workers} workers complete; estimated remaining {remaining/60:.1f} min', flush=True)
        results = sorted(completed, key=lambda pair: pair[0])
        with out.open('w') as merged:
            for _, path in results:
                with path.open() as part:
                    for line in part: merged.write(line)
    print(f'Wrote merged JSONL: {out} ({args.n_workers * args.n_scrambles_per_worker} attempted trajectories)', flush=True)

if __name__ == '__main__': main()
