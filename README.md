# SAILIR

**S**elf-supervised **AI** for **L**oop **I**ntegral **R**eduction.

Code, trained model, and benchmark data accompanying:

> **Learning to Unscramble Feynman Loop Integrals with SAILIR**
> David Shih, 2026. arXiv:2604.05034.

## Repository layout

```
sailir/
  ibp_env.py                 IBP environment (kinematics, identities, action enumeration)
  classifier.py              Action classifier
scripts/
  data_gen/                  Self-supervised trajectory generation + JSONL→tensor packing
  train/train_classifier.py  Cross-entropy training
  eval/                      Hierarchical async orchestrator + single-integral worker + replay
  kira/                      Kira inputs for the comparison side of Fig. 3
  plots/                     Fig. 2 and Fig. 3
checkpoints/best_model.pt    Published trained model
results/                     Benchmark CSVs and training log
```

## Requirements

- Python 3.10+
- PyTorch, NumPy, Matplotlib
- HTCondor (only for the original distributed triangle-box workflows; box1l uses local multiprocessing)
- Kira + Fermat (for §6 only)

```bash
pip install torch numpy matplotlib
```

The reducer and classifier also accept `--topology trianglebox` (the default)
or `--topology box1l`. The box topology has four indices, four IBP actions,
and three master integrals. Packed data must be prepared and trained with the
same topology as the records.

## Quick start

Reduce a single integral with the published checkpoint:

```bash
python scripts/eval/onestep_worker.py \
    --integral 2,1,2,1,2,2,-4 \
    --model-checkpoint checkpoints/best_model.pt \
    --output reduction.pkl \
    --beam_width 20 \
    --prime 1009 \
    --paper-masters-only -v

python scripts/eval/replay_reduction_path.py --path reduction.pkl --prime 1009
```

## End-to-end reproduction

The commands below retain the published triangle-box setup. For box1l, use
`--topology box1l` consistently for generation, packing, training, and
evaluation. The local generator is suitable for Colab and does not require
HTCondor:

```bash
python scripts/data_gen/run_local_datagen.py \
    --topology box1l --n_workers 4 --n_scrambles_per_worker 7500 \
    --seed 42 --prime 1009 --output data/box1l_training_data.jsonl

python scripts/data_gen/preprocess_to_tensors.py \
    --topology box1l --input data/box1l_training_data.jsonl \
    --output_dir data/box1l_tensors --seed 42

python scripts/train/train_classifier.py \
    --topology box1l --data_dir data/box1l_tensors \
    --output_dir checkpoints/box1l --prime 1009 --epochs 30 \
    --batch_size 256 --lr 4e-4 --device auto --resume auto

python scripts/eval/onestep_worker.py \
    --integral 1,-1,1,0 --topology box1l --box-masters-only \
    --model-checkpoint checkpoints/box1l/best_model.pt \
    --output reduction_box1l.pkl --beam_width 20 --prime 1009
```

The full Drive-backed Colab workflow, including tests, plots and hierarchical
Laporta comparisons, is in [notebooks/box1l_colab.ipynb](notebooks/box1l_colab.ipynb).

### 1. Generate training data

100 Condor workers × 1000 scrambles each (`--max_steps 25`). ≈18% of attempts
are discarded as `SKIPPED_VANISHING`; the remaining ≈82 000 trajectories yield
the $8\times10^4$ trajectories / $1.06\times10^6$ samples reported in paper §IV.B.

```bash
export SAILIR_DIR=$(pwd)
export PYTHON=$(which python)

bash scripts/data_gen/submit_datagen.sh 100 1000
condor_submit scripts/data_gen/datagen_job_custom.jdl
# ... wait for all 100 jobs to finish ...
bash scripts/data_gen/merge_outputs.sh data/raw_jsonl/
```

### 2. Pack JSONL into tensors

```bash
python scripts/data_gen/preprocess_to_tensors.py \
    --input      data/raw_jsonl/multisector_training_data.jsonl \
    --output_dir data/multisector/
```

Defaults reproduce the paper's preprocessing: `--val_split 0.1`, `--test_split 0.1`, `--seed 42`. Produces `train.pt` (≈80%), `val.pt` (≈10%), `test.pt` (≈10%).

### 3. Train

```bash
python scripts/train/train_classifier.py \
    --data_dir data/multisector/ --output_dir checkpoints/ \
    --epochs 30 --batch_size 256 --lr 4e-4 --prime 1009 --device cuda

python scripts/plots/plot_training_curve.py --log results/training_log/train.log
```

### 4. Reduce the 16 benchmark integrals

```bash
INTEGRALS=( "2,1,2,1,2,2,-4"  "1,1,2,2,1,3,-5"  "1,1,3,2,2,1,-6"  "2,3,1,1,2,1,-7"
            "2,2,2,1,1,3,-4"  "1,1,2,3,2,2,-5"  "1,4,2,1,2,1,-6"  "2,1,1,2,3,2,-7"
            "2,3,1,3,1,2,-4"  "1,2,2,2,1,4,-5"  "3,2,3,2,1,1,-6"  "3,1,1,1,1,5,-7"
            "2,3,3,3,1,1,-4"  "2,2,3,3,2,1,-5"  "3,2,1,3,2,2,-6"  "2,2,3,3,1,2,-7" )

mkdir -p logs results work-dir
for I in "${INTEGRALS[@]}"; do
    label=$(echo "$I" | tr ',' '_' | tr '-' 'm')
    python -u scripts/eval/hierarchical_reduction.py \
        --integral $I --output results/reduction_${label}.pkl \
        --work-dir work-dir/${label} \
        --model-checkpoint checkpoints/best_model.pt \
        --beam_width 20 --prime 1009 --paper-masters-only --beam-sort mixed \
        > logs/async_${label}.log 2>&1 &
done
wait
```

### 5. Plot

```bash
python scripts/eval/collect_benchmark_results.py \
    --logdir logs/ --resultdir results/ --scratchdir work-dir/

python scripts/plots/plot_benchmark_comparison.py \
    --sailir-csv results/benchmark_mixed_summary_v13.csv \
    --kira-csv   results/kira_benchmark.csv \
    --out        results/benchmark_comparison.pdf
```

`results/sailir_benchmark.csv` is the CSV used in the paper; pass it to
`--sailir-csv` to regenerate Fig. 3 verbatim.

### 6. Kira benchmark (optional)

```bash
export KIRA=/path/to/kira/bin/kira
export FERMATPATH=/path/to/fermat/fer64
cd scripts/kira/ && ./run_benchmark_s7.sh
```

## Citation

```bibtex
@article{Shih:2026jfe,
    author = "Shih, David",
    title = "{Learning to Unscramble Feynman Loop Integrals with SAILIR}",
    eprint = "2604.05034",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "4",
    year = "2026"
}
```

## License

MIT.
