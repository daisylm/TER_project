# llm-linux-perf

`llm-linux-perf` is a Python framework for measuring Linux behavior during
local LLM inference. It runs a controlled TinyLlama baseline through Ollama,
collects timing and system metrics, and writes analysis-ready tables, reports,
and optional flamegraph artifacts.

The project is organized for experimentation: the Python package keeps the
measurement logic readable, notebooks explore the generated data, and the
`results/` folder preserves complete experiment outputs.

## What This Project Measures

The current baseline focuses on one measured prompt, `baseline_01`, with a
separate warmup prompt, `baseline_02`. During each run the framework can collect:

- Streaming response timings such as time to first token, generation time, total
  time, token rate, and per-chunk latency.
- Ollama timing fields such as load duration, prompt evaluation duration, and
  generation duration.
- Linux `perf stat` counters for the benchmark client, Ollama parent process,
  and Ollama child worker processes.
- Derived CPU, cache, TLB, scheduler, page fault, and per-token metrics.
- RSS memory and CPU frequency samples while inference is running.
- Optional scheduler traces and flamegraphs for selected runs.

## Repository Structure

```text
llm-linux-perf/
|-- README.md
|-- config.yaml
|-- requirements.txt
|-- FlameGraph/
|-- datasets/
|   `-- 12_prompt_dataset.json
|-- docs/
|   `-- MASTER_PLAN.MD
|-- llm_perf/
|   |-- __init__.py
|   |-- __main__.py
|   |-- analysis.py
|   |-- cli.py
|   |-- collectors.py
|   |-- config.py
|   |-- experiment.py
|   |-- metrics.py
|   |-- ollama.py
|   |-- prompts.py
|   |-- results.py
|   `-- schemas.py
|-- notebooks/
|   |-- 02_baseline_results_analysis.ipynb
|   `-- 03_run_index_cohort_analysis.ipynb
|-- reference/
|   `-- Baseline_benchmark.py
|-- results/
|   `-- baseline_01_YYYYMMDD_HHMMSS/
`-- tests/
    |-- test_analysis.py
    |-- test_cli.py
    |-- test_experiment.py
    |-- test_metrics.py
    |-- test_ollama.py
    |-- test_perf_parser.py
    |-- test_prompts.py
    `-- test_results.py
```

Local-only folders such as `.venv/` and `__pycache__/` may exist while working
in the project, but they are not part of the source layout.

## Folder Guide

### `llm_perf/`

The main Python package. It contains the code that loads prompts, manages
Ollama, starts collectors, computes metrics, writes output files, and exposes
the terminal CLI.

- `config.py` defines the experiment configuration and result-path helpers.
- `schemas.py` defines shared dataclasses for prompts, timings, memory samples,
  perf targets, scheduler summaries, diagnoses, and run results.
- `prompts.py` loads and normalizes the JSON prompt dataset.
- `ollama.py` contains the Ollama server/process helpers and streaming HTTP
  client.
- `collectors.py` wraps Linux `perf`, memory sampling, scheduler tracing, cache
  topology collection, and flamegraph capture.
- `metrics.py` parses perf output and computes derived metrics such as IPC,
  cache miss rates, blocked time, CPU parallelism, and stability summaries.
- `experiment.py` orchestrates warmups, measured sequences, collectors, and
  final summaries.
- `results.py` creates experiment folders and incrementally writes CSV, JSON,
  Markdown report, and artifact-manifest outputs.
- `analysis.py` provides notebook-facing helpers for loading result tables,
  creating summary tables, and plotting experiment behavior.
- `cli.py` implements the `python -m llm_perf` command.
- `__main__.py` makes the package executable from the terminal.

### `datasets/`

Contains the prompt corpus used by the experiments. The current file,
`12_prompt_dataset.json`, includes 12 prompts grouped into baseline, RAM-stress,
and CPU-stress categories. The default config uses `baseline_02` for warmup and
`baseline_01` for measured runs.

### `notebooks/`

Contains exploratory analysis notebooks for completed experiments.

- `02_baseline_results_analysis.ipynb` loads an experiment folder and builds the
  main A-J analysis views: run structure, latency, throughput, streaming,
  CPU/process ownership, cache/TLB behavior, scheduling, TMA, memory, and
  frequency plots.
- `03_run_index_cohort_analysis.ipynb` compares the same run index across
  sequences to study cold-run and steady-state behavior.

### `results/`

Contains generated experiment outputs. Each run creates a timestamped folder
named like `baseline_01_20260630_110426`.

```text
results/baseline_01_YYYYMMDD_HHMMSS/
|-- config.yaml
|-- experiment_metadata.json
|-- warmups/
|   `-- sequence_NN_warmup_result.json
|-- tables/
|   |-- run_metrics.csv
|   |-- target_perf_metrics.csv
|   |-- stream_events.csv
|   |-- memory_samples.csv
|   |-- sequence_summary.csv
|   |-- metric_boxplot_summary.csv
|   |-- metric_boxplot_values.csv
|   `-- artifacts_manifest.csv
|-- reports/
|   |-- baseline_bottleneck_report.md
|   |-- sequence_comparison_report.md
|   |-- summary.json
|   `-- figures/
`-- artifacts/
    `-- sequence_NN/
        `-- run_NN/
            |-- *.perf.data
            |-- *.perf.script.txt
            |-- *.perf.folded
            `-- *_flamegraph.svg
```

The `tables/` files are the main machine-readable outputs. The `reports/` files
are human-readable summaries and generated plots. The `artifacts/` tree stores
heavier raw profiling files only for runs that were configured to keep them.

### `FlameGraph/`

Vendored FlameGraph tooling used when `--flamegraph-runs` is enabled. The
experiment code expects scripts such as `stackcollapse-perf.pl` and
`flamegraph.pl` to be available here.

### `tests/`

Unit tests for the parser, metrics, prompt loading, CLI behavior, experiment
orchestration, Ollama helpers, analysis helpers, and result writer. They are
intended to protect the refactor from silently changing the benchmark outputs.

### `reference/`

Contains `Baseline_benchmark.py`, the frozen original benchmark implementation.
It is kept as a validation reference while the project is refactored into the
smaller `llm_perf/` package.

### `docs/`

Contains project planning and design notes. `MASTER_PLAN.MD` explains the
refactor goals, class boundaries, and measurement philosophy.

## Important Top-Level Files

- `config.yaml` is the default experiment configuration. It sets the model,
  Ollama server URL, dataset path, warmup and measured prompt IDs, sequence
  count, runs per sequence, token controls, memory sampling interval, output
  directory, and optional profiling settings.
- `requirements.txt` lists the Python dependencies used by the package and
  notebooks.
- `README.md` is this external guide.

## Quick Start

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Make sure Ollama and the default model are available:

```bash
ollama pull tinyllama
```

Check the resolved experiment configuration without running a benchmark:

```bash
python -m llm_perf --config config.yaml --dry-run
```

Run the default experiment:

```bash
python -m llm_perf --config config.yaml --no-stop-ollama-after
```

For Linux systems where normal `perf` access is denied, authenticate first and
then use `--sudo-perf`:

```bash
sudo -v
python -m llm_perf --config config.yaml --sudo-perf --no-stop-ollama-after
```

Run from a host-native terminal where the Ollama process ID is visible. If you
request flamegraphs, keep the `FlameGraph/` directory available at the path set
by `flamegraph_dir`.

## Useful Commands

Run a smaller smoke experiment:

```bash
python -m llm_perf --config config.yaml --sequences 1 --runs-per-sequence 1 --no-stop-ollama-after
```

Capture flamegraphs for selected runs:

```bash
python -m llm_perf --config config.yaml --flamegraph-runs 1,8 --no-stop-ollama-after
```

Capture scheduler traces for selected global run indices:

```bash
python -m llm_perf --config config.yaml --scheduler-runs 1,8 --no-stop-ollama-after
```

Run the unit test suite:

```bash
python -m unittest discover -s tests
```

## Typical Workflow

1. Edit `config.yaml` or pass CLI overrides.
2. Run `python -m llm_perf --config config.yaml`.
3. Open the newest folder in `results/`.
4. Inspect `tables/run_metrics.csv` and `tables/target_perf_metrics.csv`.
5. Use the notebooks in `notebooks/` for plots and deeper interpretation.
6. Compare new outputs against the reference behavior when changing benchmark
   logic.

## Project Status

The repository has moved beyond the original monolithic benchmark into a shallow
Python package with a CLI, result writer, analysis helpers, notebooks, and unit
tests. The current experiment remains centered on a controlled single-prompt
baseline, with sequence-based repetitions for studying cold-run and steady-state
Linux behavior during local LLM inference.
