# LLM-Guided DSL Generation and Repair

A comparative study of LLM-guided code generation and iterative compiler-feedback repair for a domain-specific language (LIRAs DSL), with support for Google Gemini (Vertex AI) and Groq models.

## Overview

This repository contains the experimental pipeline and analysis artifacts for a factorial study evaluating how **model choice**, **system prompt design**, **few-shot examples**, and **repair prompting strategies** affect the ability of large language models to generate compilable DSL code.

The pipeline implements a generate–compile–repair loop:

1. **Generate** — An LLM produces DSL code from a natural-language scenario description, guided by a system prompt and optional few-shot examples.
2. **Compile** — The generated code is validated locally by a Java-based LIRAs compiler.
3. **Repair** — If compilation fails, compiler output is fed into a dedicated repair chat session that iteratively fixes the code.
4. **Repeat** — Steps 2–3 repeat up to a configurable `max_iterations` limit.

### Experimental Design

The study uses a 2³ full-factorial design across 8 pipeline configurations:

| Config | Model                | Repair Prompt | Few-Shot |
| ------ | -------------------- | ------------- | -------- |
| C1     | Gemini 2.5 Flash     | SPR1          | No       |
| C2     | Gemini 2.5 Flash     | SPR1          | Yes      |
| C3     | Gemini 2.5 Flash     | SPR2          | No       |
| C4     | Gemini 2.5 Flash     | SPR2          | Yes      |
| C5     | Gemini 3 Pro Preview | SPR1          | No       |
| C6     | Gemini 3 Pro Preview | SPR1          | Yes      |
| C7     | Gemini 3 Pro Preview | SPR2          | No       |
| C8     | Gemini 3 Pro Preview | SPR2          | Yes      |

Each configuration is run against 4 scenarios × 5 generation prompts × 3 shot levels = 60 runs, totaling 480 runs across all configurations.

## Repository Structure

```
├── dsl_generator.py              # Main generation + repair pipeline
├── dsl_generator_flash.py        # Lightweight fork optimized for Gemini 2.5 Flash
├── config.json                   # Runtime configuration (single-run entry point)
├── requirements.txt              # Python dependencies
├── SPs/
│   ├── Generative/               # Generation-stage system prompts (SP1–SP5)
│   └── Repair/                   # Repair-stage system prompts (SPR1, SPR2)
├── Shots/
│   ├── Generative/               # Few-shot examples for generation
│   │   ├── UserScenario_1.txt / AssistantScenario_1.txt
│   │   └── UserScenario_2.txt / AssistantScenario_2.txt
│   └── Repair/                   # Few-shot examples for repair
│       └── UserScenario_3–5.txt / AssistantScenario_3–5.txt
├── Scenarios/                    # Natural-language scenario descriptions
│   ├── Scenario_011.txt
│   ├── Scenario_016.txt
│   ├── Scenario_029.txt
│   └── Scenario_06.txt
├── DSL/                          # Pre-generated DSL baselines (per scenario/prompt/shot)
├── Runs/                         # Raw run outputs organized by configuration (C1–C8)
│   └── C<n>/<Scenario>/<SP>/RUN_<timestamp>/
│       ├── dsl/                  # Generated .LIRAs files per iteration
│       ├── compiler/             # Compiler output per iteration
│       └── run_metadata.json     # Full run telemetry and iteration log
├── Report/
│   ├── configs.csv               # Configuration factor matrix
│   ├── Histories/                # Per-config run history CSVs (c1.csv–c8.csv)
│   ├── Tables/                   # Summary tables (CSV + rendered PNG images)
│   └── Figures/                  # Publication figures
└── Utils/
    ├── run_all_pairs.py          # Batch runner for all scenario/prompt combinations
    ├── collect_run_history.py    # Extract run metadata into analysis-ready CSVs
    ├── compile_run_histories.py  # Merge per-config CSVs into a combined dataset
    ├── render_figures.py         # Generate publication figures from combined data
    ├── render_tables.py          # Render CSV tables as formatted PNG images
    ├── export_run_tables.py      # Export summary tables from combined data
    ├── run_factorial_analysis.py # Statistical analysis (main effects, interactions)
    └── run_SC_failure.py         # Supplementary failure analysis
```

## Prerequisites

- **Python 3.9+**
- **Java runtime** on `PATH` (required to run the LIRAs compiler JAR)
- **Google Cloud** project with Vertex AI API enabled (if using `provider: "gemini"`)
- **Groq API key** (if using `provider: "groq"`)

## Setup

1. **Install Python dependencies:**

   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\Scripts\activate       # Windows
   pip install -r requirements.txt
   ```

2. **Configure credentials:**

   - Gemini: place a Vertex AI service account JSON key at `keys/key.json` (gitignored), or use ADC with `gcloud auth application-default login`.
   - Groq: export `GROQ_API_KEY` (or set `groq_api_key` in `config.json`).

3. **Verify Java is available:**

   ```bash
   java -version
   ```

## Usage

### Single Run

Edit `config.json` and run:

```bash
python dsl_generator.py
```

**`config.json` reference:**

| Key                       | Type        | Description                                                              |
| ------------------------- | ----------- | ------------------------------------------------------------------------ |
| `system_prompt`           | string      | Generation prompt from `SPs/Generative/` (e.g., `"Generative/SP3.txt"`)  |
| `provider`                | string      | Model backend: `"gemini"` or `"groq"`                                   |
| `groq_api_key`            | string      | Optional Groq key in config (recommended: env var instead)                |
| `generation_model`        | string      | Vertex AI model for generation (e.g., `"gemini-3-pro-preview"`)          |
| `repair_model`            | string      | Vertex AI model for repair iterations                                    |
| `shots`                   | int \| list | Few-shot example count (0, 1, 2) or explicit `[{user, assistant}]` pairs |
| `scenario`                | string      | Scenario file from `Scenarios/`                                          |
| `repair_prompt`           | string      | Repair system prompt from `SPs/Repair/` (e.g., `"Repair/SPR1.txt"`)      |
| `repair_shots`            | int \| list | Few-shot examples for the repair chat (default: 0)                       |
| `compiler_jar`            | string      | Path to the LIRAs compiler JAR                                           |
| `max_iterations`          | int         | Maximum generate→compile→repair attempts                                 |
| `generation_temperature`  | float       | Sampling temperature for generation (default: 1.0)                       |
| `repair_temperature`      | float       | Sampling temperature for repair (default: 0.2)                           |
| `generation_only`         | bool        | Skip compile/repair, only save generated DSL                             |
| `use_generated_dsl_cache` | bool        | Load DSL from cache instead of generating                                |
| `generated_dsl_source`    | string      | Cache source: `"generated_cache"` or `"dsl_folder"`                      |
| `results_dir`             | string      | Override output directory (e.g., `"Runs/C5"`)                            |

Example for Groq in `config.json`:

```json
{
   "provider": "groq",
   "generation_model": "llama-4-scout-17b-16e-instruct-maas",
   "repair_model": "llama-4-scout-17b-16e-instruct-maas"
}
```

### Batch Runs

Run all scenario × prompt × shot combinations:

```bash
python Utils/run_all_pairs.py
python Utils/run_all_pairs.py --shots 0,1,2
python Utils/run_all_pairs.py --generation-only
python Utils/run_all_pairs.py --disable-generation --shots 0,1,2
python Utils/run_all_pairs.py --list-only
```

Run all scenarios using the single prompt/model settings already defined in `config.json`:

```bash
python Utils/run_all_scenarios.py
python Utils/run_all_scenarios.py --list-only
python Utils/run_all_scenarios.py --shots 0,1,2
```

| Flag                   | Description                                          |
| ---------------------- | ---------------------------------------------------- |
| `--generation-only`    | Save generated DSL without compiling                 |
| `--disable-generation` | Load DSL from cache and run compile/repair only      |
| `--compiler-timeout`   | Override compiler timeout (seconds)                  |
| `--inter-run-delay`    | Pause between runs to reduce API rate-limit pressure |
| `--list-only`          | List planned runs without executing                  |

## Analysis Pipeline

### 1. Collect Run Histories

Extract structured CSVs from raw `run_metadata.json` files:

```bash
python Utils/collect_run_history.py --out Report/Histories/c1.csv
```

### 2. Combine Configurations

Merge all per-config CSVs into a single comparative dataset:

```bash
python Utils/compile_run_histories.py \
  --input-glob "Report/Histories/c*.csv" \
  --outcsv Report/Histories/combined_run_histories.csv
```

### 3. Generate Tables and Figures

```bash
python Utils/export_run_tables.py
python Utils/render_figures.py
python Utils/render_tables.py
python Utils/run_factorial_analysis.py
```

### Output Artifacts

**Tables** (`Report/Tables/`):

| File                                     | Description                                           |
| ---------------------------------------- | ----------------------------------------------------- |
| `table00_study_summary.csv`              | Overall study design and run counts                   |
| `table01_config_scorecard.csv`           | Per-config success rate, iteration stats, token usage |
| `table02_prompt_scenario_matrix.csv`     | Prompt × scenario success breakdown                   |
| `table03_time_to_success.csv`            | Iterations-to-first-success distribution              |
| `table04_parameter_effects.csv`          | Main-effect sizes per experimental factor             |
| `table05_failure_by_prompt_scenario.csv` | Failure analysis by prompt and scenario               |
| `table06_status_breakdown.csv`           | Run outcome status distribution                       |

**Figures** (`Report/Figures/`):

| Figure                              | Description                                        |
| ----------------------------------- | -------------------------------------------------- |
| `fig01_success_rate_ci.png`         | Success rate per configuration with 95% Wilson CI  |
| `fig02_main_effect_forest.png`      | Main-effect forest plot                            |
| `fig03_factor_interaction.png`      | Model × few-shot × repair prompt interaction       |
| `fig04_prompt_scenario_heatmap.png` | Prompt × scenario success-rate heatmap             |
| `fig05_iterations_box_strip.png`    | Iterations to success distribution                 |
| `fig06_error_convergence.png`       | Error-score convergence by configuration           |
| `fig07_error_flow.png`              | Error-category flow from generation to post-repair |
| `fig08_scenario_difficulty.png`     | Scenario difficulty profile                        |

## Architecture

### Generation Phase

The generation model receives a **system prompt** (one of SP1–SP5), optional **few-shot examples** (user/assistant scenario pairs injected as chat history), and the **target scenario** as a user message. The model produces raw DSL code, which is extracted by stripping markdown fences and preamble.

### Repair Phase

A **separate chat session** is created with a repair-specific system prompt (SPR1 or SPR2). Each repair turn sends the failed DSL plus compiler output as a single user message. The repair chat is **stateful** by default — prior repair attempts accumulate in the conversation context, with a sliding window of the last 3 attempts to prevent regression. Repair uses lower temperature (0.2) to favor deterministic fixes.

### Telemetry

Every run produces a `run_metadata.json` file capturing:

- Configuration parameters and model identifiers
- Per-iteration DSL paths, compiler output, error scores, and validation results
- Approximate token usage and timing
- Final status (`success`, `max_iterations_reached`, `setup_error`, `crashed`)
