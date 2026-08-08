# LIRAs LLM-Guided Repair Pipeline

This repository runs an end-to-end pipeline for generating and validating LIRAs models with LLMs.

Pipeline overview:

![LIRAs pipeline overview](Img/PipelineImage.svg)

Retry behavior:

- compiler errors trigger an internal LLM repair loop;
- UPPAAL failures trigger a new feedback cycle, up to `max_uppaal_feedback_cycles`.

## 1. Setup

Create and activate a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Make sure Java is installed, because the pipeline uses:

```text
liras_compiler.jar
liras-cli.jar
liras-cli-layout2.jar
```

If UPPAAL validation is enabled, also make sure `verifyta` exists and that `verifyta_bin` in `config.json` points to the correct executable.

## 2. Set The API Key

If you use Hugging Face, export your token before running the pipeline:

```bash
export HF_TOKEN="your_huggingface_token"
```

The current `config.json` uses Hugging Face providers, so this is the command you normally need.

## 3. Configure `config.json`

The pipeline reads all settings from `config.json`.

Run command:

```bash
python3 pipeline_runner.py --config config.json
```

### Main Model Settings

| Field | Meaning |
| --- | --- |
| `generation_provider` | Provider used to generate the first LIRAs model. |
| `repair_provider` | Provider used to repair invalid LIRAs code. |
| `query_provider` | Provider used to adapt queries to the generated XML model. |
| `generation_model` | Model used for the first LIRAs generation. |
| `repair_model` | Model used during repair. |
| `query_model` | Model used for query adaptation. |
| `llm_seed` | Seed used for reproducibility, when supported by the provider. |

Supported providers are:

```text
gemini
groq
mistral
openrouter
huggingface
```

### Generation Settings

| Field | Meaning |
| --- | --- |
| `system_prompt` | Generation prompt template under `SPs/`. Example: `Generative/NewSP7.j2`. |
| `scenario` | Natural-language scenario under `Scenarios/`. Example: `NL_Specification_1.txt`. |
| `generation_temperature` | Sampling temperature for generation. |
| `generation_max_output_tokens` | Maximum output tokens for generation. |
| `shots` | Few-shot examples for generation. Use `0` for none, an integer/list for legacy shots, or `leave_one_spec_out` for NL_Specification_1/2/3 without data leakage. |
| `generation_only` | If `true`, only generate LIRAs code and stop. |
| `use_generated_dsl_cache` | If `true`, load existing LIRAs code instead of generating it. |
| `generated_dsl_root` | Root folder for generated DSL cache. Usually `GeneratedDSL`. |
| `dsl_source_root` | Root folder for manual/baseline DSL files. Usually `DSL`. |

For `shots: "leave_one_spec_out"`, examples are read from `Shots/Generative/NL_Specifications/Spec*/`.

### Repair And Compilation Settings

| Field | Meaning |
| --- | --- |
| `compiler_jar` | LIRAs compiler JAR. Usually `liras_compiler.jar`. |
| `compiler_timeout` | Compiler timeout in seconds. |
| `max_iterations` | Maximum generation/repair attempts inside one feedback cycle. |
| `repair_prompt` | Repair prompt template under `SPs/`. Example: `Repair/NewSPR7.j2`. |
| `repair_shots` | Few-shot examples for repair. Use `0` for none, `2` for both leave-one-spec-out repair cases, or `major_errors` for the same two-case mode. Lists still point to explicit shot files. |
| `repair_temperature` | Sampling temperature for repair. |
| `repair_max_output_tokens` | Maximum output tokens for repair. |
| `repair_stateless` | If `true`, each repair call is stateless. |

For `repair_shots: 2`, examples are read from `Shots/Repair/LeaveOneSpecOut/TargetSpec*/`; each target folder contains only real broken-to-corrected cases from other specifications.

### XML, Query, And UPPAAL Settings

| Field | Meaning |
| --- | --- |
| `enable_xml_export` | If `true`, export the valid LIRAs model to XML. |
| `lira_cli_jar` | CLI JAR used for XML export. Usually `liras-cli.jar`. |
| `lira_cli_timeout` | XML export timeout in seconds. |
| `enable_query_adaptation` | If `true`, adapt source queries to the generated XML model. |
| `query_system_prompt` | System prompt for query adaptation. |
| `query_user_prompt` | User prompt template for query adaptation. |
| `query_source_root` | Folder containing source queries. Usually `Queries`. |
| `enable_uppaal` | If `true`, run UPPAAL/verifyta validation. |
| `verifyta_bin` | Path to the `verifyta` executable. |
| `verifyta_timeout` | verifyta timeout in seconds. |
| `verifyta_probability_delta_threshold` | Maximum allowed probability delta. Example: `0.05`. |
| `max_uppaal_feedback_cycles` | Maximum number of full UPPAAL feedback cycles. |

Important difference:

```text
max_iterations = repair attempts inside one cycle
max_uppaal_feedback_cycles = full pipeline retries after UPPAAL feedback
```

### Output Setting

| Field | Meaning |
| --- | --- |
| `results_dir` | Base output folder for runs. Example: `Runs/Qwen3.5-9B`. |

## 4. Run The Pipeline

After editing `config.json`, run:

```bash
python3 pipeline_runner.py --config config.json
```

A new run folder is created under:

```text
<results_dir>/<scenario>/<system_prompt>/RUN_<timestamp>/
```

Example:

```text
Runs/Qwen3.5-9B/NL_Specification_1/Generative/NewSP7.j2/RUN_20260706_212813/
```

## 5. Understand The Run Output

A run contains one or more feedback cycles:

```text
RUN_20260706_212813/
├── run_metadata.json
├── ciclo1/
│   ├── run_metadata.json
│   ├── dsl/
│   ├── compiler/
│   ├── xml/
│   ├── queries/
│   ├── uppaal/
│   ├── llm_prompts.jsonl
│   ├── llm_responses.jsonl
│   └── hf_debug_responses.jsonl
└── ciclo2/
```

Useful files:

| File | Meaning |
| --- | --- |
| `run_metadata.json` | Global run result and cycle summary. |
| `cicloN/run_metadata.json` | Metadata for one cycle. |
| `dsl/SUCCESS_*.LIRAs` | Valid LIRAs file accepted by the compiler. |
| `compiler/*.compiler.txt` | Compiler output. |
| `xml/*.xml` | XML exported from the LIRAs model. |
| `queries/adapted.q` | Adapted UPPAAL queries. |
| `uppaal/*` | verifyta stdout/stderr files. |
| `hf_debug_responses.jsonl` | Hugging Face response/debug/token information. |

## 6. Generate The HTML Dashboard

Standard dashboard:

```bash
python3 Utils/build_model_runs_analysis.py \
  --runs-dir Runs \
  --output Report/model_runs_analysis.html \
  --summary
```

Dashboard with the LIRAs code shown when clicking a run:

```bash
python3 Utils/build_model_runs_analysis.py \
  --runs-dir Runs \
  --output Report/model_runs_analysis_with_liras.html \
  --include-liras-code
```

Open the generated HTML file in a browser:

```text
Report/model_runs_analysis.html
```

The dashboard reads directly from `Runs/`. No intermediate CSV files are required.

## Project Structure

```text
Scenarios/       # natural-language scenario files
SPs/             # Jinja prompt templates
SPs/Generative/  # generation prompts
SPs/Repair/      # repair prompts
SPs/Queries/     # query adaptation prompts
SPs/_partials/   # reusable prompt fragments
Queries/         # source queries
Shots/           # few-shot examples
Runs/            # all generated experimental runs
Report/          # generated dashboards and assets
GeneratedDSL/    # generated DSL cache
DSL/             # manual/baseline DSL files
Utils/           # dashboard and rerun utilities
```

Core files:

```text
pipeline_runner.py
dsl_generator.py
query_adapter.py
config.json
Utils/build_model_runs_analysis.py
Utils/rerun_same_run.py
```

## Common Issues

### Hugging Face key missing

Run:

```bash
export HF_TOKEN="your_huggingface_token"
```

### `verifyta` not found

Check this field in `config.json`:

```json
"verifyta_bin": "/path/to/verifyta"
```

### Java or JAR error

Check that Java works and that these files exist:

```text
liras_compiler.jar
liras-cli.jar
liras-cli-layout2.jar
```

### Dashboard has missing charts

Regenerate the dashboard. Chart assets are written under:

```text
Report/model_runs_analysis_assets/
```

## Recommended Workflow

1. Edit `config.json`.
2. Export the Hugging Face token if needed:

```bash
export HF_TOKEN="your_huggingface_token"
```

3. Run the pipeline:

```bash
python3 pipeline_runner.py --config config.json
```

4. Regenerate the dashboard:

```bash
python3 Utils/build_model_runs_analysis.py --runs-dir Runs --output Report/model_runs_analysis.html --summary
```

5. Open:

```text
Report/model_runs_analysis.html
```
