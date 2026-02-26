# RESEARCH_PROJECT

DSL Generator with Gemini API via Vertex AI - Automates the generation and refinement of domain-specific language code.

This project runs an automated feedback loop:

- Generate DSL with Vertex AI (Gemini)
- Validate locally with a Java-based CLI compiler (JAR)
- If compilation fails, send compiler output back to the model for repair (in a dedicated repair chat)
- Repeat up to `max_iterations`

## Setup Instructions

### 0. Local Prerequisites (Java)

You must have a Java runtime installed and accessible as `java` on your `PATH` so the script can run the compiler JAR:

```bash
java -version
```

Install options:

- macOS (Homebrew):
  - `brew install openjdk`
  - Ensure `java` is on PATH (Homebrew prints the exact `PATH`/symlink instructions after install)
- Ubuntu/Debian:
  - `sudo apt-get update && sudo apt-get install -y default-jre`
- Windows:
  - Install a JDK (e.g., Temurin/OpenJDK) and ensure `java.exe` is available in PATH

### 1. Google Cloud Project Setup

1. **Create a Google Cloud Project** (if you don't have one):
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Click "Select a project" → "New Project"
   - Enter a project name and click "Create"
   - Note your Project ID (e.g., `aerobic-stream-483313-v6`)

2. **Enable Vertex AI API**:
   - In your Google Cloud Console, go to "APIs & Services" → "Library"
   - Search for "Vertex AI API"
   - Click on it and press "Enable"

### 2. Create Service Account and Key

1. **Create a Service Account**:
   - Go to "IAM & Admin" → "Service Accounts"
   - Click "Create Service Account"
   - Enter a name (e.g., `research-project`)
   - Click "Create and Continue"

2. **Grant Permissions**:
   - Add the role: "Vertex AI User"
   - Click "Continue" → "Done"

3. **Generate JSON Key**:
   - Click on your newly created service account
   - Go to "Keys" tab
   - Click "Add Key" → "Create new key"
   - Select "JSON" format
   - Click "Create" - the key file will download automatically

4. **Save the Key**:
   - Rename the downloaded file to `key.json`
   - **For shared projects**: Create a `keys/` directory and place it there: `RESEARCH_PROJECT/keys/key.json`
   - **For personal use**: You can also place it in the project root: `RESEARCH_PROJECT/key.json`
   - **Important**: Both `keys/` directory and `key.json` are in `.gitignore` and won't be committed to version control

### 3. Create Python Virtual Environment

1. **Create virtual environment**:

   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment**:
   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. **Verify activation**:
   - Your terminal prompt should now show `(venv)` at the beginning

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Generator

```bash
python dsl_generator.py
```

The script will automatically detect and use the `key.json` file from either:

1. `keys/key.json` (recommended for shared projects)
2. `key.json` (root directory, for backward compatibility)

## Alternative: Use gcloud CLI Authentication

Instead of a service account key, you can use gcloud CLI:

1. Install [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
2. Run: `gcloud auth application-default login`
3. Remove or rename `key.json` from the `keys/` directory
4. The script will use your gcloud credentials instead

## Project Structure

```
RESEARCH_PROJECT/
├── dsl_generator.py          # Main generator script
├── config.json               # Configuration file (edit this!)
├── requirements.txt          # Python dependencies
├── keys/                     # API keys directory (gitignored)
│   ├── key.json              # Your Google Cloud credentials
│   └── README.md             # Key setup instructions
├── SPs/                      # System prompts
│   ├── SystemPrompt1.txt
│   ├── SystemPrompt2.txt
│   └── ...
├── Shots/                    # Few-shot learning examples
│   ├── UserScenario_1.txt
│   ├── AssistantScenario_1.txt
│   └── ...
├── Scenarios/                # Test scenarios to generate DSL code for
│   ├── UserScenario_011.txt
│   ├── UserScenario_016.txt
│   └── ...
└── Results/                  # Generated DSL code outputs and metadata
    └── Runs/                  # Organized per-run output folders
        └── <Scenario>/<SystemPrompt>/RUN_<timestamp>/
            ├── dsl/           # Generated DSL outputs (.LIRAs)
            ├── compiler/      # Compiler outputs per attempt (*.compiler.txt)
            └── run_metadata.json
```

## How to Use the Generator

### 1. Configure Your Generation Run

Edit the `config.json` file to specify your configuration:

**Simple Integer Format (Recommended):**

```json
{
  "system_prompt": "SystemPrompt1.txt",
  "generation_model": "gemini-2.0-flash-lite-001",
  "repair_model": "gemini-2.0-flash-lite-001",
  "shots": 2,
  "scenario": "UserScenario_011.txt",
  "repair_prompt": "RepairPrompt.txt",
  "repair_shots": 0,
  "compiler_jar": "Compiler/liras-compiler.jar",
  "max_iterations": 10
}
```

Optional:

- **`repair_prompt`**: Repair-stage prompt template used only during refinement (when you paste compiler output). Defaults to `SPs/RepairPrompt.txt`. You can provide either a filename under `SPs/`, a relative path from the project root, or an absolute path.

Included templates:

- `SPs/RepairPrompt.txt`: Full repair guidance (fixes based on compiler output + pragmatic DSL repair heuristics).
- `SPs/RepairPrompt_Extended.txt`: More verbose/structured repair guidance (useful when compiler messages are ambiguous or repeated).

This will automatically load:

- `Shots/UserScenario_1.txt` → `Shots/AssistantScenario_1.txt`
- `Shots/UserScenario_2.txt` → `Shots/AssistantScenario_2.txt`

**Advanced Format (Custom Shot Pairs):**

```json
{
  "system_prompt": "SystemPrompt1.txt",
  "generation_model": "gemini-2.0-flash-lite-001",
  "repair_model": "gemini-2.0-flash-lite-001",
  "shots": [
    {
      "user": "UserScenario_1.txt",
      "assistant": "AssistantScenario_1.txt"
    },
    {
      "user": "UserScenario_2.txt",
      "assistant": "AssistantScenario_2.txt"
    }
  ],
  "scenario": "UserScenario_011.txt",
  "repair_prompt": "RepairPrompt.txt",
  "repair_shots": 0,
  "compiler_jar": "Compiler/liras-compiler.jar",
  "max_iterations": 10
}
```

**Configuration Options:**

- **`system_prompt`**: The system prompt file from the `SPs/` directory that defines the AI's role and instructions
- **`generation_model` (required)**: Vertex AI Gemini model name used for initial generation (e.g., `gemini-2.0-flash-lite-001`, `gemini-2.5-pro`)
- **`repair_model` (required)**: Vertex AI Gemini model name used for repair iterations (can be the same as `generation_model`)
- **`shots`**:
  - **Integer**: Number of shot examples (e.g., `2` loads UserScenario_1.txt + AssistantScenario_1.txt, UserScenario_2.txt + AssistantScenario_2.txt from Shots/ directory)
  - **Array**: Custom shot pairs with specific file names
  - **0 or []**: Zero-shot learning (no examples)
- **`scenario`**: The scenario file from `Scenarios/` directory to generate DSL code for
- **`repair_prompt`**: Repair-stage system prompt used by the repair chat. Defaults to `SPs/RepairPrompt.txt`. You can provide either a filename under `SPs/`, a relative path from the project root, or an absolute path.
- **`repair_shots`**: Optional additional few-shots used only for the repair chat. Can be an integer (like `shots`) or an explicit list of `{user, assistant}` pairs. Use `0`/`[]` for none.
- **`compiler_jar` (required)**: Path to the runnable compiler JAR used for validation (relative to project root or absolute path)
- **`max_iterations` (required)**: Maximum number of generate→compile→repair attempts before stopping

#### Available Gemini Models

**Gemini 2.5 Models:**

- **`gemini-2.5-pro`**: Most capable model for complex reasoning
  - Input: 2M tokens | Output: 8,192 tokens
  - Best for: Initial generation with complex DSL patterns
- **`gemini-2.5-flash`**: Fast responses with good quality
  - Input: 1M tokens | Output: 8,192 tokens
  - Best for: Repair iterations or when speed matters

**Gemini 3 Models (Preview):**

- **`gemini-3-pro-preview-0205`**: Latest preview with enhanced reasoning
  - Input: 2M tokens | Output: 8,192 tokens
  - Features: Thinking mode, better code generation
  - Best for: Experimental runs with cutting-edge capabilities
- **`gemini-3-flash-preview`**: Fast generation with improved quality
  - Input: 1M tokens | Output: 8,192 tokens
  - Best for: High-volume experiments or rapid iteration

**Recommended Configurations:**

```json
// Balanced: Quality generation + Fast repair
{
  "generation_model": "gemini-2.5-pro",
  "repair_model": "gemini-2.5-flash"
}

// Latest: Test Gemini 3 capabilities
{
  "generation_model": "gemini-3-pro-preview-0205",
  "repair_model": "gemini-3-pro-preview-0205"
}
```

**Token Usage per Iteration:**

- System Prompt: ~500-1,500 tokens
- Few-shot Examples (2 shots): ~2,000-4,000 tokens
- User Scenario: ~200-500 tokens
- DSL Output: ~300-1,000 tokens
- Compiler Feedback: ~100-500 tokens
- **Total**: ~3,000-8,000 tokens (well within all model limits)

### 2. Run the Generator

Activate your virtual environment and run:

```bash
source venv/bin/activate  # On macOS/Linux
python dsl_generator.py
```

### 3. Review the Context

The generator will display:

- System instruction
- Chat history (few-shot examples)
- Current scenario
- Token counts

**Type `y` to continue or `n` to abort.**

### 4. Get Initial DSL Code

The AI will generate DSL code based on your configuration, validate it with the local compiler JAR, and automatically repair on failure.

Each attempt is saved under a dedicated run directory:

```
Results/Runs/<Scenario>/<SystemPrompt>/RUN_<timestamp>/
  dsl/ITER0_<timestamp>.LIRAs
  compiler/ITER0_<timestamp>.LIRAs.compiler.txt
  run_metadata.json
```

### 5. Automated Repair Loop

On each iteration:

- The script saves the raw model output (no sanitization) as a `.LIRAs` file.
- Runs: `java -jar <compiler_jar> <dsl_file>`
- If compilation fails, the combined `stdout+stderr` is fed into a dedicated repair chat.

Repair chat methodology:

- A new conversation is created whose **system prompt** is the selected `repair_prompt` file.
- Each repair iteration sends a **single user message** that concatenates:
  - the generation system prompt
  - the generation scenario
  - the previous DSL output
  - the compiler output
- Repair iterations persist within the same repair chat session.

Stopping conditions:

- Success when the compiler returns exit code `0`, OR when the compiler emits no output (empty `stdout+stderr`).
- Stops after `max_iterations` attempts.
- If validation cannot be performed (missing `java`, missing JAR, missing DSL file, or compiler timeout), the run stops and does not attempt LLM repair.

### 6. Results

Each run produces a single `run_metadata.json` that is updated over time with telemetry, iteration outcomes, and a compact summary.

## Example Workflows

### Zero-Shot Learning (No Examples)

```json
{
  "system_prompt": "SystemPrompt1.txt",
  "shots": 0,
  "scenario": "UserScenario_011.txt"
}
```

### One-Shot Learning (One Example)

```json
{
  "system_prompt": "SystemPrompt2.txt",
  "shots": 1,
  "scenario": "UserScenario_016.txt"
}
```

This automatically loads `UserScenario_1.txt` + `AssistantScenario_1.txt` as chat history.

### Few-Shot Learning (Multiple Examples)

```json
{
  "system_prompt": "SystemPrompt3.txt",
  "shots": 2,
  "scenario": "UserScenario_029.txt"
}
```

This automatically loads:

- `UserScenario_1.txt` + `AssistantScenario_1.txt`
- `UserScenario_2.txt` + `AssistantScenario_2.txt`

### Custom Shot Pairs (Advanced)

```json
{
  "system_prompt": "SystemPrompt4.txt",
  "shots": [
    {
      "user": "UserScenario_1.txt",
      "assistant": "AssistantScenario_1.txt"
    },
    {
      "user": "CustomScenario.txt",
      "assistant": "CustomResponse.txt"
    }
  ],
  "scenario": "UserScenario_06.txt"
}
```

## Batch Runs (run_all_pairs)

Use `run_all_pairs.py` to execute all Scenario/SystemPrompt pairs with a shared template config.

Common commands:

```bash
python run_all_pairs.py
python run_all_pairs.py --shots 0,1,2
python run_all_pairs.py --generation-only
python run_all_pairs.py --disable-generation --shots 0,1,2
python run_all_pairs.py --disable-generation --shots 0,1,2 --compiler-timeout 120
python run_all_pairs.py --disable-generation --shots 0,1,2 --compiler-timeout 120 --inter-run-delay 2
python run_all_pairs.py --list-only
```

Notes:

- `--generation-only` skips compile/repair and only saves the generated DSL.
- `--disable-generation` loads DSL from cache (e.g., `DSL/Scenario_6/SP1_Shot0.txt`) and runs repair/compile.
- `--compiler-timeout` overrides compiler timeout in seconds for the batch run.
- `--inter-run-delay` adds a pause between runs to reduce resource pressure and transient timeouts.

## Run History Export (collect_run_history)

The `collect_run_history.py` script scans all `run_metadata.json` files under `Results/` and writes a single CSV:

```bash
python collect_run_history.py
```

Output:

- `Report/run_history.csv` (overwritten each run)

The CSV includes:

- Run metadata (scenario, system prompt, shots, models, timestamps, status).
- Cache provenance (cache source and resolved DSL path).
- Iteration diagnostics (compiler error/warning counts, error score, validation results).
- Derived metrics (AUC error score, improvement ratios, monotonicity, duration).

## Tips for Best Results

1. **System Prompts**: Create clear, detailed system prompts that explain the DSL syntax and rules
2. **Few-Shot Examples**: Provide diverse examples that cover different DSL patterns
3. **Compiler Feedback**: Check the saved `compiler/*.compiler.txt` outputs when diagnosing repeated failures
4. **Iterative Process**: The AI learns from errors - don't give up after the first attempt!

## Troubleshooting

**Error: config.json not found**

- Make sure `config.json` exists in the project root directory

**Error: Authentication failed**

- Verify `key.json` is present and valid
- Check that Vertex AI API is enabled in your Google Cloud project
- Ensure the service account has "Vertex AI User" role

**Error: File not found (scenarios/prompts)**

- Check that file names in `config.json` match exactly (case-sensitive)
- Verify files exist in the correct directories (`SPs/`, `Shots/`, `Scenarios/`)

**Error: `java` not found / compiler JAR cannot run**

- Ensure `java -version` works
- Ensure `compiler_jar` in `config.json` points to an existing JAR

## Deactivate Virtual Environment

When you're done:

```bash
deactivate
```

- `dsl_generator.py` - Main script
- `key.json` - Service account credentials (not committed to git)

## Configuration

Edit `config.json` to set:

- System prompt file
- Few-shot example pairs
- Target scenario file
- Compiler JAR path (`compiler_jar`)
- Maximum iterations (`max_iterations`)
