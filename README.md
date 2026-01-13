# RESEARCH_PROJECT

DSL Generator with Gemini API via Vertex AI - Automates the generation and refinement of domain-specific language code.

## Setup Instructions

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
   - Place it in the project root directory: `RESEARCH_PROJECT/key.json`
   - **Important**: This file is already in `.gitignore` and won't be committed to version control

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

The script will automatically detect and use the `key.json` file in the project directory.

## Alternative: Use gcloud CLI Authentication

Instead of a service account key, you can use gcloud CLI:

1. Install [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
2. Run: `gcloud auth application-default login`
3. Remove or rename `key.json` from the project directory
4. The script will use your gcloud credentials instead

## Project Structure

```
RESEARCH_PROJECT/
├── dsl_generator.py          # Main generator script
├── config.json               # Configuration file (edit this!)
├── key.json                  # Google Cloud credentials (not in git)
├── requirements.txt          # Python dependencies
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
    └── [scenario]/[system_prompt]/
```

## How to Use the Generator

### 1. Configure Your Generation Run

Edit the `config.json` file to specify your configuration:

**Simple Integer Format (Recommended):**

```json
{
  "system_prompt": "SystemPrompt1.txt",
  "shots": 2,
  "scenario": "UserScenario_011.txt"
}
```

This will automatically load:

- `Shots/UserScenario_1.txt` → `Shots/AssistantScenario_1.txt`
- `Shots/UserScenario_2.txt` → `Shots/AssistantScenario_2.txt`

**Advanced Format (Custom Shot Pairs):**

```json
{
  "system_prompt": "SystemPrompt1.txt",
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
  "scenario": "UserScenario_011.txt"
}
```

**Configuration Options:**

- **`system_prompt`**: The system prompt file from the `SPs/` directory that defines the AI's role and instructions
- **`shots`**:
  - **Integer**: Number of shot examples (e.g., `2` loads UserScenario_1.txt + AssistantScenario_1.txt, UserScenario_2.txt + AssistantScenario_2.txt from Shots/ directory)
  - **Array**: Custom shot pairs with specific file names
  - **0 or []**: Zero-shot learning (no examples)
- **`scenario`**: The scenario file from `Scenarios/` directory to generate DSL code for

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

The AI will generate DSL code based on your configuration. The code will be displayed in the terminal and automatically saved to:

```
Results/[scenario_name]/[system_prompt_name]/ITER0_[timestamp].txt
```

### 5. Iterative Refinement (Optional)

If the generated code has compilation errors:

1. Copy the error message from your DSL compiler
2. Paste it into the terminal when prompted
3. Press `Ctrl+D` (macOS/Linux) or `Ctrl+Z` then Enter (Windows) to submit
4. The AI will analyze the error and generate corrected code
5. Repeat as needed

**Special commands:**

- Type `success` if the code compiled successfully
- Type `exit` to quit without marking as successful

### 6. Results

Each iteration is saved with metadata:

```
Results/UserScenario_011/SystemPrompt1/
├── ITER0_20260113_143022.txt            # Initial generation
├── ITER0_20260113_143022_metadata.json  # Configuration metadata
├── ITER1_20260113_143145.txt            # First refinement
├── ITER1_20260113_143145_metadata.json
└── SUCCESS_20260113_143301.txt          # Successful compilation
```

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

## Tips for Best Results

1. **System Prompts**: Create clear, detailed system prompts that explain the DSL syntax and rules
2. **Few-Shot Examples**: Provide diverse examples that cover different DSL patterns
3. **Error Messages**: Paste complete error messages including line numbers for better refinement
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

## Deactivate Virtual Environment

When you're done:

```bash
deactivate
```

- `dsl_generator.py` - Main script
- `key.json` - Service account credentials (not committed to git)

## Configuration

Edit the `CONFIG` dictionary in `dsl_generator.py` to set:

- System prompt file
- Few-shot example pairs
- Target scenario file
