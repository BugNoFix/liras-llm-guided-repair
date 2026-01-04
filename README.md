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

If `requirements.txt` doesn't exist, install manually:

```bash
pip install google-cloud-aiplatform vertexai
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

- `SPs/` - System prompts
- `Shots/` - Few-shot learning examples (user scenarios + assistant DSL code)
- `Scenarios/` - Test scenarios to generate DSL code for
- `Results/` - Generated DSL code outputs and metadata
- `dsl_generator.py` - Main script
- `key.json` - Service account credentials (not committed to git)

## Configuration

Edit the `CONFIG` dictionary in `dsl_generator.py` to set:

- System prompt file
- Few-shot example pairs
- Target scenario file
