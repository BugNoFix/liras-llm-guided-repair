import os
import shlex
import subprocess
from google import genai
from google.genai import types
from pathlib import Path
from datetime import datetime
import json
from typing import Optional

import os
from pathlib import Path

# 🔴 KEY PATH CONFIGURATION
# For shared projects: place your personal key in keys/key.json
# The keys/ directory is gitignored to prevent sharing credentials
PROJECT_ROOT = Path(__file__).parent
KEYS_DIR = PROJECT_ROOT / "keys"

# Check for key in keys/ directory first (recommended for shared projects)
KEY_PATH = KEYS_DIR / "key.json"
if not KEY_PATH.exists():
    # Fallback to root directory for backward compatibility
    KEY_PATH = PROJECT_ROOT / "key.json"

if KEY_PATH.exists():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(KEY_PATH)
    print(f"✅ SUCCESS: Auth key found at {KEY_PATH}")
else:
    print(f"❌ ERROR: Auth key NOT found")
    print(f"Please place your key.json in one of these locations:")
    print(f"  - {KEYS_DIR / 'key.json'} (recommended for shared projects)")
    print(f"  - {PROJECT_ROOT / 'key.json'} (backward compatibility)")
    print(f"\nCreate the keys/ directory if it doesn't exist.")

class DSLGenerator:
    def __init__(self, project_id: str, location: str = "global", service_account_key: str = None):
        """
        Initialize the DSL Generator with Vertex AI credentials
        
        Args:
            project_id: Google Cloud Project ID
            location: Region for Vertex AI (default: global)
            service_account_key: Path to service account JSON key file (optional)
        """
        # Set credentials if service account key is provided
        if service_account_key:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_key
            print(f"✓ Using service account: {service_account_key}")
        
        # Initialize Gemini 3 client
        self.client = genai.Client(vertexai=True, project=project_id, location=location)
        self.project_id = project_id
        self.location = location
        
        # Define workspace paths
        self.base_path = Path(__file__).parent
        self.sp_path = self.base_path / "SPs"
        self.shots_path = self.base_path / "Shots"
        self.scenarios_path = self.base_path / "Scenarios"
        self.results_path = self.base_path / "Results"
        self.repair_prompt_template_path = self.base_path / "SPs" / "RepairPrompt.txt"
        
        self.model = None
        self.chat = None
        self.repair_model = None
        self.repair_chat = None
        self.current_config = {}
        self.last_dsl_code = None

        # Generation context (captured once during generate phase; reused during repair prompts)
        self.generation_system_prompt_text: Optional[str] = None
        self.generation_scenario_text: Optional[str] = None
        self.initial_dsl_from_generation: Optional[str] = None

        # Repair chat state
        self.repair_iteration_count = 0
        self.last_repair_prompt_included_previous_dsl: Optional[bool] = None

        # Per-run state (initialized when starting an automated session)
        self.run_id: Optional[str] = None
        self.run_dir: Optional[Path] = None
        self.run_dsl_dir: Optional[Path] = None
        self.run_compiler_dir: Optional[Path] = None
        self.run_metadata_path: Optional[Path] = None
        self.run_metadata: Optional[dict] = None

        # Last validation details (populated by validate_code)
        self.last_validation = {
            "returncode": None,
            "timed_out": False,
            "java_missing": False,
            "jar_missing": False,
            "dsl_missing": False,
        }

        # Lightweight, approximate usage telemetry (persists in result metadata).
        # We intentionally do NOT store prompt/response text here, only sizes/estimates.
        self.telemetry = {
            "llm_calls": 0,
            "prompt_chars_total": 0,
            "response_chars_total": 0,
            "prompt_tokens_est_total": 0,
            "response_tokens_est_total": 0,
            "last_call": None,
        }

    def _normalize_shots(self, shots) -> list[dict]:
        """Normalize shots config into a list of {user, assistant} pairs.

        Supports:
        - int: N -> UserScenario_1..N.txt + AssistantScenario_1..N.txt
        - list[dict]: already pairs
        - None/0/[]: no shots
        """
        if not shots:
            return []

        if isinstance(shots, int):
            if shots <= 0:
                return []
            shot_pairs = []
            for i in range(1, shots + 1):
                shot_pairs.append({
                    "user": f"UserScenario_{i}.txt",
                    "assistant": f"AssistantScenario_{i}.txt",
                })
            return shot_pairs

        if isinstance(shots, list):
            return shots

        raise ValueError("shots must be an integer, a list of pairs, or empty")

    def _estimate_tokens(self, text: str) -> int:
        """Very rough token estimate based on character length.

        Rule of thumb: ~4 characters per token for English-ish text.
        This is approximate and meant only for local monitoring.
        """
        if not text:
            return 0
        return max(1, int(round(len(text) / 4)))

    def _record_llm_call(self, kind: str, prompt_text: str, response_text: str) -> None:
        prompt_chars = len(prompt_text or "")
        response_chars = len(response_text or "")
        prompt_tokens_est = self._estimate_tokens(prompt_text or "")
        response_tokens_est = self._estimate_tokens(response_text or "")

        self.telemetry["llm_calls"] += 1
        self.telemetry["prompt_chars_total"] += prompt_chars
        self.telemetry["response_chars_total"] += response_chars
        self.telemetry["prompt_tokens_est_total"] += prompt_tokens_est
        self.telemetry["response_tokens_est_total"] += response_tokens_est
        self.telemetry["last_call"] = {
            "kind": kind,
            "timestamp": datetime.now().isoformat(),
            "prompt_chars": prompt_chars,
            "response_chars": response_chars,
            "prompt_tokens_est": prompt_tokens_est,
            "response_tokens_est": response_tokens_est,
        }

        if self.run_metadata is not None:
            self.run_metadata.setdefault("llm_call_history", []).append(self.telemetry["last_call"])
            self.run_metadata["telemetry"] = self.telemetry
            self._persist_run_metadata()

    def _init_run_metadata(self, config: dict, compiler_jar_path: Path, max_iterations: int) -> None:
        """Initialize per-run directory + metadata file.

        This is called once per run; metadata is then updated on every LLM call and validation.
        """
        sp_name = config["system_prompt"].replace(".txt", "")
        scenario_name = config["scenario"].replace(".txt", "")

        # User-friendly hierarchy: every run is fully contained in its own directory.
        # Results/
        #   Runs/<Scenario>/<SystemPrompt>/RUN_<run_id>/
        #     dsl/
        #     compiler/
        runs_root = self.results_path / "Runs" / scenario_name / sp_name
        runs_root.mkdir(parents=True, exist_ok=True)

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = runs_root / f"RUN_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.run_dsl_dir = self.run_dir / "dsl"
        self.run_compiler_dir = self.run_dir / "compiler"
        self.run_dsl_dir.mkdir(parents=True, exist_ok=True)
        self.run_compiler_dir.mkdir(parents=True, exist_ok=True)

        self.run_metadata_path = self.run_dir / "run_metadata.json"
        self.run_metadata = {
            "run_id": self.run_id,
            "run_started_at": datetime.now().isoformat(),
            "project_id": self.project_id,
            "location": self.location,
            "system_prompt": config.get("system_prompt"),
            "shots": config.get("shots"),
            "scenario": config.get("scenario"),
            "repair_prompt": str(self.repair_prompt_template_path),
            "generation_model": config.get("generation_model"),
            "repair_model": config.get("repair_model"),
            "repair_shots": config.get("repair_shots"),
            "compiler_jar": str(compiler_jar_path),
            "max_iterations": max_iterations,
            "run_dir": str(self.run_dir),
            "dsl_dir": str(self.run_dsl_dir),
            "compiler_dir": str(self.run_compiler_dir),
            "iterations": [],
            "telemetry": self.telemetry,
            "llm_call_history": [],
            "status": "running",
            "run_finished_at": None,
        }

        self._persist_run_metadata()

    def _persist_run_metadata(self) -> None:
        if not self.run_metadata_path or self.run_metadata is None:
            return

        # Compute a compact summary to simplify downstream analysis.
        iterations = self.run_metadata.get("iterations", []) or []
        validations_attempted = sum(1 for it in iterations if it.get("validated"))
        compiler_failures = sum(1 for it in iterations if it.get("validated") and not it.get("is_valid"))
        compiler_successes = sum(1 for it in iterations if it.get("validated") and it.get("is_valid"))
        setup_errors = sum(1 for it in iterations if it.get("ended_because") == "validation_setup_error")

        first_success_iter = None
        final_success_dsl_path = None
        for it in iterations:
            if it.get("validated") and it.get("is_valid"):
                first_success_iter = it.get("iteration")
                final_success_dsl_path = it.get("dsl_path")
                break

        total_compiler_feedback_chars = 0
        for it in iterations:
            total_compiler_feedback_chars += int(it.get("compiler_feedback_chars") or 0)

        self.run_metadata["summary"] = {
            "iterations_recorded": len(iterations),
            "validations_attempted": validations_attempted,
            "compiler_failures": compiler_failures,
            "compiler_successes": compiler_successes,
            "setup_errors": setup_errors,
            "first_success_iteration": first_success_iter,
            "final_success_dsl_path": final_success_dsl_path,
            "total_compiler_feedback_chars": total_compiler_feedback_chars,
            "llm_calls": int((self.telemetry or {}).get("llm_calls") or 0),
            "prompt_tokens_est_total": int((self.telemetry or {}).get("prompt_tokens_est_total") or 0),
            "response_tokens_est_total": int((self.telemetry or {}).get("response_tokens_est_total") or 0),
        }

        self.run_metadata["updated_at"] = datetime.now().isoformat()
        with open(self.run_metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.run_metadata, f, indent=2)
        
    def load_file(self, filepath: Path) -> str:
        """Load and return content from a text file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    def list_available_files(self):
        """List all available system prompts, shots, and scenarios"""
        print("\n=== Available System Prompts ===")
        sp_files = sorted(self.sp_path.glob("*.txt"))
        for i, f in enumerate(sp_files, 1):
            print(f"{i}. {f.name}")
        
        print("\n=== Available Shots ===")
        shot_files = sorted(self.shots_path.glob("*.txt"))
        for i, f in enumerate(shot_files, 1):
            print(f"{i}. {f.name}")
        
        print("\n=== Available Scenarios ===")
        scenario_files = sorted(self.scenarios_path.glob("*.txt"))
        for i, f in enumerate(scenario_files, 1):
            print(f"{i}. {f.name}")
    
    def start_conversation(self, system_prompt_file: str, shots, scenario_file: str, model_name: str):
        """
        Start a new conversation with configured system prompt, shot pairs, and scenario
        
        Args:
            system_prompt_file: Name of the system prompt file (e.g., "SystemPrompt1.txt")
            shots: Integer (number of shots, e.g., 2 loads UserScenario_1.txt + AssistantScenario_1.txt,
                          UserScenario_2.txt + AssistantScenario_2.txt) OR
                   List of dicts with 'user' and 'assistant' shot file names for backwards compatibility
            scenario_file: Name of the scenario file to process (e.g., "UserScenario_011.txt")
        """
        shot_pairs = self._normalize_shots(shots)
        
        # Store configuration
        self.current_config = {
            "system_prompt": system_prompt_file,
            "shots": shot_pairs,
            "scenario": scenario_file,
            "repair_prompt": str(self.repair_prompt_template_path),
            "timestamp": datetime.now().isoformat()
        }
        
        # Load system prompt
        system_prompt = self.load_file(self.sp_path / system_prompt_file)

        # Capture generation context for the repair phase
        self.generation_system_prompt_text = system_prompt
        
        # Store model name for chat
        self.model_name = model_name
        
        # Build chat history from few-shot examples
        history = []
        if shot_pairs:
            for pair in shot_pairs:
                # Load user scenario example
                user_content = self.load_file(self.shots_path / pair["user"])
                # Load assistant DSL code example
                assistant_content = self.load_file(self.shots_path / pair["assistant"])
                
                # Add to history
                history.append(types.Content(role="user", parts=[types.Part(text=user_content)]))
                history.append(types.Content(role="model", parts=[types.Part(text=assistant_content)]))
        
        # Store chat history and system prompt for subsequent messages
        self.chat_history = history
        self.system_prompt = system_prompt
        
        # Load the actual scenario to process
        scenario_content = self.load_file(self.scenarios_path / scenario_file)

        # Capture generation context for the repair phase
        self.generation_scenario_text = scenario_content
        
        print("\n=== Starting Conversation with Gemini ===")
        print(f"System Prompt: {system_prompt_file}")
        print(f"Chat History: {len(shot_pairs)} few-shot example(s)")
        for i, pair in enumerate(shot_pairs, 1):
            print(f"  Example {i}: {pair['user']} → {pair['assistant']}")
        print(f"Scenario: {scenario_file}")
        print("\n" + "="*50 + "\n")
        
        # Display the full context
        self._display_full_prompt(system_prompt, history, scenario_content)
        
        # Send the actual scenario and get DSL code
        current_history = self.chat_history + [types.Content(role="user", parts=[types.Part(text=scenario_content)])]
        
        response = self.client.models.generate_content(
            model=model_name,
            contents=current_history,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=1.0
            )
        )
        
        # Update chat history with user message and response
        self.chat_history.append(types.Content(role="user", parts=[types.Part(text=scenario_content)]))
        self.chat_history.append(types.Content(role="model", parts=[types.Part(text=response.text)]))

        # Telemetry: record prompt/response sizes (no sanitization)
        self._record_llm_call("generate", scenario_content, response.text)

        return response.text

    def _build_repair_user_prompt(
        self,
        compiler_output: str,
        previous_dsl: Optional[str] = None,
        include_previous_dsl: bool = False,
    ) -> str:
        """Build a minimal repair user prompt.

        - First repair turn: include PREVIOUS_DSL + COMPILER_OUTPUT.
        - Subsequent turns: include COMPILER_OUTPUT only (model uses its previous assistant
          output as PREVIOUS_DSL, per the repair system prompt).
        """
        if include_previous_dsl:
            return (
                "PREVIOUS_DSL:\n"
                + (previous_dsl or "")
                + "\n\nCOMPILER_OUTPUT:\n"
                + (compiler_output or "")
            )
        return "COMPILER_OUTPUT:\n" + (compiler_output or "")

    def _fill_repair_system_prompt_template(self, template_text: str) -> str:
        """Fill the repair system prompt template with static generation artifacts.

        We intentionally avoid str.format because the embedded DSL/system prompt/scenario
        may contain curly braces.
        """
        if not self.generation_system_prompt_text or not self.generation_scenario_text:
            raise RuntimeError("Generation context missing; cannot build filled repair system prompt")
        if self.initial_dsl_from_generation is None:
            raise RuntimeError("Initial DSL from generation missing; cannot build filled repair system prompt")

        filled = template_text
        replacements = {
            "generation_system_prompt": self.generation_system_prompt_text,
            "generation_user_scenario": self.generation_scenario_text,
            "initial_dsl": self.initial_dsl_from_generation,
        }

        for key, value in replacements.items():
            filled = filled.replace("{" + key + "}", value or "")
        return filled

    def _ensure_repair_chat(self, repair_model_name: str, repair_shots) -> None:
        """Start a dedicated repair chat session if not already created.

        The repair chat uses the configured repair prompt file content as system instruction.
        Optionally includes repair few-shot history if configured.
        """
        if hasattr(self, 'repair_chat_history'):
            return

        if not self.repair_prompt_template_path.exists():
            raise FileNotFoundError(f"Repair prompt not found: {self.repair_prompt_template_path}")

        repair_template = self.load_file(self.repair_prompt_template_path)
        repair_system_prompt = self._fill_repair_system_prompt_template(repair_template)
        self.repair_model_name = repair_model_name
        self.repair_system_prompt = repair_system_prompt

        history = []
        shot_pairs = self._normalize_shots(repair_shots)
        if shot_pairs:
            for pair in shot_pairs:
                user_content = self.load_file(self.shots_path / pair["user"])
                assistant_content = self.load_file(self.shots_path / pair["assistant"])
                history.append(types.Content(role="user", parts=[types.Part(text=user_content)]))
                history.append(types.Content(role="model", parts=[types.Part(text=assistant_content)]))

        self.repair_chat_history = history
        self.repair_iteration_count = 0

        if self.run_metadata is not None:
            self.run_metadata.setdefault("repair", {})
            self.run_metadata["repair"].setdefault("repair_chat_initialized_at", datetime.now().isoformat())
            self.run_metadata["repair"]["repair_prompt_template_path"] = str(self.repair_prompt_template_path)
            self.run_metadata["repair"]["repair_system_prompt_chars"] = len(repair_system_prompt or "")
            self._persist_run_metadata()

    def repair_with_compiler_output(self, previous_dsl: str, compiler_output: str, repair_model_name: str, repair_shots) -> str:
        """Run one repair iteration using the dedicated repair chat."""
        self._ensure_repair_chat(repair_model_name=repair_model_name, repair_shots=repair_shots)

        include_previous_dsl = self.repair_iteration_count == 0
        # Reliability guardrail: if previous_dsl is empty/missing, include it.
        if not include_previous_dsl and not (previous_dsl or "").strip():
            include_previous_dsl = True

        prompt = self._build_repair_user_prompt(
            compiler_output=compiler_output,
            previous_dsl=previous_dsl,
            include_previous_dsl=include_previous_dsl,
        )
        self.last_repair_prompt_included_previous_dsl = include_previous_dsl

        print("\n=== Sending repair request to Gemini (repair chat) ===")
        
        # Build current history with new user message
        current_history = self.repair_chat_history + [types.Content(role="user", parts=[types.Part(text=prompt)])]
        
        response = self.client.models.generate_content(
            model=self.repair_model_name,
            contents=current_history,
            config=types.GenerateContentConfig(
                system_instruction=self.repair_system_prompt,
                temperature=1.0
            )
        )
        
        # Update repair chat history
        self.repair_chat_history.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
        self.repair_chat_history.append(types.Content(role="model", parts=[types.Part(text=response.text)]))

        self.repair_iteration_count += 1

        # Telemetry: record repair prompt/response sizes (no sanitization)
        self._record_llm_call("repair", prompt, response.text)
        return response.text
    
    def _display_full_prompt(self, system_prompt: str, history: list, scenario: str):
        """Display the full context that will be sent to the AI"""
        print("\n" + "="*80)
        print("FULL CONTEXT BEING SENT TO AI:")
        print("="*80)
        print("\n[SYSTEM INSTRUCTION]")
        print(system_prompt)
        print("\n" + "-"*80)
        
        if history:
            print("\n[CHAT HISTORY - Few-Shot Examples]")
            for i, content in enumerate(history):
                role = content.role.upper()
                text = content.parts[0].text if content.parts else ""
                print(f"\n{role} Message {i//2 + 1}:")
                print(text)
                print("-"*80)
        
        print("\n[CURRENT USER PROMPT]")
        print(scenario)
        print("="*80)
        print(f"System instruction length: {len(system_prompt)} characters")
        print(f"History messages: {len(history)}")
        print(f"Current prompt length: {len(scenario)} characters")
        print("="*80 + "\n")
        
        # Ask if user wants to continue
        response = input("Continue with this configuration? (y/n): ").strip().lower()
        if response != 'y':
            print("Aborted by user.")
            exit(0)
    
    def _extract_dsl_code(self, response_text: str) -> str:
        """
        Return the model output as-is.

        Note: Output formatting/constraints are enforced via editable prompt files (system prompt
        and repair prompt), not by in-code sanitization.
        
        Args:
            response_text: Raw response from the AI
            
        Returns:
            Clean DSL code without markdown or explanations
        """
        return response_text

    def validate_code(self, dsl_filepath: Path, compiler_jar_path: Path) -> tuple[bool, str]:
        """Validate DSL code by running the local Java-based compiler.

        Runs:
            java -jar <compiler_jar_path> <dsl_filepath>

        Returns:
            (is_valid, feedback)
            - is_valid is True when exit code == 0
            - feedback is combined stdout+stderr for logging/repair
        """
        dsl_filepath = Path(dsl_filepath)
        compiler_jar_path = Path(compiler_jar_path)

        # Reset last validation markers
        self.last_validation = {
            "returncode": None,
            "timed_out": False,
            "java_missing": False,
            "jar_missing": False,
            "dsl_missing": False,
        }

        if not compiler_jar_path.exists():
            self.last_validation["jar_missing"] = True
            return (
                False,
                (
                    "[VALIDATION_SETUP_ERROR] Compiler JAR not found.\n"
                    f"Expected at: {compiler_jar_path}\n"
                    "Set 'compiler_jar' in config.json to the correct path, or place the JAR there."
                ),
            )

        if not dsl_filepath.exists():
            self.last_validation["dsl_missing"] = True
            return (
                False,
                (
                    "[VALIDATION_SETUP_ERROR] DSL file to validate was not found.\n"
                    f"Expected at: {dsl_filepath}"
                ),
            )

        cmd = ["java", "-jar", str(compiler_jar_path), str(dsl_filepath)]

        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except FileNotFoundError as e:
            # Most commonly: Java not installed / not on PATH
            self.last_validation["java_missing"] = True
            feedback = (
                "[VALIDATION_SETUP_ERROR] Compiler invocation failed: 'java' was not found on PATH.\n"
                "Install a JRE/JDK and ensure 'java' is available, then retry.\n\n"
                f"Command: {shlex.join(cmd)}\n"
                f"Details: {e}"
            )
            return False, feedback
        except subprocess.TimeoutExpired:
            self.last_validation["timed_out"] = True
            feedback = (
                "[VALIDATION_SETUP_ERROR] Compiler invocation timed out after 10 seconds.\n"
                f"Command: {shlex.join(cmd)}\n"
                f"DSL file: {dsl_filepath}"
            )
            return False, feedback

        self.last_validation["returncode"] = completed.returncode

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        feedback = (stdout + ("\n" if stdout and stderr else "") + stderr).strip("\n")
        return completed.returncode == 0, feedback

    def configure_repair_prompt(self, repair_prompt: Optional[str]):
        """Configure which repair prompt template to use.

        If `repair_prompt` is:
        - None/empty: defaults to SPs/RepairPrompt.txt
        - a filename: resolved under SPs/
        - a relative path: resolved relative to project root
        - an absolute path: used as-is
        """
        if not repair_prompt:
            self.repair_prompt_template_path = self.base_path / "SPs" / "RepairPrompt.txt"
            return

        candidate = Path(repair_prompt)
        if candidate.is_absolute() and candidate.exists():
            self.repair_prompt_template_path = candidate
            return

        rel_to_root = self.base_path / repair_prompt
        if rel_to_root.exists():
            self.repair_prompt_template_path = rel_to_root
            return

        rel_to_sps = self.sp_path / repair_prompt
        self.repair_prompt_template_path = rel_to_sps
    
    def refine_with_error(self, error_message: str) -> str:
        """
        Send compilation error back to the API for refinement
        
        Args:
            error_message: The error message from compilation attempt
            
        Returns:
            Refined DSL code
        """
        if not self.chat:
            raise RuntimeError("No active conversation. Start a conversation first.")

        template = None
        if self.repair_prompt_template_path.exists():
            template = self.load_file(self.repair_prompt_template_path)

        if template and ("{previous_dsl}" in template) and ("{compiler_output}" in template):
            refinement_prompt = template.format(
                previous_dsl=self.last_dsl_code or "",
                compiler_output=error_message or "",
            )
        else:
            refinement_prompt = (
                "PREVIOUS_DSL:\n"
                + (self.last_dsl_code or "")
                + "\n\nCOMPILER_OUTPUT:\n"
                + (error_message or "")
                + "\n\nReturn ONLY the corrected DSL text."
            )
        
        print("\n=== Sending error feedback to Gemini ===")
        print(f"Error:\n{error_message}")
        print("\n" + "="*50 + "\n")
        
        response = self.chat.send_message(refinement_prompt)

        # Telemetry: record repair prompt/response sizes (no sanitization)
        self._record_llm_call("repair", refinement_prompt, response.text)
        return response.text
    
    def save_result(self, dsl_code: str, iteration: int = 0, success: bool = False) -> Path:
        """
        Save generated DSL code to results directory
        
        Args:
            dsl_code: The generated DSL code
            iteration: The iteration number (0 for initial, 1+ for refinements)
            success: Whether this version compiled successfully
        """
        if self.run_dsl_dir is None:
            print("Warning: No active run directory; cannot save result")
            return Path()
        
        # Create filename
        status = "SUCCESS" if success else f"ITER{iteration}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{status}_{timestamp}.LIRAs"
        
        # Save DSL code
        filepath = self.run_dsl_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(dsl_code)
        
        print(f"\n✓ Saved result to: {filepath}")

        return filepath
    
    def run_automated_session(self, config: dict):
        """Run an automated session with hardcoded configuration"""
        print("\n" + "="*60)
        print("DSL Generator - Automated Mode")
        print("="*60)
        
        shot_pairs = self._normalize_shots(config.get('shots'))
        
        # Display configuration
        print("\n=== CURRENT CONFIGURATION ===")
        print(f"System Prompt: {config['system_prompt']}")
        if shot_pairs:
            print(f"Shot Pairs: {len(shot_pairs)} example(s)")
            for i, pair in enumerate(shot_pairs, 1):
                print(f"  Example {i}: {pair['user']} → {pair['assistant']}")
        else:
            print("Shot Pairs: None (zero-shot learning)")
        print(f"Scenario: {config['scenario']}")
        print(f"Generation Model: {config['generation_model']}")
        print(f"Repair Model: {config['repair_model']}")
        if config.get("repair_shots"):
            repair_shot_pairs = self._normalize_shots(config.get("repair_shots"))
            print(f"Repair Shot Pairs: {len(repair_shot_pairs)} example(s)")
        else:
            print("Repair Shot Pairs: None")
        print("="*60 + "\n")

        # Optional: configure which repair prompt to use for refinement iterations
        self.configure_repair_prompt(config.get("repair_prompt"))

        # Resolve compiler JAR path (relative paths are relative to project root)
        compiler_jar_path = Path(config["compiler_jar"])
        if not compiler_jar_path.is_absolute():
            compiler_jar_path = (self.base_path / compiler_jar_path)

        if not compiler_jar_path.exists():
            print("\n❌ Error: compiler_jar path does not exist")
            print(f"Path: {compiler_jar_path}")
            print("Fix config.json to point at your compiler JAR.")
            return

        max_iterations = int(config["max_iterations"])
        if max_iterations < 1:
            max_iterations = 1

        # Initialize per-run metadata and output directory
        self._init_run_metadata(config, compiler_jar_path, max_iterations)
        
        # Start conversation and get initial DSL code
        try:
            response_text = self.start_conversation(
                config['system_prompt'],
                config['shots'],
                config['scenario'],
                model_name=config['generation_model'],
            )
            # IMPORTANT: preserve raw model output exactly as received (no sanitization)
            dsl_code = response_text
            self.last_dsl_code = dsl_code
            self.initial_dsl_from_generation = dsl_code

            print("\n=== GENERATED DSL CODE ===")
            print(dsl_code)
            print("\n" + "="*60 + "\n")

            # Automated compile/repair loop
            iteration = 0
            while iteration < max_iterations:
                print("\n" + "="*60)
                print(f"Validation iteration {iteration + 1}/{max_iterations}")
                print(f"Compiler JAR: {compiler_jar_path}")
                print("="*60)

                # Save the raw DSL output immediately for debugging/repro
                saved_dsl_path = self.save_result(dsl_code, iteration=iteration, success=False)

                # Validate via local compiler
                is_valid, feedback = self.validate_code(saved_dsl_path, compiler_jar_path)

                # Persist compiler output to a sidecar file for research/debugging
                if self.run_compiler_dir is None:
                    compiler_output_path = saved_dsl_path.with_suffix(saved_dsl_path.suffix + ".compiler.txt")
                else:
                    compiler_output_path = self.run_compiler_dir / (saved_dsl_path.name + ".compiler.txt")
                try:
                    with open(compiler_output_path, "w", encoding="utf-8") as f:
                        f.write(feedback or "")
                except Exception as e:
                    print(f"Warning: Could not write compiler output file: {e}")

                # If validation can't run (missing Java/JAR/timeout), stop rather than prompting LLM.
                if feedback.startswith("[VALIDATION_SETUP_ERROR]"):
                    print("\n✗ Validation could not be performed")
                    print("\n=== VALIDATION SETUP ERROR ===")
                    print(feedback)
                    print("\nStopping without attempting LLM repair.")

                    if self.run_metadata is not None:
                        self.run_metadata["status"] = "setup_error"
                        self.run_metadata["run_finished_at"] = datetime.now().isoformat()
                        self.run_metadata.setdefault("iterations", []).append(
                            {
                                "iteration": iteration,
                                "dsl_path": str(saved_dsl_path),
                                "compiler_output_path": str(compiler_output_path),
                                "validated": False,
                                "is_valid": False,
                                "validation": self.last_validation,
                                "ended_because": "validation_setup_error",
                            }
                        )
                        self._persist_run_metadata()
                    break

                # Initialize repair chat as soon as compiler output is available.
                # (This creates the chat + sets system prompt, but does not call the model.)
                self._ensure_repair_chat(
                    repair_model_name=config["repair_model"],
                    repair_shots=config.get("repair_shots"),
                )

                if is_valid:
                    print("\n✓ Compiler reported SUCCESS")
                    success_artifact = self.save_result(dsl_code, iteration=iteration, success=True)
                    if self.run_metadata is not None:
                        self.run_metadata["status"] = "success"
                        self.run_metadata["run_finished_at"] = datetime.now().isoformat()
                        self.run_metadata.setdefault("iterations", []).append(
                            {
                                "iteration": iteration,
                                "dsl_path": str(saved_dsl_path),
                                "success_artifact_path": str(success_artifact),
                                "compiler_output_path": str(compiler_output_path),
                                "validated": True,
                                "is_valid": True,
                                "validation": self.last_validation,
                            }
                        )
                        self._persist_run_metadata()

                    print("\n✓ Session completed successfully!")
                    break

                # Guardrail: if the compiler produced no error output at all, treat as success.
                # This helps when the compiler signals success without a clean exit code.
                no_error_output = not (feedback or "").strip()
                if no_error_output:
                    print("\n✓ No compiler errors detected (empty output)")
                    success_artifact = self.save_result(dsl_code, iteration=iteration, success=True)

                    if self.run_metadata is not None:
                        self.run_metadata["status"] = "success_no_output"
                        self.run_metadata["run_finished_at"] = datetime.now().isoformat()
                        self.run_metadata.setdefault("iterations", []).append(
                            {
                                "iteration": iteration,
                                "dsl_path": str(saved_dsl_path),
                                "success_artifact_path": str(success_artifact),
                                "compiler_output_path": str(compiler_output_path),
                                "validated": True,
                                "is_valid": True,
                                "validation": self.last_validation,
                                "ended_because": "no_compiler_output",
                            }
                        )
                        self._persist_run_metadata()

                    print("\n✓ Session completed successfully!")
                    break

                print("\n✗ Compiler reported FAILURE")
                if feedback:
                    print("\n=== COMPILER OUTPUT (stdout+stderr) ===")
                    print(feedback)
                    print("\n" + "="*60 + "\n")
                else:
                    print("\n(No compiler output captured.)\n")

                if iteration + 1 >= max_iterations:
                    print(f"\nReached max iterations ({max_iterations}). Stopping.")

                    if self.run_metadata is not None:
                        self.run_metadata["status"] = "max_iterations_reached"
                        self.run_metadata["run_finished_at"] = datetime.now().isoformat()
                        self.run_metadata.setdefault("iterations", []).append(
                            {
                                "iteration": iteration,
                                "dsl_path": str(saved_dsl_path),
                                "compiler_output_path": str(compiler_output_path),
                                "validated": True,
                                "is_valid": False,
                                "validation": self.last_validation,
                                "compiler_feedback_chars": len(feedback or ""),
                            }
                        )
                        self._persist_run_metadata()
                    break

                if self.run_metadata is not None:
                    self.run_metadata.setdefault("iterations", []).append(
                        {
                            "iteration": iteration,
                            "dsl_path": str(saved_dsl_path),
                            "compiler_output_path": str(compiler_output_path),
                            "validated": True,
                            "is_valid": False,
                            "validation": self.last_validation,
                            "compiler_feedback_chars": len(feedback or ""),
                        }
                    )
                    self._persist_run_metadata()

                # Repair with LLM using a dedicated repair chat (system prompt = repair prompt).
                dsl_code = self.repair_with_compiler_output(
                    previous_dsl=dsl_code,
                    compiler_output=feedback,
                    repair_model_name=config['repair_model'],
                    repair_shots=config.get('repair_shots'),
                )

                # Annotate last iteration with repair prompt mode for research.
                if self.run_metadata is not None:
                    try:
                        last_it = self.run_metadata.get("iterations", [])[-1]
                        last_it["repair_prompt_included_previous_dsl"] = bool(
                            self.last_repair_prompt_included_previous_dsl
                        )
                        last_it["repair_prompt_mode"] = (
                            "dsl_plus_compiler"
                            if self.last_repair_prompt_included_previous_dsl
                            else "compiler_only"
                        )
                        self._persist_run_metadata()
                    except Exception:
                        pass
                self.last_dsl_code = dsl_code
                print("\n=== REFINED DSL CODE ===")
                print(dsl_code)
                print("\n" + "="*60 + "\n")

                iteration += 1
        
        except Exception as e:
            print(f"\nError during session: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point"""
    print("DSL Generator with Gemini API")
    print("="*60)
    
    # Load configuration from config.json
    config_file = Path(__file__).parent / "config.json"
    if not config_file.exists():
        print("\n❌ Error: config.json file not found!")
        print("Please create a config.json file with the following structure:")
        print("""
{
  "system_prompt": "SystemPrompt1.txt",
    "generation_model": "gemini-2.0-flash-lite-001",
    "repair_model": "gemini-2.0-flash-lite-001",
  "shots": 2,
    "scenario": "UserScenario_011.txt",
        "repair_prompt": "RepairPrompt.txt",
    "compiler_jar": "Compiler/liras-compiler.jar",
        "max_iterations": 10,
        "repair_shots": 0
}

Or for custom shot pairs:
{
  "system_prompt": "SystemPrompt1.txt",
    "generation_model": "gemini-2.0-flash-lite-001",
    "repair_model": "gemini-2.0-flash-lite-001",
  "shots": [
    {
      "user": "UserScenario_1.txt",
      "assistant": "AssistantScenario_1.txt"
    }
  ],
    "scenario": "UserScenario_011.txt",
        "repair_prompt": "RepairPrompt.txt",
    "compiler_jar": "Compiler/liras-compiler.jar",
        "max_iterations": 10,
        "repair_shots": []
}
""")
        return
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"\n✓ Loaded configuration from config.json")
    except Exception as e:
        print(f"\n❌ Error reading config.json: {e}")
        return
    
    # Validate configuration
    required_keys = [
        "system_prompt",
        "generation_model",
        "repair_model",
        "shots",
        "scenario",
        "compiler_jar",
        "max_iterations",
    ]
    for key in required_keys:
        if key not in config:
            print(f"\n❌ Error: Missing required key '{key}' in config.json")
            return

    # Validate max_iterations type
    if not isinstance(config["max_iterations"], int):
        print(f"\n❌ Error: 'max_iterations' must be an integer in config.json")
        return

    # Validate model keys
    if not isinstance(config["generation_model"], str) or not config["generation_model"].strip():
        print(f"\n❌ Error: 'generation_model' must be a non-empty string in config.json")
        return
    if not isinstance(config["repair_model"], str) or not config["repair_model"].strip():
        print(f"\n❌ Error: 'repair_model' must be a non-empty string in config.json")
        return
    
    # Validate shots type
    if not isinstance(config["shots"], (int, list)):
        print(f"\n❌ Error: 'shots' must be an integer or a list in config.json")
        return

    # Validate optional repair_shots type
    if "repair_shots" in config and config["repair_shots"] is not None:
        if not isinstance(config["repair_shots"], (int, list)):
            print(f"\n❌ Error: 'repair_shots' must be an integer or a list in config.json")
            return
    
    # Use the KEY_PATH that was already detected at module level
    service_account_key = None
    project_id = None
    
    if KEY_PATH.exists():
        print("✓ Found key.json file")
        try:
            with open(KEY_PATH, 'r') as f:
                key_data = json.load(f)
                project_id = key_data.get("project_id")
                service_account_key = str(KEY_PATH)
                print(f"✓ Using service account: {key_data.get('client_email')}")
                print(f"✓ Project ID: {project_id}")
        except Exception as e:
            print(f"Warning: Could not read key.json: {e}")
    
    # If no key.json found, ask user for authentication method
    if not KEY_PATH.exists() or not project_id:
        print("\n=== AUTHENTICATION OPTIONS ===")
        print("1. Use gcloud CLI authentication (recommended for local dev)")
        print("2. Use service account JSON key file")
        
        auth_choice = input("\nChoice (1/2): ").strip()
        
        if auth_choice == "2":
            service_account_key = input("Enter path to service account JSON key file: ").strip()
            if not os.path.exists(service_account_key):
                print(f"Error: File not found: {service_account_key}")
                return
        elif auth_choice == "1":
            print("\n✓ Using gcloud CLI authentication")
            print("Make sure you've run: gcloud auth application-default login")
        else:
            print("Invalid choice. Exiting.")
            return
        
        # Get project ID
        project_id = input("\nEnter your Google Cloud Project ID: ").strip()
        
        if not project_id:
            print("Error: Project ID is required")
            return
    
    # Initialize generator
    generator = DSLGenerator(project_id, service_account_key=service_account_key)
    
    # Run automated session with configuration from config.json
    generator.run_automated_session(config)


if __name__ == "__main__":
    main()
