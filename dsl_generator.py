import os
import shlex
import subprocess
from google import genai
from google.genai import types
from pathlib import Path
from datetime import datetime
import json
from typing import Optional

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
else:
    # Rely on Application Default Credentials (ADC) when available.
    pass

class DSLGenerator:
    def __init__(
        self,
        project_id: str,
        location: str = "global",
        service_account_key: str = None,
        *,
        generation_temperature: float = 1.0,
        repair_temperature: float = 0.2,
    ):
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
        
        # Initialize Gemini 3 client
        self.client = genai.Client(vertexai=True, project=project_id, location=location)
        self.project_id = project_id
        self.location = location

        # Decoding parameters (tracked as experimental variables)
        self.generation_temperature = float(generation_temperature)

        # Server-side chat support (SDK/version dependent). If unavailable or errors at runtime,
        # we fall back to stateless generate_content with explicit contents history.
        self.supports_server_chat = hasattr(self.client, "chats")
        
        # Define workspace paths
        self.base_path = Path(__file__).parent
        self.sp_path = self.base_path / "SPs"
        self.shots_path = self.base_path / "Shots"
        self.scenarios_path = self.base_path / "Scenarios"
        self.results_path = self.base_path / "Results"
        self.repair_prompt_template_path = self.base_path / "SPs" / "RepairPrompt.txt"
        
        self.model = None
        # Backward-compat alias: when server-side chat is enabled, we set self.chat to the
        # generation chat instance.
        self.chat = None
        self.generation_chat = None
        self.repair_model = None
        self.repair_chat = None
        self.chat_history = []
        self.repair_chat_history = None
        self.current_config = {}
        self.last_dsl_code = None

        # Generation context (captured once during generate phase; reused during repair prompts)
        self.generation_system_prompt_text: Optional[str] = None
        self.generation_scenario_text: Optional[str] = None
        self.initial_dsl_from_generation: Optional[str] = None

        # Repair chat state
        self.repair_iteration_count = 0
        self.last_repair_prompt_included_previous_dsl: Optional[bool] = None

        # Repair strategy: mitigate context poisoning by making each repair call stateless.
        # We still reuse the same repair system prompt and (optional) repair shots, but we do
        # NOT include prior failed repair turns in subsequent requests.
        self.repair_stateless = True

        # Repair decoding defaults (favor determinism to reduce regressions).
        self.repair_temperature = float(repair_temperature)

        # Approximate compiler error progress tracking (used as a light “monotonic improvement” signal)
        self.compiler_error_score_history: list[dict] = []

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

    def _maybe_create_server_chat(self, *, model_name: str, system_instruction: str, history: list[types.Content]):
        """Best-effort create a server-side chat session.

        If the installed SDK/model does not support server-side chats, returns None and flips
        supports_server_chat off for the rest of the run.
        """
        if not self.supports_server_chat:
            return None

        try:
            return self.client.chats.create(
                model=model_name,
                history=history,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=self.generation_temperature,
                ),
            )
        except Exception as e:
            # Batch-friendly: fall back silently when server-side chats are unavailable.
            self.supports_server_chat = False
            return None

    def _build_shot_history(self, shot_pairs: list[dict]) -> list[types.Content]:
        """Build alternating user/model Content messages from configured shot pairs."""
        history: list[types.Content] = []
        if not shot_pairs:
            return history

        for pair in shot_pairs:
            user_content = self.load_file(self.shots_path / pair["user"])
            assistant_content = self.load_file(self.shots_path / pair["assistant"])
            history.append(types.Content(role="user", parts=[types.Part(text=user_content)]))
            history.append(types.Content(role="model", parts=[types.Part(text=assistant_content)]))
        return history

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
            "generation_temperature": float(getattr(self, "generation_temperature", 1.0)),
            "repair_model": config.get("repair_model"),
            "repair_temperature": float(getattr(self, "repair_temperature", 0.2)),
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
    
    def list_available_files(self) -> dict:
        """Return available system prompts, shots, and scenarios (no printing)."""
        sp_files = [p.name for p in sorted(self.sp_path.glob("*.txt"))]
        shot_files = [p.name for p in sorted(self.shots_path.glob("*.txt"))]
        scenario_files = [p.name for p in sorted(self.scenarios_path.glob("*.txt"))]
        return {
            "system_prompts": sp_files,
            "shots": shot_files,
            "scenarios": scenario_files,
        }
    
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
        history = self._build_shot_history(shot_pairs)
        
        # Store chat history and system prompt for subsequent messages
        self.chat_history = history
        self.system_prompt = system_prompt
        
        # Load the actual scenario to process
        scenario_content = self.load_file(self.scenarios_path / scenario_file)

        # Capture generation context for the repair phase
        self.generation_scenario_text = scenario_content

        # Batch-friendly: avoid printing full prompts/configuration to stdout.

        # Send the actual scenario and get DSL code
        # Prefer server-side chat (stateful) if available, else fallback to stateless contents replay.
        if self.generation_chat is None and self.chat is None:
            self.generation_chat = self._maybe_create_server_chat(
                model_name=model_name,
                system_instruction=system_prompt,
                history=self.chat_history,
            )
            # Backward compat alias
            self.chat = self.generation_chat

        if self.generation_chat is not None:
            response = self.generation_chat.send_message(scenario_content)
        else:
            current_history = self.chat_history + [
                types.Content(role="user", parts=[types.Part(text=scenario_content)])
            ]
            response = self.client.models.generate_content(
                model=model_name,
                contents=current_history,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=self.generation_temperature,
                ),
            )
        
        response_text = response.text or ""

        # Update chat history with user message and response
        self.chat_history.append(types.Content(role="user", parts=[types.Part(text=scenario_content)]))
        self.chat_history.append(types.Content(role="model", parts=[types.Part(text=response_text)]))

        # Telemetry: record prompt/response sizes (no sanitization)
        self._record_llm_call("generate", scenario_content, response_text)

        return response_text

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
        strategy = (
            "FIX_STRATEGY:\n"
            "1) Focus on the *first/root* compiler error; later errors may be cascading.\n"
            "2) Make the smallest possible change to fix it (avoid rewrites).\n"
            "3) Preserve unrelated structure and names; do not introduce new features.\n"
            "4) Return ONLY the full corrected LIrAs DSL text (no markdown, no explanations).\n"
        )

        if include_previous_dsl:
            return (
                strategy
                + "\nPREVIOUS_DSL:\n"
                + (previous_dsl or "")
                + "\n\nCOMPILER_OUTPUT:\n"
                + (compiler_output or "")
            )
        return strategy + "\nCOMPILER_OUTPUT:\n" + (compiler_output or "")

    def _score_compiler_output(self, compiler_output: str) -> dict:
        """Compute a lightweight, approximate error severity score.

        This is heuristic (compiler output formats vary) but useful for tracking whether
        iterations are trending better or worse.
        """
        text = (compiler_output or "").strip()
        if not text:
            return {
                "feedback_chars": 0,
                "lines": 0,
                "error_lines": 0,
                "warning_lines": 0,
                "score": 0,
            }

        lines = text.splitlines()
        error_lines = 0
        warning_lines = 0
        for line in lines:
            lowered = line.lower()
            # Broad matching; avoids assuming a specific compiler format.
            if "error" in lowered:
                error_lines += 1
            if "warning" in lowered:
                warning_lines += 1

        # Score weights: errors dominate; long feedback also usually indicates worse state.
        score = (error_lines * 10) + (warning_lines * 2) + min(len(text), 2000) // 200

        return {
            "feedback_chars": len(text),
            "lines": len(lines),
            "error_lines": error_lines,
            "warning_lines": warning_lines,
            "score": int(score),
        }

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
        if self.repair_chat_history is not None:
            return

        if not self.repair_prompt_template_path.exists():
            raise FileNotFoundError(f"Repair prompt not found: {self.repair_prompt_template_path}")

        repair_template = self.load_file(self.repair_prompt_template_path)
        repair_system_prompt = self._fill_repair_system_prompt_template(repair_template)
        self.repair_model_name = repair_model_name
        self.repair_system_prompt = repair_system_prompt

        shot_pairs = self._normalize_shots(repair_shots)
        history = self._build_shot_history(shot_pairs)

        self.repair_chat_history = history
        self.repair_iteration_count = 0

        # Best-effort server-side repair chat session.
        # NOTE: When repair_stateless is enabled, we intentionally do NOT use a stateful chat
        # for repair iterations (to avoid history poisoning). We still keep the shot history
        # around and re-send it each turn.
        if self.repair_stateless:
            self.repair_chat = None
        else:
            self.repair_chat = self._maybe_create_server_chat(
                model_name=self.repair_model_name,
                system_instruction=self.repair_system_prompt,
                history=self.repair_chat_history,
            )

        if self.run_metadata is not None:
            self.run_metadata.setdefault("repair", {})
            self.run_metadata["repair"].setdefault("repair_chat_initialized_at", datetime.now().isoformat())
            self.run_metadata["repair"]["repair_prompt_template_path"] = str(self.repair_prompt_template_path)
            self.run_metadata["repair"]["repair_system_prompt_chars"] = len(repair_system_prompt or "")
            self.run_metadata["repair"]["repair_stateless"] = bool(self.repair_stateless)
            self.run_metadata["repair"]["repair_uses_server_chat"] = bool(self.repair_chat is not None)
            self.run_metadata["repair"]["repair_temperature"] = float(self.repair_temperature)
            self._persist_run_metadata()

    def repair_with_compiler_output(self, previous_dsl: str, compiler_output: str, repair_model_name: str, repair_shots) -> str:
        """Run one repair iteration using the dedicated repair chat."""
        self._ensure_repair_chat(repair_model_name=repair_model_name, repair_shots=repair_shots)

        # Poisoning mitigation: always include the current DSL + current compiler output.
        # This anchors each iteration to the latest attempt instead of relying on prior
        # conversation state.
        include_previous_dsl = True

        prompt = self._build_repair_user_prompt(
            compiler_output=compiler_output,
            previous_dsl=previous_dsl,
            include_previous_dsl=include_previous_dsl,
        )
        self.last_repair_prompt_included_previous_dsl = include_previous_dsl

        if self.repair_stateless:
            # Stateless repair: re-send only (repair shots + current prompt).
            # Do NOT append prior failures to history.
            current_contents = (self.repair_chat_history or []) + [
                types.Content(role="user", parts=[types.Part(text=prompt)])
            ]
            response = self.client.models.generate_content(
                model=self.repair_model_name,
                contents=current_contents,
                config=types.GenerateContentConfig(
                    system_instruction=self.repair_system_prompt,
                    temperature=self.repair_temperature,
                    max_output_tokens=8192,
                ),
            )
        else:
            # Stateful repair (legacy / optional)
            if self.repair_chat is not None:
                response = self.repair_chat.send_message(prompt)
            else:
                current_history = (self.repair_chat_history or []) + [
                    types.Content(role="user", parts=[types.Part(text=prompt)])
                ]
                response = self.client.models.generate_content(
                    model=self.repair_model_name,
                    contents=current_history,
                    config=types.GenerateContentConfig(
                        system_instruction=self.repair_system_prompt,
                        temperature=self.repair_temperature,
                        max_output_tokens=8192,
                    ),
                )

            # Only in stateful mode do we accumulate the conversation history.
            repair_text = response.text or ""
            self.repair_chat_history.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
            self.repair_chat_history.append(types.Content(role="model", parts=[types.Part(text=repair_text)]))

        repair_text = response.text or ""
        self.repair_iteration_count += 1

        # Telemetry: record repair prompt/response sizes (no sanitization)
        self._record_llm_call("repair", prompt, repair_text)
        return repair_text
    
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
        # Legacy compatibility shim: route refinement through the dedicated repair flow so
        # it works with both server-side and stateless modes.
        repair_model_name = getattr(self, "repair_model_name", None) or getattr(self, "model_name", None)
        if not repair_model_name:
            raise RuntimeError("No active model. Start a conversation first.")

        # refine_with_error historically had no shots configuration; default to 0.
        return self.repair_with_compiler_output(
            previous_dsl=self.last_dsl_code or "",
            compiler_output=error_message or "",
            repair_model_name=repair_model_name,
            repair_shots=0,
        )
    
    def save_result(self, dsl_code: str, iteration: int = 0, success: bool = False) -> Path:
        """
        Save generated DSL code to results directory
        
        Args:
            dsl_code: The generated DSL code
            iteration: The iteration number (0 for initial, 1+ for refinements)
            success: Whether this version compiled successfully
        """
        if self.run_dsl_dir is None:
            return Path()
        
        # Create filename
        status = "SUCCESS" if success else f"ITER{iteration}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{status}_{timestamp}.LIRAs"
        
        # Save DSL code
        filepath = self.run_dsl_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(dsl_code or "")

        return filepath
    
    def run_automated_session(self, config: dict):
        """Run an automated session with hardcoded configuration"""
        shot_pairs = self._normalize_shots(config.get('shots'))

        # Optional: configure which repair prompt to use for refinement iterations
        self.configure_repair_prompt(config.get("repair_prompt"))

        # Resolve compiler JAR path (relative paths are relative to project root)
        compiler_jar_path = Path(config["compiler_jar"])
        if not compiler_jar_path.is_absolute():
            compiler_jar_path = (self.base_path / compiler_jar_path)

        max_iterations = int(config["max_iterations"])
        if max_iterations < 1:
            max_iterations = 1

        # Initialize per-run metadata and output directory
        self._init_run_metadata(config, compiler_jar_path, max_iterations)

        # Fail fast if the compiler JAR is missing, but still record the interruption.
        if not compiler_jar_path.exists():
            if self.run_metadata is not None:
                self.run_metadata["status"] = "setup_error"
                self.run_metadata["interrupted"] = True
                self.run_metadata["breaking_error"] = {
                    "type": "compiler_jar_missing",
                    "message": f"compiler_jar path does not exist: {compiler_jar_path}",
                    "when": datetime.now().isoformat(),
                    "where": "run_automated_session/preflight",
                }
                self.run_metadata["run_finished_at"] = datetime.now().isoformat()
                self._persist_run_metadata()
            print(f"\n❌ Error: compiler_jar path does not exist: {compiler_jar_path}")
            return
        
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

            # Automated compile/repair loop
            iteration = 0
            while iteration < max_iterations:
                # Save the raw DSL output immediately for debugging/repro
                saved_dsl_path = self.save_result(dsl_code, iteration=iteration, success=False)

                # Validate via local compiler
                is_valid, feedback = self.validate_code(saved_dsl_path, compiler_jar_path)

                # Track approximate “progress” signal from compiler output.
                score = self._score_compiler_output(feedback)
                self.compiler_error_score_history.append(score)

                # Persist compiler output to a sidecar file for research/debugging
                if self.run_compiler_dir is None:
                    compiler_output_path = saved_dsl_path.with_suffix(saved_dsl_path.suffix + ".compiler.txt")
                else:
                    compiler_output_path = self.run_compiler_dir / (saved_dsl_path.name + ".compiler.txt")
                try:
                    with open(compiler_output_path, "w", encoding="utf-8") as f:
                        f.write(feedback or "")
                except Exception as e:
                    # Avoid noisy stdout in batch mode.
                    pass

                # If validation can't run (missing Java/JAR/timeout), stop rather than prompting LLM.
                if feedback.startswith("[VALIDATION_SETUP_ERROR]"):
                    print("\n❌ Validation setup error; stopping without attempting LLM repair.")
                    print(feedback)

                    if self.run_metadata is not None:
                        self.run_metadata["status"] = "setup_error"
                        self.run_metadata["interrupted"] = True
                        self.run_metadata["breaking_error"] = {
                            "type": "validation_setup_error",
                            "message": "Validation could not be performed (see compiler feedback)",
                            "when": datetime.now().isoformat(),
                            "where": "run_automated_session/validate_code",
                            "iteration": int(iteration),
                            "feedback_preview": (feedback or "")[:5000],
                        }
                        self.run_metadata["run_finished_at"] = datetime.now().isoformat()
                        self.run_metadata.setdefault("iterations", []).append(
                            {
                                "iteration": iteration,
                                "dsl_path": str(saved_dsl_path),
                                "compiler_output_path": str(compiler_output_path),
                                "validated": False,
                                "is_valid": False,
                                "validation": self.last_validation,
                                "compiler_error_score": score,
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
                                "compiler_error_score": score,
                            }
                        )
                        self._persist_run_metadata()
                    break

                # Guardrail: if the compiler produced no error output at all, treat as success.
                # This helps when the compiler signals success without a clean exit code.
                no_error_output = not (feedback or "").strip()
                if no_error_output:
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
                                "compiler_error_score": score,
                                "ended_because": "no_compiler_output",
                            }
                        )
                        self._persist_run_metadata()
                    break

                if iteration + 1 >= max_iterations:
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
                                "compiler_error_score": score,
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
                            "compiler_error_score": score,
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

                iteration += 1
        
        except Exception as e:
            print(f"\n❌ Error during session: {e}")
            # Persist interruption details so downstream collectors can skip the run.
            if self.run_metadata is not None:
                try:
                    import traceback

                    current_iteration = None
                    try:
                        current_iteration = int(locals().get("iteration"))
                    except Exception:
                        current_iteration = None

                    self.run_metadata["status"] = "crashed"
                    self.run_metadata["interrupted"] = True
                    self.run_metadata["breaking_error"] = {
                        "type": type(e).__name__,
                        "message": str(e),
                        "when": datetime.now().isoformat(),
                        "where": "run_automated_session/exception",
                        "iteration": current_iteration,
                        "traceback": traceback.format_exc()[-20000:],
                    }
                    self.run_metadata["run_finished_at"] = datetime.now().isoformat()
                    self._persist_run_metadata()
                except Exception:
                    pass


def main():
    """Main entry point"""
    # Load configuration from config.json
    config_file = Path(__file__).parent / "config.json"
    if not config_file.exists():
        print(f"\n❌ Error: config.json file not found: {config_file}")
        return
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
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

    # Optional decoding parameters (experimental variables)
    generation_temperature = config.get("generation_temperature", 1.0)
    repair_temperature = config.get("repair_temperature", 0.2)
    for name, value in [("generation_temperature", generation_temperature), ("repair_temperature", repair_temperature)]:
        if not isinstance(value, (int, float)):
            print(f"\n❌ Error: '{name}' must be a number in config.json")
            return
        if float(value) < 0.0:
            print(f"\n❌ Error: '{name}' must be >= 0.0")
            return

    # Optional: Vertex AI location/region (model availability can be region-dependent)
    location = config.get("location", "global")
    if not isinstance(location, str) or not location.strip():
        print(f"\n❌ Error: 'location' must be a non-empty string when provided in config.json")
        return
    location = location.strip()
    
    # Validate shots type
    if not isinstance(config["shots"], (int, list)):
        print(f"\n❌ Error: 'shots' must be an integer or a list in config.json")
        return

    # Validate optional repair_shots type
    if "repair_shots" in config and config["repair_shots"] is not None:
        if not isinstance(config["repair_shots"], (int, list)):
            print(f"\n❌ Error: 'repair_shots' must be an integer or a list in config.json")
            return
    
    # Non-interactive authentication + project resolution.
    # Priority order:
    # - project_id: config.json -> key.json -> env
    # - credentials: config.json -> key.json -> ADC
    service_account_key = None
    project_id = None

    cfg_project_id = config.get("project_id")
    if isinstance(cfg_project_id, str) and cfg_project_id.strip():
        project_id = cfg_project_id.strip()

    cfg_key_path = (
        config.get("service_account_key_path")
        or config.get("service_account_key")
        or config.get("credentials_path")
    )
    if isinstance(cfg_key_path, str) and cfg_key_path.strip():
        candidate = Path(cfg_key_path).expanduser()
        if not candidate.is_absolute():
            candidate = (Path(__file__).parent / candidate)
        if not candidate.exists():
            print(f"\n❌ Error: service account key file not found: {candidate}")
            return
        service_account_key = str(candidate)

    # If no explicit key path provided, use detected key.json if present
    if service_account_key is None and KEY_PATH.exists():
        service_account_key = str(KEY_PATH)

    # If we have a key file, attempt to pull project_id from it (unless already set)
    if service_account_key is not None:
        try:
            with open(service_account_key, "r") as f:
                key_data = json.load(f)
            if not project_id:
                project_id = key_data.get("project_id")
        except Exception as e:
            # Keep going; project_id might be provided via config/env, and ADC may still work.
            pass

    # If still no project_id, fall back to environment variables
    if not project_id:
        for env_key in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "GOOGLE_PROJECT_ID"):
            env_val = os.environ.get(env_key)
            if env_val and env_val.strip():
                project_id = env_val.strip()
                break

    if not project_id:
        print(
            "\n❌ Error: Google Cloud Project ID not found. "
            "Set 'project_id' in config.json or export GOOGLE_CLOUD_PROJECT (or include project_id in key.json)."
        )
        return

    if service_account_key is None:
        # Using ADC (gcloud / workload identity).
        pass
    
    # Initialize generator
    generator = DSLGenerator(
        project_id,
        location=location,
        service_account_key=service_account_key,
        generation_temperature=float(generation_temperature),
        repair_temperature=float(repair_temperature),
    )
    
    # Run automated session with configuration from config.json
    generator.run_automated_session(config)


if __name__ == "__main__":
    main()
