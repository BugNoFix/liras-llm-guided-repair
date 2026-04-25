import os
import re
import shlex
import subprocess
import time
from google import genai
from google.genai import types
from google.genai.errors import ClientError as GenaiClientError
try:
    from google.api_core.exceptions import ResourceExhausted
except ImportError:
    ResourceExhausted = None
from pathlib import Path
from datetime import datetime
import json
from typing import Optional

# ============================================================================
# dsl_generator_flash.py — Lightweight fork optimized for gemini-2.5-flash
# Differences from dsl_generator.py:
#   - Minimal retry logic (1 retry, 1s delay) instead of exponential backoff
#   - Suited for Flash's higher RPM/TPM quotas
# ============================================================================
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
        repair_max_output_tokens: int = 16384,
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
        self.generative_sp_path = self.sp_path / "Generative"
        self.repair_sp_path = self.sp_path / "Repair"
        self.shots_path = self.base_path / "Shots"
        self.generative_shots_path = self.shots_path / "Generative"
        self.repair_shots_path = self.shots_path / "Repair"
        self.scenarios_path = self.base_path / "Scenarios"
        self.results_path = self.base_path / "Results"
        self.repair_prompt_template_path = self.repair_sp_path / "SystemPromptRepair1.txt"
        if not self.repair_prompt_template_path.exists():
            self.repair_prompt_template_path = self.sp_path / "RepairPrompt.txt"
        self.generated_dsl_root = self.base_path / "GeneratedDSL"
        
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

        # Repair strategy: stateful mode sends shots/system prompt once and keeps a
        # server-side chat session, dramatically reducing per-call token volume.
        # Set to True to fall back to stateless mode if context poisoning is a concern.
        self.repair_stateless = False

        # Repair decoding defaults (favor determinism to reduce regressions).
        self.repair_temperature = float(repair_temperature)
        self.repair_max_output_tokens = int(repair_max_output_tokens)
        self.compiler_timeout = 60

        # Approximate compiler error progress tracking (used as a light “monotonic improvement” signal)
        self.compiler_error_score_history: list[dict] = []

        # Windowed context: track immediate history of attempts and errors
        # to help the repair prompt contrast previous failures.
        self.repair_history_window: list[dict] = []  # Stores {'dsl': str, 'error': str}
        self.max_window_size: int = 3  # Keep last 3 turns — enough to avoid ping-pong regressions

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

    def _call_with_backoff(self, func, *, label: str, max_retries: int = 1):
        """Run a callable with a single fast retry on 429 (tuned for Flash quotas)."""
        _retryable = (GenaiClientError,)
        if ResourceExhausted is not None:
            _retryable = (GenaiClientError, ResourceExhausted)

        for attempt in range(max_retries + 1):
            try:
                return func()
            except _retryable as exc:
                if isinstance(exc, GenaiClientError) and getattr(exc, 'code', None) != 429:
                    raise
                if attempt == max_retries:
                    raise
                print(f"[BACKOFF] {label} hit 429. Waiting 1s before retry...")
                time.sleep(1)

    def _build_shot_history(self, shot_pairs: list[dict], *, shots_base_path: Path) -> list[types.Content]:
        """Build alternating user/model Content messages from configured shot pairs."""
        history: list[types.Content] = []
        if not shot_pairs:
            return history

        for pair in shot_pairs:
            user_content = self.load_file(shots_base_path / pair["user"])
            assistant_content = self.load_file(shots_base_path / pair["assistant"])
            history.append(types.Content(role="user", parts=[types.Part(text=user_content)]))
            history.append(types.Content(role="model", parts=[types.Part(text=assistant_content)]))
        return history

    def _normalize_shots(self, shots, *, start_index: int = 1) -> list[dict]:
        """Normalize shots config into a list of {user, assistant} pairs.

        Supports:
        - int: N -> UserScenario_<start_index>.. and AssistantScenario_<start_index>..
        - list[dict]: already pairs
        - None/0/[]: no shots
        """
        if not shots:
            return []

        if isinstance(shots, int):
            if shots <= 0:
                return []
            shot_pairs = []
            first = int(start_index)
            last = first + shots
            for i in range(first, last):
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

    @staticmethod
    def _strip_extension(name: str) -> str:
        return name.replace(".txt", "") if isinstance(name, str) else str(name)

    @staticmethod
    def _extract_first_int(text: str) -> Optional[int]:
        if not text:
            return None
        match = re.search(r"(\d+)", text)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    def _resolve_generated_dsl_path(self, config: dict) -> Path:
        """Resolve cache path for generated DSL based on scenario, system prompt, and shots."""
        explicit_path = config.get("generated_dsl_path")
        if isinstance(explicit_path, str) and explicit_path.strip():
            candidate = Path(explicit_path).expanduser()
            if not candidate.is_absolute():
                candidate = self.base_path / candidate
            return candidate

        root = config.get("generated_dsl_root")
        if isinstance(root, str) and root.strip():
            base_root = Path(root).expanduser()
            if not base_root.is_absolute():
                base_root = self.base_path / base_root
        else:
            base_root = self.generated_dsl_root

        scenario_id = self._strip_extension(config.get("scenario", "scenario"))
        sp_id = self._strip_extension(config.get("system_prompt", "system_prompt"))
        shots = config.get("shots")
        shots_label = str(shots) if shots is not None else "unknown"

        return base_root / scenario_id / sp_id / f"shots_{shots_label}.LIRAs"

    def _resolve_existing_dsl_path_from_dsl_folder(self, config: dict) -> Path:
        """Resolve an existing DSL path in the DSL/Scenario_X folder structure."""
        dsl_root = config.get("dsl_source_root")
        if isinstance(dsl_root, str) and dsl_root.strip():
            base_root = Path(dsl_root).expanduser()
            if not base_root.is_absolute():
                base_root = self.base_path / base_root
        else:
            base_root = self.base_path / "DSL"

        scenario_name = self._strip_extension(config.get("scenario", ""))
        scenario_num = self._extract_first_int(scenario_name)
        if scenario_num is None:
            raise ValueError(f"Could not extract scenario number from '{scenario_name}'")
        scenario_dir = base_root / f"Scenario_{scenario_num}"

        sp_name = self._strip_extension(config.get("system_prompt", ""))
        sp_num = self._extract_first_int(sp_name)
        if sp_num is None:
            raise ValueError(f"Could not extract system prompt number from '{sp_name}'")

        shots = config.get("shots")
        if not isinstance(shots, int):
            raise ValueError("'shots' must be an integer when loading DSL from DSL folder")

        base_name = f"SP{sp_num}_Shot{shots}"
        exact_path = scenario_dir / f"{base_name}.txt"
        if exact_path.exists():
            return exact_path

        time_candidates = sorted(scenario_dir.glob(f"{base_name}_Time*.txt"))
        if time_candidates:
            return time_candidates[-1]

        raise FileNotFoundError(
            f"DSL file not found for scenario={scenario_dir.name}, system_prompt=SP{sp_num}, "
            f"shots={shots} under {scenario_dir}"
        )

    def _resolve_cached_dsl_path(self, config: dict) -> Path:
        """Resolve which cache source to use for loading DSL."""
        source = (config.get("generated_dsl_source") or "generated_cache").strip().lower()
        if source in ("dsl", "dsl_folder", "runs", "run_folder"):
            return self._resolve_existing_dsl_path_from_dsl_folder(config)
        return self._resolve_generated_dsl_path(config)

    def _save_generated_dsl_cache(self, dsl_code: str, config: dict) -> Optional[Path]:
        """Persist generated DSL to a deterministic cache path."""
        if not dsl_code:
            return None
        cache_path = self._resolve_generated_dsl_path(config)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        content = dsl_code
        if content and not content.endswith("\n"):
            content += "\n"
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(content)
        return cache_path

    def _load_generated_dsl_cache(self, config: dict) -> str:
        """Load generated DSL from cache based on config."""
        cache_path = self._resolve_cached_dsl_path(config)
        if not cache_path.exists():
            raise FileNotFoundError(f"Generated DSL cache not found: {cache_path}")
        return self.load_file(cache_path)

    def _record_llm_call(self, kind: str, prompt_text: str, response_obj) -> None:
        """Record telemetry for an LLM call, including model metadata.

        Args:
            kind: 'generate' or 'repair'
            prompt_text: The prompt string sent to the model
            response_obj: The raw Gemini response object (GenerateContentResponse)
        """
        # Extract metadata from the Gemini response object
        candidate = response_obj.candidates[0] if response_obj.candidates else None
        finish_reason = candidate.finish_reason if candidate else "UNKNOWN"
        safety_ratings = [
            {"category": str(r.category), "probability": str(r.probability)}
            for r in (candidate.safety_ratings if candidate and candidate.safety_ratings else [])
        ]

        response_text = response_obj.text or ""
        prompt_chars = len(prompt_text or "")
        response_chars = len(response_text)
        prompt_tokens_est = self._estimate_tokens(prompt_text or "")
        response_tokens_est = self._estimate_tokens(response_text)

        self.telemetry["llm_calls"] += 1
        self.telemetry["prompt_chars_total"] += prompt_chars
        self.telemetry["response_chars_total"] += response_chars
        self.telemetry["prompt_tokens_est_total"] += prompt_tokens_est
        self.telemetry["response_tokens_est_total"] += response_tokens_est
        self.telemetry["last_call"] = {
            "kind": kind,
            "timestamp": datetime.now().isoformat(),
            "finish_reason": str(finish_reason),
            "safety_ratings": safety_ratings,
            "prompt_chars": prompt_chars,
            "response_chars": response_chars,
            "prompt_tokens_est": prompt_tokens_est,
            "response_tokens_est": response_tokens_est,
        }

        # Log finish_reason to stdout for quick diagnosis
        if response_chars == 0:
            print(f"[WARNING] {kind} response was 0 chars. finish_reason={finish_reason}")

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

        # Allow config to override the output root (e.g. for parallel batch isolation).
        # If the directory exists results are inserted alongside; otherwise it is created.
        cfg_results_dir = config.get("results_dir")
        if cfg_results_dir and str(cfg_results_dir).strip():
            results_base = Path(cfg_results_dir.strip())
            if not results_base.is_absolute():
                results_base = self.base_path / results_base
        else:
            results_base = self.results_path

        # User-friendly hierarchy: every run is fully contained in its own directory.
        # <results_base>/
        #   <Scenario>/<SystemPrompt>/RUN_<run_id>/
        #     dsl/
        #     compiler/
        runs_root = results_base / scenario_name / sp_name
        runs_root.mkdir(parents=True, exist_ok=True)

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = runs_root / f"RUN_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.run_dsl_dir = self.run_dir / "dsl"
        self.run_compiler_dir = self.run_dir / "compiler"
        self.run_dsl_dir.mkdir(parents=True, exist_ok=True)
        self.run_compiler_dir.mkdir(parents=True, exist_ok=True)

        self.run_metadata_path = self.run_dir / "run_metadata.json"
        use_cached_generation = bool(config.get("use_generated_dsl_cache"))
        generated_dsl_source = config.get("generated_dsl_source") or "generated_cache"
        if use_cached_generation:
            try:
                generated_dsl_cache_path = str(self._resolve_cached_dsl_path(config))
            except Exception as e:
                generated_dsl_cache_path = f"<unresolved: {type(e).__name__}>"
        else:
            generated_dsl_cache_path = str(self._resolve_generated_dsl_path(config))

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
            "repair_max_output_tokens": int(getattr(self, "repair_max_output_tokens", 16384)),
            "repair_shots": config.get("repair_shots"),
            "use_generated_dsl_cache": use_cached_generation,
            "generated_dsl_source": generated_dsl_source,
            "generated_dsl_cache_path": generated_dsl_cache_path,
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
        sp_files.extend([f"Generative/{p.name}" for p in sorted(self.generative_sp_path.glob("*.txt"))])
        generative_shots = [p.name for p in sorted(self.generative_shots_path.glob("*.txt"))]
        repair_shots = [p.name for p in sorted(self.repair_shots_path.glob("*.txt"))]
        scenario_files = [p.name for p in sorted(self.scenarios_path.glob("*.txt"))]
        return {
            "system_prompts": sp_files,
            "shots": generative_shots,
            "generative_shots": generative_shots,
            "repair_shots": repair_shots,
            "scenarios": scenario_files,
        }

    def _resolve_system_prompt_path(self, system_prompt_file: str) -> Path:
        """Resolve generation system prompt path across legacy and nested SP layouts."""
        candidate = Path(system_prompt_file).expanduser()
        candidates = []

        if candidate.is_absolute():
            candidates.append(candidate)
        else:
            candidates.extend(
                [
                    self.base_path / candidate,
                    self.sp_path / candidate,
                    self.generative_sp_path / candidate,
                ]
            )
            if candidate.parent == Path("."):
                candidates.append(self.generative_sp_path / candidate.name)

        for path in candidates:
            if path.exists() and path.is_file():
                return path

        searched = "\n - ".join(str(p) for p in candidates)
        raise FileNotFoundError(
            f"System prompt not found: {system_prompt_file}\n"
            f"Searched:\n - {searched}"
        )

    def _default_repair_prompt_path(self) -> Path:
        preferred = [
            self.repair_sp_path / "SystemPromptRepair1.txt",
            self.sp_path / "RepairPrompt.txt",
        ]
        for path in preferred:
            if path.exists() and path.is_file():
                return path
        return preferred[0]
    
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
        system_prompt_path = self._resolve_system_prompt_path(system_prompt_file)
        system_prompt = self.load_file(system_prompt_path)

        # Capture generation context for the repair phase
        self.generation_system_prompt_text = system_prompt
        
        # Store model name for chat
        self.model_name = model_name
        
        # Build chat history from few-shot examples
        history = self._build_shot_history(shot_pairs, shots_base_path=self.generative_shots_path)
        
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

        print(f"[GENERATE] Sending scenario to generation model ({model_name})...")
        if self.generation_chat is not None:
            response = self._call_with_backoff(
                lambda: self.generation_chat.send_message(scenario_content),
                label="generation_chat.send_message",
            )
        else:
            current_history = self.chat_history + [
                types.Content(role="user", parts=[types.Part(text=scenario_content)])
            ]
            response = self._call_with_backoff(
                lambda: self.client.models.generate_content(
                    model=model_name,
                    contents=current_history,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=self.generation_temperature,
                        safety_settings=[
                            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                        ],
                    ),
                ),
                label="generation.generate_content",
            )
        
        response_text = response.text or ""
        print(f"[GENERATE] Response received ({len(response_text)} chars)")
        if not response_text.strip():
            print("[WARNING] Generation model returned an empty response")

        # Update chat history with user message and response
        self.chat_history.append(types.Content(role="user", parts=[types.Part(text=scenario_content)]))
        self.chat_history.append(types.Content(role="model", parts=[types.Part(text=response_text)]))

        # Telemetry: record prompt/response sizes + model metadata
        self._record_llm_call("generate", scenario_content, response)

        return response_text

    def _build_repair_user_prompt(
        self,
        compiler_output: str,
        previous_dsl: Optional[str] = None,
        include_previous_dsl: bool = False,
    ) -> str:
        """Build a repair user prompt with delta reasoning.

        Anchors the model with the original generation while highlighting the
        specific recent failure so the model can reason about what NOT to repeat.
        """
        parts: list[str] = []

        parts.append("### REPAIR TASK")
        parts.append(
            "The DSL below failed to compile. Compare it to the "
            "COMPILER_OUTPUT to identify why and fix it.\n"
        )

        # The most recent failed attempt (the DSL the model must fix)
        if include_previous_dsl and previous_dsl:
            parts.append("### PREVIOUS_FAILED_ATTEMPT")
            parts.append(previous_dsl)
            parts.append("")

        # Current compiler errors (truncated to reduce noise)
        parts.append("### CURRENT_COMPILER_OUTPUT")
        parts.append(self._truncate_compiler_output(compiler_output or ""))
        parts.append("")

        # Instruction block
        parts.append(
            "### INSTRUCTION\n"
            "1. Fix the FIRST error reported. Do not repeat the same edit used in PREVIOUS_FAILED_ATTEMPT.\n"
            "2. If the error is 'unresolved reference', check the Pattern block at the top.\n"
            "3. Output ONLY the full corrected LIRAs text."
        )

        return "\n".join(parts)

    @staticmethod
    def _truncate_compiler_output(compiler_output: str, max_errors: int = 5) -> str:
        """Keep only the first `max_errors` unique [ERROR] lines plus any header.

        The Xtext compiler often emits cascading duplicates (e.g., 18 errors
        from one root cause).  Sending all of them wastes tokens and dilutes
        the model's focus.  We keep the first few unique error lines which
        almost always contain the root cause.
        """
        if not compiler_output:
            return ""
        lines = compiler_output.splitlines()
        kept: list[str] = []
        error_count = 0
        seen: set[str] = set()
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("[ERROR]"):
                # Deduplicate identical error messages
                if stripped in seen:
                    continue
                seen.add(stripped)
                error_count += 1
                if error_count > max_errors:
                    kept.append(f"... ({len([l for l in lines if l.strip().startswith('[ERROR]')])} total errors, showing first {max_errors})")
                    break
            kept.append(line)
        return "\n".join(kept)

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
        """Return the repair system prompt.

        No placeholders remain after the context trimming — the prompt is used as-is.
        This method is kept for backward compatibility and future extensibility.
        """
        return template_text

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

        shot_pairs = self._normalize_shots(repair_shots, start_index=3)
        history = self._build_shot_history(shot_pairs, shots_base_path=self.repair_shots_path)

        self.repair_chat_history = history
        self._repair_shot_message_count = len(history)  # preserve count for sliding window pruning
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
        """Run one repair iteration using the dedicated repair chat.

        Includes stagnation detection: if the repair output is identical to the
        previous DSL, the method retries once at temperature 0.7 to force a
        different structural interpretation.
        """
        self._ensure_repair_chat(repair_model_name=repair_model_name, repair_shots=repair_shots)

        # Always include the current DSL + current compiler output.
        include_previous_dsl = True

        prompt = self._build_repair_user_prompt(
            compiler_output=compiler_output,
            previous_dsl=previous_dsl,
            include_previous_dsl=include_previous_dsl,
        )
        self.last_repair_prompt_included_previous_dsl = include_previous_dsl

        # Use a local temperature variable to allow dynamic bumping on stagnation.
        current_temp = self.repair_temperature
        repair_text = ""

        for attempt in range(2):  # Try twice if the first fix is a duplicate
            print(f"[REPAIR] Sending repair request to {self.repair_model_name} (temp={current_temp}, attempt={attempt})...")

            if self.repair_stateless:
                current_contents = (self.repair_chat_history or []) + [
                    types.Content(role="user", parts=[types.Part(text=prompt)])
                ]
                response = self._call_with_backoff(
                    lambda: self.client.models.generate_content(
                        model=self.repair_model_name,
                        contents=current_contents,
                        config=types.GenerateContentConfig(
                            system_instruction=self.repair_system_prompt,
                            temperature=current_temp,
                            max_output_tokens=self.repair_max_output_tokens,
                            response_mime_type="text/plain",
                            safety_settings=[
                                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                            ],
                        ),
                    ),
                    label="repair.generate_content",
                )
            else:
                if self.repair_chat is not None:
                    response = self._call_with_backoff(
                        lambda: self.repair_chat.send_message(prompt),
                        label="repair_chat.send_message",
                    )
                else:
                    current_history = (self.repair_chat_history or []) + [
                        types.Content(role="user", parts=[types.Part(text=prompt)])
                    ]
                    response = self._call_with_backoff(
                        lambda: self.client.models.generate_content(
                            model=self.repair_model_name,
                            contents=current_history,
                            config=types.GenerateContentConfig(
                                system_instruction=self.repair_system_prompt,
                                temperature=current_temp,
                                max_output_tokens=self.repair_max_output_tokens,
                                response_mime_type="text/plain",
                                safety_settings=[
                                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                                ],
                            ),
                        ),
                        label="repair.generate_content_stateful",
                    )

            repair_text = (response.text or "").strip()

            # Stagnation detection: if the output is identical to the input DSL,
            # jump to a high temperature to break the deterministic loop.
            if repair_text == (previous_dsl or "").strip() and attempt == 0:
                print(f"[REPAIR] Stagnation detected -- output identical to input. Bumping temp {current_temp} -> 0.7")
                current_temp = 0.7
                continue
            break

        # Only in stateful mode do we accumulate the conversation history.
        if not self.repair_stateless:
            self.repair_chat_history.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
            self.repair_chat_history.append(types.Content(role="model", parts=[types.Part(text=repair_text)]))

            # Sliding window: keep initial shot messages + only the last N repair turns
            # to prevent context poisoning from accumulated failed attempts.
            shot_count = getattr(self, '_repair_shot_message_count', 0)
            max_repair_turns = self.max_window_size  # each turn = 2 messages (user + model)
            shot_messages = self.repair_chat_history[:shot_count]
            repair_messages = self.repair_chat_history[shot_count:]
            max_repair_messages = max_repair_turns * 2
            if len(repair_messages) > max_repair_messages:
                self.repair_chat_history = shot_messages + repair_messages[-max_repair_messages:]
                print(f"[REPAIR] Pruned stateful history: kept {shot_count} shot msgs + {max_repair_messages} repair msgs")

        self.repair_iteration_count += 1
        print(f"[REPAIR] Response received ({len(repair_text)} chars)")
        if not repair_text.strip():
            print("[WARNING] Repair model returned an empty response")

        # Update the windowed history for downstream analysis.
        self.repair_history_window.append({"dsl": repair_text, "error": compiler_output or ""})
        if len(self.repair_history_window) > self.max_window_size:
            self.repair_history_window = self.repair_history_window[-self.max_window_size:]

        # Telemetry: record repair prompt/response sizes + model metadata
        self._record_llm_call("repair", prompt, response)
        return repair_text
    
    def _extract_dsl_code(self, response_text: str) -> str:
        """Clean model output by stripping markdown fences and conversational preamble.

        Args:
            response_text: Raw response from the AI

        Returns:
            Clean DSL code without markdown or explanations
        """
        if not response_text:
            return ""

        # Strip markdown fences if the model hallucinated them
        text = response_text.replace("```liras", "").replace("```dsl", "").replace("```xtext", "").replace("```", "").strip()

        # Remove conversational preamble lines (e.g., "Here is the fixed DSL:")
        preamble_words = ["here", "fixed", "repair", "corrected", "updated", "below"]
        lines = text.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not any(word in stripped.lower() for word in preamble_words):
                return "\n".join(lines[i:]).strip()

        return text

    def validate_code(self, dsl_filepath: Path, compiler_jar_path: Path, compiler_timeout: int = None) -> tuple[bool, str]:
        """Validate DSL code by running the local Java-based compiler.

        Runs:
            java -jar <compiler_jar_path> <dsl_filepath>

        Args:
            dsl_filepath: Path to the DSL file to validate
            compiler_jar_path: Path to the compiler JAR
            compiler_timeout: Timeout in seconds (default: 60, or config value)

        Returns:
            (is_valid, feedback)
            - is_valid is True when exit code == 0
            - feedback is combined stdout+stderr for logging/repair
        """
        dsl_filepath = Path(dsl_filepath)
        compiler_jar_path = Path(compiler_jar_path)

        # Use provided timeout, fall back to instance config, then to 60s default
        if compiler_timeout is None:
            compiler_timeout = getattr(self, 'compiler_timeout', 60)

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
                timeout=compiler_timeout,
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
                f"[VALIDATION_SETUP_ERROR] Compiler invocation timed out after {compiler_timeout} seconds.\n"
                f"Command: {shlex.join(cmd)}\n"
                f"DSL file: {dsl_filepath}\n"
                f"Possible fixes: (1) Increase 'compiler_timeout' in config.json, "
                f"(2) Optimize DSL complexity, (3) Check system resources"
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
            self.repair_prompt_template_path = self._default_repair_prompt_path()
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
        if rel_to_sps.exists():
            self.repair_prompt_template_path = rel_to_sps
            return

        rel_to_repair = self.repair_sp_path / repair_prompt
        if rel_to_repair.exists():
            self.repair_prompt_template_path = rel_to_repair
            return

        if candidate.parent == Path("."):
            named_in_repair = self.repair_sp_path / candidate.name
            if named_in_repair.exists():
                self.repair_prompt_template_path = named_in_repair
                return

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
    
    def _cleanup_resources(self) -> None:
        """Clean up resources to prevent accumulation during batch execution."""
        try:
            # Close chat sessions
            self.chat = None
            self.generation_chat = None
            self.repair_chat = None
            self.chat_history = []
            self.repair_chat_history = None
            
            # Clear large data structures
            self.compiler_error_score_history.clear()
            self.repair_history_window.clear()
            
            # Reset state
            self.last_dsl_code = None
            self.generation_system_prompt_text = None
            self.generation_scenario_text = None
            self.initial_dsl_from_generation = None
        except Exception:
            # Silently ignore cleanup errors
            pass

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
        
        # Save DSL code (ensure trailing newline for compiler friendliness)
        filepath = self.run_dsl_dir / filename
        content = dsl_code or ""
        if content and not content.endswith("\n"):
            content += "\n"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        return filepath
    
    def run_automated_session(self, config: dict):
        """Run an automated session with hardcoded configuration"""
        shot_pairs = self._normalize_shots(config.get('shots'))
        generation_only = bool(config.get("generation_only"))

        # Optional: configure which repair prompt to use for refinement iterations
        self.configure_repair_prompt(config.get("repair_prompt"))

        # Allow config to override the default repair stateless/stateful mode
        if "repair_stateless" in config:
            self.repair_stateless = bool(config["repair_stateless"])

        # Resolve compiler JAR path (relative paths are relative to project root)
        compiler_jar_raw = config.get("compiler_jar")
        compiler_jar_path = Path(compiler_jar_raw) if compiler_jar_raw else Path("")
        if compiler_jar_raw and not compiler_jar_path.is_absolute():
            compiler_jar_path = (self.base_path / compiler_jar_path)

        max_iterations = int(config.get("max_iterations", 1))
        if max_iterations < 1:
            max_iterations = 1

        compiler_timeout_raw = config.get("compiler_timeout", self.compiler_timeout)
        try:
            compiler_timeout = int(compiler_timeout_raw)
        except (TypeError, ValueError):
            compiler_timeout = int(self.compiler_timeout)
        if compiler_timeout <= 0:
            compiler_timeout = int(self.compiler_timeout)

        use_cached_generation = bool(config.get("use_generated_dsl_cache"))
        dsl_source_mode = "cache" if use_cached_generation else "generation"
        dsl_source_detail = "n/a"
        if use_cached_generation:
            try:
                dsl_source_detail = str(self._resolve_cached_dsl_path(config))
            except Exception as e:
                dsl_source_detail = f"<unresolved: {type(e).__name__}>"

        repair_shots_cfg = config.get("repair_shots")
        repair_shot_pairs = self._normalize_shots(repair_shots_cfg, start_index=3)
        repair_shot_count = len(repair_shot_pairs)

        # Initialize per-run metadata and output directory
        self._init_run_metadata(config, compiler_jar_path, max_iterations)

        print(
            "[START] "
            f"scenario={config.get('scenario')} "
            f"sp={config.get('system_prompt')} "
            f"gen_model={config.get('generation_model')} "
            f"gen_shots={config.get('shots')} "
            f"source={dsl_source_mode} "
            f"max_iter={max_iterations} timeout={compiler_timeout}s"
        )
        print(
            "[REPAIR_CFG] "
            f"model={config.get('repair_model')} "
            f"prompt={self.repair_prompt_template_path.name} "
            f"strategy={'stateless' if self.repair_stateless else 'stateful'} "
            f"temp={self.repair_temperature} max_tokens={self.repair_max_output_tokens} "
            f"jshots_cfg={repair_shots_cfg} jshots_loaded={repair_shot_count}"
        )

        # Fail fast if the compiler JAR is missing, but still record the interruption.
        if not generation_only and not compiler_jar_path.exists():
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
            print(f"\n[ERROR] compiler_jar path does not exist: {compiler_jar_path}")
            return
        
        # Start conversation and get initial DSL code
        try:
            if use_cached_generation:
                print(f"[GENERATE] Skipping generation; loading DSL from cache: {dsl_source_detail}")
                dsl_code = self._load_generated_dsl_cache(config)
            else:
                response_text = self.start_conversation(
                    config['system_prompt'],
                    config['shots'],
                    config['scenario'],
                    model_name=config['generation_model'],
                )
                # Clean model output: strip any markdown fences or conversational preamble
                dsl_code = self._extract_dsl_code(response_text)
                self._save_generated_dsl_cache(dsl_code, config)

            self.last_dsl_code = dsl_code
            self.initial_dsl_from_generation = dsl_code

            if generation_only:
                print("[GENERATE] Generation-only mode; skipping compile/repair.")
                saved_dsl_path = self.save_result(dsl_code, iteration=0, success=False)
                if self.run_metadata is not None:
                    self.run_metadata["status"] = "generation_only"
                    self.run_metadata["run_finished_at"] = datetime.now().isoformat()
                    self.run_metadata.setdefault("iterations", []).append(
                        {
                            "iteration": 0,
                            "dsl_path": str(saved_dsl_path),
                            "validated": False,
                            "is_valid": False,
                            "ended_because": "generation_only",
                        }
                    )
                    self._persist_run_metadata()
                return

            # Automated compile/repair loop
            iteration = 0
            while iteration < max_iterations:
                print(f"\n--- Iteration {iteration}/{max_iterations - 1} ---")

                # Save the raw DSL output immediately for debugging/repro
                saved_dsl_path = self.save_result(dsl_code, iteration=iteration, success=False)

                # Validate via local compiler
                print("[COMPILE] Validating DSL...")
                is_valid, feedback = self.validate_code(saved_dsl_path, compiler_jar_path, compiler_timeout=compiler_timeout)

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
                    print("\n[ERROR] Validation setup error; stopping without attempting LLM repair.")
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
                    print("[OK] Compilation successful")
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
                    print("[OK] No compiler errors (treating as success)")
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

                print(f"[FAIL] Compilation failed -- {score.get('error_lines', '?')} error(s), {score.get('warning_lines', '?')} warning(s)")

                if iteration + 1 >= max_iterations:
                    print("[STOP] Max iterations reached")
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
                print(f"[REPAIR] Requesting LLM repair (iteration {iteration} -> {iteration + 1})...")
                repair_result = self.repair_with_compiler_output(
                    previous_dsl=dsl_code,
                    compiler_output=feedback,
                    repair_model_name=config['repair_model'],
                    repair_shots=config.get('repair_shots'),
                )

                # Guard: do NOT overwrite dsl_code with an empty repair response.
                # Keep the previous DSL so the next iteration can retry from a valid state.
                if not repair_result or not repair_result.strip():
                    print("[WARNING] Repair returned empty -- keeping previous DSL for next iteration")
                    if self.run_metadata is not None:
                        try:
                            last_it = self.run_metadata.get("iterations", [])[-1]
                            last_it["repair_returned_empty"] = True
                            self._persist_run_metadata()
                        except Exception:
                            pass
                else:
                    dsl_code = self._extract_dsl_code(repair_result)

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
            print(f"\n[ERROR] Error during session: {e}")
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
        finally:
            # Clean up resources to prevent accumulation during batch execution
            self._cleanup_resources()


def main():
    """Main entry point"""
    # Load configuration from config.json
    config_file = Path(__file__).parent / "config.json"
    if not config_file.exists():
        print(f"\n[ERROR] config.json file not found: {config_file}")
        return
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"\n[ERROR] Error reading config.json: {e}")
        return
    
    # Validate configuration
    generation_only = bool(config.get("generation_only"))
    required_keys = [
        "system_prompt",
        "generation_model",
        "shots",
        "scenario",
    ]
    if not generation_only:
        required_keys.extend(["repair_model", "compiler_jar", "max_iterations"])
    for key in required_keys:
        if key not in config:
            print(f"\n[ERROR] Missing required key '{key}' in config.json")
            return

    # Validate max_iterations type when needed
    if not generation_only:
        if not isinstance(config.get("max_iterations"), int):
            print(f"\n[ERROR] 'max_iterations' must be an integer in config.json")
            return

    # Validate model keys
    if not isinstance(config["generation_model"], str) or not config["generation_model"].strip():
        print(f"\n[ERROR] 'generation_model' must be a non-empty string in config.json")
        return
    if not generation_only:
        if not isinstance(config["repair_model"], str) or not config["repair_model"].strip():
            print(f"\n[ERROR] 'repair_model' must be a non-empty string in config.json")
            return

    # Optional decoding parameters (experimental variables)
    generation_temperature = config.get("generation_temperature", 1.0)
    repair_temperature = config.get("repair_temperature", 0.2)
    for name, value in [("generation_temperature", generation_temperature), ("repair_temperature", repair_temperature)]:
        if not isinstance(value, (int, float)):
            print(f"\n[ERROR] '{name}' must be a number in config.json")
            return
        if float(value) < 0.0:
            print(f"\n[ERROR] '{name}' must be >= 0.0")
            return

    compiler_timeout = config.get("compiler_timeout", 60)
    if not isinstance(compiler_timeout, (int, float)):
        print("\n[ERROR] 'compiler_timeout' must be a number in config.json")
        return
    if float(compiler_timeout) <= 0:
        print("\n[ERROR] 'compiler_timeout' must be > 0 in config.json")
        return

    # Optional: Vertex AI location/region (model availability can be region-dependent)
    location = config.get("location", "global")
    if not isinstance(location, str) or not location.strip():
        print(f"\n[ERROR] 'location' must be a non-empty string when provided in config.json")
        return
    location = location.strip()
    
    # Validate shots type
    if not isinstance(config["shots"], (int, list)):
        print(f"\n[ERROR] 'shots' must be an integer or a list in config.json")
        return

    # Validate optional repair_shots type
    if not generation_only:
        if "repair_shots" in config and config["repair_shots"] is not None:
            if not isinstance(config["repair_shots"], (int, list)):
                print(f"\n[ERROR] 'repair_shots' must be an integer or a list in config.json")
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
            print(f"\n[ERROR] Service account key file not found: {candidate}")
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
            "\n[ERROR] Google Cloud Project ID not found. "
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
        repair_max_output_tokens=int(config.get("repair_max_output_tokens", 16384)),
    )
    
    # Run automated session with configuration from config.json
    generator.run_automated_session(config)


if __name__ == "__main__":
    main()
