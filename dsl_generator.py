import os
import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession, Content, Part
from pathlib import Path
from datetime import datetime
import json


class DSLGenerator:
    def __init__(self, project_id: str, location: str = "us-central1", service_account_key: str = None):
        """
        Initialize the DSL Generator with Vertex AI credentials
        
        Args:
            project_id: Google Cloud Project ID
            location: Region for Vertex AI (default: us-central1)
            service_account_key: Path to service account JSON key file (optional)
        """
        # Set credentials if service account key is provided
        if service_account_key:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_key
            print(f"✓ Using service account: {service_account_key}")
        
        vertexai.init(project=project_id, location=location)
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
        self.current_config = {}
        self.last_dsl_code = None
        
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
    
    def start_conversation(self, system_prompt_file: str, shots, scenario_file: str):
        """
        Start a new conversation with configured system prompt, shot pairs, and scenario
        
        Args:
            system_prompt_file: Name of the system prompt file (e.g., "SystemPrompt1.txt")
            shots: Integer (number of shots, e.g., 2 loads UserScenario_1.txt + AssistantScenario_1.txt,
                          UserScenario_2.txt + AssistantScenario_2.txt) OR
                   List of dicts with 'user' and 'assistant' shot file names for backwards compatibility
            scenario_file: Name of the scenario file to process (e.g., "UserScenario_011.txt")
        """
        # Convert integer shots to list of shot pairs
        if isinstance(shots, int):
            shot_pairs = []
            for i in range(1, shots + 1):
                shot_pairs.append({
                    "user": f"UserScenario_{i}.txt",
                    "assistant": f"AssistantScenario_{i}.txt"
                })
        else:
            shot_pairs = shots if shots else []
        
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
        
        # Create model with system instruction
        self.model = GenerativeModel(
            "gemini-2.0-flash-lite-001",
            system_instruction=system_prompt
        )
        
        # Build chat history from few-shot examples using Content objects
        history = []
        if shot_pairs:
            for pair in shot_pairs:
                # Load user scenario example
                user_content = self.load_file(self.shots_path / pair["user"])
                # Load assistant DSL code example
                assistant_content = self.load_file(self.shots_path / pair["assistant"])
                
                # Add as Content objects
                history.append(Content(role="user", parts=[Part.from_text(user_content)]))
                history.append(Content(role="model", parts=[Part.from_text(assistant_content)]))
        
        # Start chat with manufactured history
        self.chat = self.model.start_chat(history=history)
        
        # Load the actual scenario to process
        scenario_content = self.load_file(self.scenarios_path / scenario_file)
        
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
        response = self.chat.send_message(scenario_content)
        
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
                text = content.parts[0].text
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

    def configure_repair_prompt(self, repair_prompt: str | None):
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

        if template:
            refinement_prompt = template.format(
                previous_dsl=self.last_dsl_code or "",
                compiler_output=error_message or "",
            )
        else:
            refinement_prompt = f"""The previous DSL code produced the following compilation error:

ERROR:
{error_message}

Please analyze the error and generate a corrected version of the DSL code that fixes this issue.
Provide ONLY the corrected DSL code without any explanations or markdown formatting.
"""
        
        print("\n=== Sending error feedback to Gemini ===")
        print(f"Error:\n{error_message}")
        print("\n" + "="*50 + "\n")
        
        response = self.chat.send_message(refinement_prompt)
        return response.text
    
    def save_result(self, dsl_code: str, iteration: int = 0, success: bool = False):
        """
        Save generated DSL code to results directory
        
        Args:
            dsl_code: The generated DSL code
            iteration: The iteration number (0 for initial, 1+ for refinements)
            success: Whether this version compiled successfully
        """
        if not self.current_config:
            print("Warning: No active configuration to save")
            return
        
        # Create result directory structure
        sp_name = self.current_config["system_prompt"].replace(".txt", "")
        scenario_name = self.current_config["scenario"].replace(".txt", "")
        
        result_dir = self.results_path / scenario_name / sp_name
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        status = "SUCCESS" if success else f"ITER{iteration}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{status}_{timestamp}.txt"
        
        # Save DSL code
        filepath = result_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(dsl_code)
        
        # Save metadata
        metadata = {
            **self.current_config,
            "iteration": iteration,
            "success": success,
            "saved_at": datetime.now().isoformat()
        }
        metadata_file = result_dir / f"{status}_{timestamp}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Saved result to: {filepath}")
    
    def run_automated_session(self, config: dict):
        """Run an automated session with hardcoded configuration"""
        print("\n" + "="*60)
        print("DSL Generator - Automated Mode")
        print("="*60)
        
        # Convert integer shots to list of shot pairs if needed
        shots = config['shots']
        if isinstance(shots, int):
            shot_pairs = []
            for i in range(1, shots + 1):
                shot_pairs.append({
                    "user": f"UserScenario_{i}.txt",
                    "assistant": f"AssistantScenario_{i}.txt"
                })
        else:
            shot_pairs = shots if shots else []
        
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
        print("="*60 + "\n")

        # Optional: configure which repair prompt to use for refinement iterations
        self.configure_repair_prompt(config.get("repair_prompt"))
        
        # Start conversation and get initial DSL code
        try:
            response_text = self.start_conversation(
                config['system_prompt'],
                config['shots'],
                config['scenario']
            )
            dsl_code = response_text
            self.last_dsl_code = dsl_code
            print("\n=== GENERATED DSL CODE ===")
            print(dsl_code)
            print("\n" + "="*60 + "\n")
            
            # Save initial result
            self.save_result(dsl_code, iteration=0)
            
            # Refinement loop
            iteration = 1
            while True:
                print("\n" + "="*60)
                print("Paste compilation error below (or type 'success' if compiled successfully, 'exit' to quit):")
                print("="*60)
                
                # Read multiline input until user presses Ctrl+Z (Windows) or Ctrl+D (Unix) then Enter
                lines = []
                try:
                    while True:
                        line = input()
                        if line.lower() in ['success', 'exit']:
                            user_input = line.lower()
                            break
                        lines.append(line)
                except EOFError:
                    user_input = '\n'.join(lines)
                
                if not lines and 'user_input' in locals():
                    # Single-line command
                    if user_input == 'success':
                        self.save_result(dsl_code, iteration=iteration, success=True)
                        print("\n✓ Session completed successfully!")
                        break
                    elif user_input == 'exit':
                        print("\nExiting session.")
                        break
                else:
                    # Multiline error message
                    error_msg = '\n'.join(lines) if lines else user_input
                    
                    if not error_msg.strip():
                        print("No error message provided. Please try again.")
                        continue
                    
                    dsl_code = self.refine_with_error(error_msg)
                    self.last_dsl_code = dsl_code
                    print("\n=== REFINED DSL CODE ===")
                    print(dsl_code)
                    print("\n" + "="*60 + "\n")
                    
                    self.save_result(dsl_code, iteration=iteration)
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
  "shots": 2,
  "scenario": "UserScenario_011.txt"
}

Or for custom shot pairs:
{
  "system_prompt": "SystemPrompt1.txt",
  "shots": [
    {
      "user": "UserScenario_1.txt",
      "assistant": "AssistantScenario_1.txt"
    }
  ],
  "scenario": "UserScenario_011.txt"
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
    required_keys = ["system_prompt", "shots", "scenario"]
    for key in required_keys:
        if key not in config:
            print(f"\n❌ Error: Missing required key '{key}' in config.json")
            return
    
    # Validate shots type
    if not isinstance(config["shots"], (int, list)):
        print(f"\n❌ Error: 'shots' must be an integer or a list in config.json")
        return
    
    # Check for key.json in current directory
    key_file = Path(__file__).parent / "key.json"
    service_account_key = None
    project_id = None
    
    if key_file.exists():
        print("✓ Found key.json file")
        try:
            with open(key_file, 'r') as f:
                key_data = json.load(f)
                project_id = key_data.get("project_id")
                service_account_key = str(key_file)
                print(f"✓ Using service account: {key_data.get('client_email')}")
                print(f"✓ Project ID: {project_id}")
        except Exception as e:
            print(f"Warning: Could not read key.json: {e}")
            key_file = None
    
    # If no key.json found, ask user for authentication method
    if not key_file.exists() or not project_id:
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
