class MemoryManager:
    def __init__(self, path_registry: PathRegistry, llm, run_id=""):
        # Links to the central PathRegistry for storage locations
        self.path_registry = path_registry
        self.dir_name = f"{path_registry.ckpt_memory}"
        
        # If no Run ID is provided, generate a new 8-character random ID
        self.run_id = run_id if run_id else self.new_run_id()
        
        # Chains the summary template with the LLM and a string parser
        self.llm_agent_trace = agent_summary_template | llm | StrOutputParser()

        self._make_all_dirs()
        # If continuing an existing run, pull the previous summary from memory
        self.run_id_mem = self.pull_agent_summary_from_mem(self.run_id) if run_id else None

    def new_run_id(self) -> str:
        """Generates a random 8-character ID (e.g., 'A1B2C3D4')."""
        length = 8
        characters = string.ascii_uppercase + string.digits
        return "".join(random.choice(characters) for _ in range(length))

    def generate_agent_summary(self, agent_trace):
        """
        Uses the LLM to summarize the trace and saves it to a JSON file.
        Key format: {RUN_ID}.{SUMMARY_NUMBER} (e.g., 'A1B2C3D4.0')
        """
        # Invoke the LLM chain to process the raw agent logs
        llm_out = self.llm_agent_trace.invoke({"agent_trace": agent_trace})
        
        # Determine the version number based on how many summaries already exist
        key_str = f"{self.run_id}.{self.get_summary_number()}"
        run_summary = {key_str: llm_out}
        
        # Persist the summary to the central 'agent_run_summaries.json'
        self._write_to_json(run_summary, self.agent_trace_summary)

    def pull_agent_summary_from_mem(self, run_id: str, run_num: int = 0):
        """Retrieves a specific historical summary to provide context for the current LLM prompt."""
        run_id_full = f"{run_id}.{run_num}"
        if not os.path.exists(self.agent_trace_summary):
            return "Path does not exist."
            
        with open(self.agent_trace_summary, "r") as f:
            summary = json.load(f)
            
        # Returns the text summary if found, otherwise an error message
        return summary.get(run_id_full, f"Run ID not found. Keys are: {list(summary.keys())}")
