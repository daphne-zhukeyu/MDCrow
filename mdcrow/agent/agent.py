import os
from datetime import datetime
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.agents.structured_chat.base import StructuredChatAgent
from ..tools import get_relevant_tools, make_all_tools
from ..utils import PathRegistry, SetCheckpoint, _make_llm
from .memory import MemoryManager
from .prompt import openaifxn_prompt, structured_prompt

load_dotenv() # Load environment variables like API keys

class AgentType:
    """Registry to manage different LangChain agent architectures."""
    valid_models = {
        "Structured": StructuredChatAgent,
        "OpenAIFunctionsAgent": OpenAIFunctionsAgent,
    }

    @classmethod
    def get_agent(cls, model_name: str = "OpenAIFunctionsAgent"):
        # Returns the specific Agent class based on the requested type
        try:
            agent = cls.valid_models[model_name]
            return agent
        except KeyError:
            raise ValueError(f"Invalid agent type: {model_name}")

class MDCrow:
    def __init__(
        self,
        tools=None,
        agent_type="OpenAIFunctionsAgent",
        model="gpt-4-1106-preview", # Uses GPT-4 Turbo by default
        tools_model=None,
        temp=0.1, # Low temperature for more deterministic/factual output
        streaming=True,
        verbose=False,
        ckpt_dir="ckpt", # Default directory for checkpoints
        top_k_tools=20, # Limit tools sent to LLM to save context/cost
        use_human_tool=False,
        uploaded_files=[],
        run_id="",
        use_memory=False,
        modifysim_no_run=False,
        paper_dir=None,
    ):
        # Initialize the main LLM and a specialized LLM for tool selection
        self.llm = _make_llm(model, temp, streaming)
        if tools_model is None:
            tools_model = model
        self.tools_llm = _make_llm(tools_model, temp, streaming)

        self.use_memory = use_memory
        # Singleton registry ensures all components find the same file paths
        self.path_registry = PathRegistry.get_instance(ckpt_dir, paper_dir)
        self.ckpt_dir = self.path_registry.ckpt_dir
        # MemoryManager tracks historical context and agent summaries
        self.memory = MemoryManager(self.path_registry, self.tools_llm, run_id=run_id)
        self.run_id = self.memory.run_id
        self.uploaded_files = uploaded_files
        self.agent = None
        self.agent_type = agent_type
        self.top_k_tools = top_k_tools
        self.use_human_tool = use_human_tool
        self.user_tools = tools
        self.verbose = verbose

        # Register any user-provided files into the Path Registry
        if self.uploaded_files:
            self.add_file(self.uploaded_files)
        self.modifysim_no_run = modifysim_no_run

    def _add_single_file(self, file_path, description=None):
        # Generates a unique ID (UPL_timestamp) for every uploaded file
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        i = 0
        ID = "UPL_" + str(i) + timestamp
        while ID in self.path_registry.list_path_names():
            i += 1
            ID = "UPL_" + str(i) + timestamp
        if not description:
            description = "User uploaded file"
        # Maps the human-readable ID to the physical disk path
        self.path_registry.map_path(ID, file_path, description=description)

    def add_file(self, uploaded_files):
        # Recursive handler for single files, lists, or (path, description) tuples
        if isinstance(uploaded_files, str):
            self._add_single_file(uploaded_files)
        elif isinstance(uploaded_files, tuple):
            self._add_single_file(uploaded_files[0], description=uploaded_files[1])
        elif isinstance(uploaded_files, list):
            for file_path in uploaded_files:
                self.add_file(file_path)
        else:
            raise ValueError("Invalid input format for files.")

    def _initialize_tools_and_agent(self, user_input=None):
        """Dynamic tool discovery based on user query."""
        if self.user_tools is not None:
            self.tools = self.user_tools
        else:
            # Filter tools to find only relevant ones (Independent Variables/自变量)
            if self.top_k_tools != "all" and user_input is not None:
                self.tools = get_relevant_tools(
                    query=user_input,
                    llm=self.tools_llm,
                    top_k_tools=self.top_k_tools,
                    human=self.use_human_tool,
                )
            else:
                self.tools = make_all_tools(
                    self.tools_llm,
                    human=self.use_human_tool,
                    modifysim_no_run=self.modifysim_no_run,
                )
        # Create the execution engine that links the LLM and the tools
        return AgentExecutor.from_agent_and_tools(
            tools=self.tools,
            agent=AgentType.get_agent(self.agent_type).from_llm_and_tools(
                self.llm,
                self.tools,
            ),
            verbose=self.verbose,
            handle_parsing_errors=True,
        )

    def run(self, user_input, callbacks=None):
        # Entry point for execution; incorporates memory context into the prompt
        run_memory = self.memory.run_id_mem if self.use_memory else None
        if self.agent_type == "Structured":
            self.prompt = structured_prompt.format(input=user_input, context=run_memory)
        elif self.agent_type == "OpenAIFunctionsAgent":
            self.prompt = openaifxn_prompt.format(input=user_input, context=run_memory)
        
        self.agent = self._initialize_tools_and_agent(user_input)
        model_output = self.agent.invoke(self.prompt, callbacks=callbacks)
        
        # After execution, summarize the run and store it for future steps
        if self.use_memory:
            self.memory.generate_agent_summary(model_output)
        return model_output, self.run_id

    def iter(self, user_input, include_run_info=True):
        # Generator that streams the agent's step-by-step reasoning process
        run_memory = self.memory.run_id_mem if self.use_memory else None
        if self.agent is None:
            self.agent = self._initialize_tools_and_agent(user_input)
        
        # Formatting specific prompts for iterative execution
        if self.agent_type == "Structured":
            self.prompt = structured_prompt.format(input=user_input, context=run_memory)
        elif self.agent_type == "OpenAIFunctionsAgent":
            self.prompt = openaifxn_prompt.format(input=user_input, context=run_memory)
            
        for step in self.agent.iter(self.prompt, include_run_info=include_run_info):
            yield step

    def force_clear_mem(self, all=False) -> str:
        # Safety mechanism to physically delete checkpoint directories
        if all:
            ckpt_dir = os.path.abspath(os.path.dirname(self.path_registry.ckpt_dir))
        else:
            ckpt_dir = self.path_registry.ckpt_dir
        
        confirmation = input("Confirm clearing memory/checkpoints? (yes/no): ")
        if confirmation.lower() == "yes":
            set_ckpt = SetCheckpoint()
            set_ckpt.clear_all_ckpts(ckpt_dir)
            return "All checkpoints removed."
        return "Action canceled."
