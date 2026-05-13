#This defines a specialized tool that uses the PaperQA library to enable the agent to perform academic literature searches and Retrieval-Augmented Generation (RAG) over a local directory of PDF papers.
from typing import Optional

import nest_asyncio
import paperqa
from langchain.base_language import BaseLanguageModel
from langchain.tools import BaseTool

from mdcrow.utils import PathRegistry


def scholar2result_llm(llm, query, path_registry):
    """Orchestrates the PaperQA library to search local PDFs and generate an answer."""
    # Retrieve the directory containing PDFs from the central path management system
    paper_directory = path_registry.ckpt_papers 
    
    # Validation: Ensure the user actually provided a source for the literature
    if paper_directory is None:
        raise ValueError(
            "'paper_dir' is None. To use this tool, the user "
            "must provide a directory with PDFs at the start."
        )
    
    # Configure PaperQA settings based on the provided LLM type
    llm_name = llm.model_name
    if llm_name.startswith("gpt") or llm_name.startswith("claude"):
        settings = paperqa.Settings(
            llm=llm_name, # Use the specific model (e.g., GPT-4) for the full pipeline
            summary_llm=llm_name, # Use the same model for summarizing individual paper chunks
            temperature=llm.temperature,
            paper_directory=paper_directory,
        )
    else:
        # Fallback to default models if the provided LLM is not directly compatible with PaperQA settings
        settings = paperqa.Settings(
            temperature=llm.temperature,
            paper_directory=paper_directory,
        )
    
    # Execute the search and RAG process (Retrieval-Augmented Generation)
    response = paperqa.ask(query, settings=settings)
    answer = response.answer.formatted_answer
    
    # Debugging/User hint: If no answer found, suggest checking the directory contents
    if "I cannot answer." in answer:
        answer += f" Check to ensure there's papers in {paper_directory}"
    return answer


class Scholar2ResultLLM(BaseTool):
    """LangChain-compatible tool interface for the literature search function."""
    name = "LiteratureSearch"
    description = (
        "Useful to answer questions that may be found in literature. "
        "Ask a specific question as the input."
    )
    llm: BaseLanguageModel = None
    path_registry: Optional[PathRegistry]

    def __init__(self, llm, path_registry):
        # Establish the connection between the Agent's brain (LLM) and its file storage
        super().__init__()
        self.llm = llm
        self.path_registry = path_registry

    def _run(self, query) -> str:
        """Synchronously executes the search; includes fix for nested event loops."""
        # Necessary because PaperQA uses async calls which can conflict with the agent's loop
        nest_asyncio.apply()
        try:
            return scholar2result_llm(self.llm, query, self.path_registry)
        except Exception as e:
            # Return technical error context back to the agent for potential self-correction
            return f"Failed. {type(e).__name__}: {e}"

    async def _arun(self, query) -> str:
        """Asynchronous execution is currently not supported."""
        raise NotImplementedError("this tool does not support async")
