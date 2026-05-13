#This defines utility tools that allow the agent to interact with the PathRegistry to manage file mappings.
# defining tools
from typing import Optional

from langchain.tools import BaseTool

from mdcrow.utils import PathRegistry


class MapPath2Name(BaseTool):
    """Tool for associating a human-readable name with a physical file path."""
    name = "MapPath2Name"
    # Detailed instructions for the LLM on how to format inputs and interpret success
    description = """Input the desired filename
    followed by the file's path, separated by a comma.
    Make sure the name is first, then the path.
    Your path should look something like: name.pdb
    Stores the path in the registry with the
    name provided in the filename.
    If the output says Path mapped to name,
    then it was successful.
    You do not need to check that file was created."""
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry]):
        # Initialize the tool with a reference to the central PathRegistry
        super().__init__()
        self.path_registry = path_registry

    def _run(self, file_and_path: str) -> str:
        """Execute the mapping logic synchronously."""
        try:
            # Check for registry initialization to prevent NoneType errors
            if self.path_registry is None:
                return "Failed. Path registry not initialized"
            # Ensure the input string contains the required delimiter
            if "," not in file_and_path:
                return "Failed. Please separate filename and path with a comma"
            # Split input into components: name (ID) and the actual file path
            file, path = file_and_path.split(",")
            # Register the mapping in the shared singleton registry
            map_name = self.path_registry.map_path(file, path)
            return "Succeeded. " + map_name
        except Exception:
            # Generic error handling for filesystem or registry write issues
            return "Failed. Error writing paths to file"

    async def _arun(self, file_name: str) -> str:
        """Asynchronous execution is currently not supported."""
        raise NotImplementedError


class ListRegistryPaths(BaseTool):
    """Tool for the agent to query all currently registered files and their purposes."""
    name = "ListRegistryPaths"
    # Instructions for the agent to retrieve its 'spatial awareness' of files
    description = """Use this tool to list all paths saved in memory.
    Input the word 'paths' and the tool will return a list of all names
    in the registry that are mapped to paths."""
    path_registry: Optional[PathRegistry]

    def __init__(self, path_registry: Optional[PathRegistry]):
        # Connect to the central path management system
        super().__init__()
        self.path_registry = path_registry

    def _run(self, paths: str) -> str:
        """Execute the directory listing logic."""
        try:
            # Validation check for the registry instance
            if self.path_registry is None:
                return "Failed. Path registry not initialized"
            # Fetches a formatted string of IDs, paths, and descriptions (descriptions)
            return "Succeeded. " + self.path_registry.list_path_names_and_descriptions()
        except Exception:
            # Error handling for retrieval failures
            return "Failed. Error listing paths"

    async def _arun(self, paths: str) -> str:
        """Asynchronous execution is currently not supported."""
        raise NotImplementedError
