# This module implements a persistent registry to map File IDs to actual disk paths.
import json
import os
from datetime import datetime
from enum import Enum
from typing import Optional
from mdcrow.utils.set_ckpt import SetCheckpoint

# Defines standardized categories for files handled by the Agent.
class FileType(Enum):
    PROTEIN = 1      # e.g., PDB files
    SIMULATION = 2   # e.g., Python simulation scripts
    RECORD = 3       # e.g., CSV data results
    FIGURE = 4       # e.g., Generated plots (PNG)
    UNKNOWN = 5

class PathRegistry:
    """
    A Singleton class that manages directory structures and tracks file metadata in a JSON registry.
    """
    instance = None
    set_ckpt = SetCheckpoint()

    @classmethod
    def get_instance(cls, ckpt_dir=None, paper_dir=None):
        # Implementation of the Singleton pattern to ensure only one registry exists per run.
        if not cls.instance or ckpt_dir is not None:
            cls.instance = cls(ckpt_dir, paper_dir)
        return cls.instance

    def __init__(self, ckpt_dir: str = "ckpt", paper_dir=None):
        # Initializes the checkpoint (ckpt) subdirectory and creates necessary folders.
        self._set_ckpt(ckpt_dir)
        self._set_paper_dir(paper_dir)
        self._make_all_dirs()      # Creates subfolders like /pdb, /records, /figures, /simulations.
        self._init_path_registry() # Loads or creates the 'paths_registry.json' index file.

    def _make_all_dirs(self):
        # Systematically creates the directory structure for organizing simulation outputs.
        self.json_file_path = os.path.join(self.ckpt_dir, "paths_registry.json")
        # ... logic to create /pdb, /records, /simulations, etc. ...

    def map_path(self, file_id, path, description=None):
        """
        The primary 'write' method: saves a file path and its description to the JSON registry.
        """
        # Converts relative paths to absolute paths to ensure the Agent can find files from any directory.
        full_path = self._get_full_path(path)
        path_dict = {
            file_id: {"path": full_path, "name": os.path.basename(full_path), "description": description}
        }
        self._save_mapping_to_json(path_dict)
        return f"Path mapped to name: {file_id}"

    def get_mapped_path(self, fileid):
        """
        The primary 'read' method: retrieves the absolute disk path for a given File ID.
        """
        # Allows the Agent to access files by simple IDs (e.g., 'sim0_123456') rather than long paths.
        with open(self.json_file_path, "r") as json_file:
            data = json.load(json_file)
            return data.get(fileid, {}).get("path", "Name not found.")

    def get_fileid(self, file_name: str, type: FileType):
        """
        Generates a unique, searchable ID for a file based on its type and a timestamp.
        """
        # Ensures that IDs (e.g., PROTEIN -> PDBID_timestamp) remain unique and organized.
        # ... logic to extract 6-digit timestamp suffixes ...

    def write_file_name(self, type: FileType, **kwargs):
        """
        A complex naming engine that generates standardized filenames for all MDCrow outputs.
        """
        # Dynamically constructs names like 'FIG_analysis_SimID_timestamp.png' based on provided metadata.
        # This acts as the 'grammar' for how MDCrow names its files.
