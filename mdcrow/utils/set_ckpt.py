# This module manages the physical creation and cleanup of experiment workspace directories.
import os
import shutil

class SetCheckpoint:
    def find_root_dir(self):
        """
        Traverses upward from the current directory to find the project root identified by 'setup.py'.
        """
        # This ensures that file paths remain consistent regardless of where the script is launched.
        current_dir = os.getcwd()
        while current_dir != "/":
            if "setup.py" in os.listdir(current_dir):
                return os.path.abspath(current_dir)
            else:
                current_dir = os.path.dirname(current_dir)
        return None

    def make_ckpt_parent_folder(self, ckpt_dir: str = "ckpt"):
        """
        Creates the main 'ckpt/' directory at the project root if it doesn't already exist.
        """
        root = self.find_root_dir()
        if not root:
            raise ValueError("Root directory not found.")
        ckpt_path = os.path.join(root, ckpt_dir)
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        return ckpt_path

    def set_ckpt_subdir(self, ckpt_dir: str = "ckpt", ckpt_parent_folder: str = "ckpt"):
        """
        Establishes a specific subdirectory for a single run, using versioning (e.g., ckpt_0, ckpt_1).
        """
        # If a specific name is provided, it uses that; otherwise, it increments a counter.
        ckpt_parent_path = self.make_ckpt_parent_folder(ckpt_parent_folder)
        
        # Iterates until it finds a folder name that does not yet exist.
        ckpt_iter = 0
        ckpt_subdir = os.path.join(ckpt_parent_path, f"{ckpt_dir}_{ckpt_iter}")
        while os.path.exists(ckpt_subdir):
            ckpt_iter += 1
            ckpt_subdir = os.path.join(ckpt_parent_path, f"{ckpt_dir}_{ckpt_iter}")
            
        # Creates the final path for the PathRegistry to use.
        os.makedirs(ckpt_subdir, exist_ok=True)
        return ckpt_subdir

    def clear_ckpt(self, ckpt_dir: str = "ckpt", parent_ckpt_path: str = "ckpt", keep_root: bool = False):
        """
        Deletes a specific checkpoint folder to free up space or reset an experiment.
        """
        parent_ckpt_path = self.make_ckpt_parent_folder(parent_ckpt_path)
        ckpt_path = os.path.join(parent_ckpt_path, ckpt_dir)
        if os.path.exists(ckpt_path):
            shutil.rmtree(ckpt_path) # Recursively deletes the directory and all its files.
            if keep_root:
                os.makedirs(ckpt_path)

    def clear_all_ckpts(self, parent_ckpt_path: str = "ckpt", keep_root: bool = False):
        """
        Wipes the entire checkpoint directory, removing all history of previous runs.
        """
        parent_ckpt_path = self.make_ckpt_parent_folder(parent_ckpt_path)
        shutil.rmtree(parent_ckpt_path)
        if keep_root:
            self.make_ckpt_parent_folder(parent_ckpt_path)
