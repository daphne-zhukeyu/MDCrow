# This module provides utility functions for loading trajectories and saving analysis outputs.
import os
import warnings
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from .path_registry import FileType, PathRegistry

def load_single_traj(path_registry, top_fileid=None, traj_fileid=None, traj_required=False, ignore_warnings=False):
    """
    Loads a single trajectory using mdtraj by resolving File IDs through the PathRegistry.
    """
    # Validates that the registry is correctly provided to look up file paths.
    if not isinstance(path_registry, PathRegistry):
        raise ValueError("path_registry must be an instance of PathRegistry.")
    
    # ... logic to resolve file paths from the registry ...

    # Returns an mdtraj.Trajectory object containing coordinates and topology.
    # md.load() handles the heavy lifting of reading the physical files.
    return md.load(traj_path, top=top_path)

def load_traj_with_ref(path_registry, top_id, traj_id=None, ref_top_id=None, ref_traj_id=None, traj_required=False, ignore_warnings=False):
    """
    Loads a main trajectory and an optional reference trajectory for comparison.
    """
    # Commonly used when calculating RMSD (Root Mean Square Deviation) against a reference structure.
    traj = load_single_traj(path_registry, top_id, traj_id, traj_required, ignore_warnings)
    # ... logic to load reference ...
    return traj, ref_traj

def save_to_csv(path_registry, data_to_save, analysis_name, description=None, header=""):
    """
    Exports numerical results (Dependent Variables) to a CSV and registers the file ID.
    """
    # Automatically handles file naming to avoid overwriting existing results.
    # Registers the new CSV in path_registry so the Agent can find it later.
    np.savetxt(file_path, data_to_save, delimiter=",", header=header)
    path_registry.map_path(file_id, file_path, description=description)
    return file_id

def save_plot(path_registry, fig_analysis, description=None):
    """
    Saves the current matplotlib figure as a PNG and logs it in the registry.
    """
    # Verifies there is an active plot to save; otherwise, raises an error.
    if plt.gcf().get_axes() == []:
        raise ValueError("No plot detected. Failed to save.")
    
    # Generates a standardized filename and saves the image to the figures directory.
    plt.savefig(fig_path)
    path_registry.map_path(fig_id, fig_path, description=description)
    return fig_id
