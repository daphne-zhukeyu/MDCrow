1. Core Architecture: The "Function-Tool" Pattern
The system follows a hierarchical design built on the LangChain framework, dividing responsibilities into two layers:

Logic Layer (Functions): Classes like SASAFunctions and VisFunctions handle raw computations using scientific libraries (e.g., mdtraj, numpy) and external process calls.

Interface Layer (Tools): Classes inheriting from BaseTool (e.g., SolventAccessibleSurfaceArea, VisualizeProtein) wrap the logic for AI Agent execution, defining names and descriptions for LLM discovery.

2. Key Components & Files
Analytical Tools:

ComputeDSSP: Assigns secondary structure (Helix, Strand, Coil).

SolventAccessibleSurfaceArea: Computes SASA using the Shrake-Rupley algorithm.

ComputeAsphericity / ComputeAcylindricity: Measures molecular shape descriptors.

Visualization Tools:

VisualizeProtein: Implements a dual-pathway for rendering (PNG via molrender or interactive .ipynb via nglview).

Infrastructure (Registry):

PathRegistry: Acts as the central hub for mapping unique File IDs to physical paths, ensuring data consistency across different analysis steps.

3. Workflow Summary
Input: The PathRegistry resolves file IDs into paths.

Processing: mdtraj performs trajectory analysis or structural calculations.

Output: Raw data is saved as .npy/.csv, and visuals are saved as .png or notebooks, all registered back to the PathRegistry.
MDCrow is an LLM-agent based toolset for Molecular Dynamics.
It's built using Langchain and uses a collection of tools to set up and execute molecular dynamics simulations, particularly in OpenMM.


## Environment Setup
To use the OpenMM features in the agent, please set up a conda environment following these steps.
```
conda env create -n mdcrow -f environment.yaml
conda activate mdcrow
```

If you already have a conda environment, you can install dependencies before you activate it with the following steps.
- Install the necessary conda dependencies: `conda env update -n <YOUR_CONDA_ENV_HERE> -f environment.yaml`



## Installation
```
pip install git+https://github.com/ur-whitelab/MDCrow.git
```

## Usage
The next step is to set up your API keys in your environment. An API key for an LLM provider is necessary for this project. Supported LLM providers are OpenAI, TogetherAI, Fireworks, and Anthropic.
We recommend setting up the API keys in a .env file. You can use the provided .env.example file as a template.
1. Copy the `.env.example` file and rename it to `.env`: `cp .env.example .env`
2. Replace the placeholder values in `.env` with your actual keys

You can ask MDCrow to conduct molecular dynamics tasks using OpenAI's GPT model.
```
from mdcrow import MDCrow

agent = MDCrow(model="gpt-3.5-turbo")
agent.run("Simulate protein 1ZNI at 300 K for 0.1 ps and calculate the RMSD over time.")
```
Note: To distinguish Together models from the rest, you'll need to add the "together\" prefix in the model flag, such as `agent = MDCrow(model="together/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")`

## LLM Providers
By default, we support LLMs through OpenAI API. However, feel free to use other LLM providers. Make sure to install the necessary package for it. Here's a list of packages required for alternative LLM providers we support:
- `pip install langchain-together` to use models from TogetherAI
- `pip install langchain-anthropic` to use models from Anthropic
- `pip install langchain-fireworks` to use models from Fireworks


## Contributing

We value and appreciate all contributions to MDCrow.
