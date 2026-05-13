'''
This acts as the Tool Assembly Factory for the MDCrow agent. 
It defines the available capabilities and provides a mechanism to selectively load tools based on the user's research needs.
'''
import os
import numpy as np
from dotenv import load_dotenv
from langchain import agents
from langchain.base_language import BaseLanguageModel
from langchain_openai import OpenAIEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from mdcrow.utils import PathRegistry
# Importing specific tool classes from the base_tools module
from .base_tools import (
    CleaningToolFunction, ComputeAcylindricity, ComputeAngles,
    ComputeAsphericity, ComputeDSSP, ComputeGyrationTensor,
    ComputeLPRMSD, ComputeRelativeShapeAntisotropy, ComputeRMSD,
    ComputeRMSF, ContactsTool, DistanceMatrixTool, GetActiveSites,
    GetAllKnownSites, GetAllSequences, GetBindingSites, GetGeneNames,
    GetInteractions, GetKineticProperties, GetPDB3DInfo,
    GetPDBProcessingInfo, GetProteinAssociatedKeywords, GetProteinFunction,
    GetRelevantSites, GetSequenceInfo, GetSubunitStructure,
    GetTurnsBetaSheetsHelices, GetUniprotID, HydrogenBondTool,
    ListRegistryPaths, MapProteinRepresentation, ModifyBaseSimulationScriptTool,
    MomentOfInertia, PackMolTool, PCATool, PPIDistance,
    ProteinName2PDBTool, RadiusofGyrationTool, RDFTool,
    SaltBridgeTool, Scholar2ResultLLM, SetUpandRunFunction,
    SimulationOutputFigures, SmallMolPDB, SolventAccessibleSurfaceArea,
    SummarizeProteinStructure, UniprotID2Name, VisualizeProtein,
)

def make_all_tools(
    llm: BaseLanguageModel,
    human=False,
    modifysim_no_run=False,
):
    """Initializes and returns the complete list of available tools."""
    load_dotenv()
    all_tools = []
    # Get the singleton instance of PathRegistry for shared file context
    path_instance = PathRegistry.get_instance() 
    
    if llm:
        # Load standard LangChain math tools for calculations
        all_tools += agents.load_tools(["llm-math"], llm)
        # Add the core simulation modification tool (Independent Variable control)
        all_tools += [
            ModifyBaseSimulationScriptTool(
                path_registry=path_instance, llm=llm, modifysim_no_run=modifysim_no_run
            ),
        ]
        # Enable literature search if paper directory is set in registry
        if path_instance.ckpt_papers:
            all_tools += [Scholar2ResultLLM(llm=llm, path_registry=path_instance)]
        # Optionally add a tool for human-in-the-loop intervention
        if human:
            all_tools += [agents.load_tools(["human"], llm)[0]]

    # Define the primary molecular dynamics analysis and utility tools
    base_tools = [
        SummarizeProteinStructure(path_registry=path_instance),
        ComputeAcylindricity(path_registry=path_instance),
        ComputeAsphericity(path_registry=path_instance),
        ComputeDSSP(path_registry=path_instance),
        ComputeGyrationTensor(path_registry=path_instance),
        ComputeRelativeShapeAntisotropy(path_registry=path_instance),
        ComputeAngles(path_registry=path_instance),
        CleaningToolFunction(path_registry=path_instance),
        ComputeLPRMSD(path_registry=path_instance),
        ComputeRMSD(path_registry=path_instance),
        ComputeRMSF(path_registry=path_instance),
        ContactsTool(path_registry=path_instance),
        DistanceMatrixTool(path_registry=path_instance),
        HydrogenBondTool(path_registry=path_instance),
        ListRegistryPaths(path_registry=path_instance),
        MomentOfInertia(path_registry=path_instance),
        PackMolTool(path_registry=path_instance),
        PCATool(path_registry=path_instance),
        PPIDistance(path_registry=path_instance),
        ProteinName2PDBTool(path_registry=path_instance),
        RadiusofGyrationTool(path_registry=path_instance),
        RDFTool(path_registry=path_instance),
        SaltBridgeTool(path_registry=path_instance),
        SetUpandRunFunction(path_registry=path_instance),
        SimulationOutputFigures(path_registry=path_instance),
        SmallMolPDB(path_registry=path_instance),
        SolventAccessibleSurfaceArea(path_registry=path_instance),
        VisualizeProtein(path_registry=path_instance),
        MapProteinRepresentation(),
        UniprotID2Name(),
        GetBindingSites(),
        GetActiveSites(),
        GetRelevantSites(),
        GetAllKnownSites(),
        GetProteinFunction(),
        GetProteinAssociatedKeywords(),
        GetAllSequences(),
        GetInteractions(),
        GetSubunitStructure(),
        GetSequenceInfo(),
        GetPDBProcessingInfo(),
        GetPDB3DInfo(),
        GetTurnsBetaSheetsHelices(),
        GetUniprotID(),
        GetGeneNames(),
        GetKineticProperties(),
    ]

    all_tools += base_tools
    return all_tools

def get_relevant_tools(query, llm: BaseLanguageModel, top_k_tools=15, human=False):
    """Uses vector similarity to select the best tools for a specific query."""
    # Generate the master tool list
    all_tools = make_all_tools(llm, human=human)
    if not all_tools:
        return None

    # Combine tool names and descriptions for vectorization
    tool_texts = [f"{tool.name} {tool.description}" for tool in all_tools]

    # Use high-dimensional embeddings if OpenAI key is available
    if "OPENAI_API_KEY" in os.environ:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        try:
            # Vectorize all tools and the user's research query
            tool_vectors = np.array(embeddings.embed_documents(tool_texts))
            query_vector = np.array(embeddings.embed_query(query)).reshape(1, -1)
        except Exception as e:
            print(f"Error generating embeddings for tool retrieval: {e}")
            return None
    else:
        # Fallback to TF-IDF (keyword frequency) if API key is missing
        vectorizer = TfidfVectorizer()
        tool_vectors = vectorizer.fit_transform(tool_texts)
        query_vector = vectorizer.transform([query])

    # Calculate cosine similarity between query and tools
    similarities = cosine_similarity(query_vector, tool_vectors).flatten()
    # Select the top 'k' most relevant tools to reduce LLM context noise
    k = min(max(top_k_tools, 1), len(all_tools))
    if k == 0:
        return None
    # Sort tools by similarity score in descending order
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    retrieved_tools = [all_tools[i] for i in top_k_indices]

    return retrieved_tools
