# This module acts as a factory to instantiate Large Language Model (LLM) objects.
import importlib.util
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def check_package_exists(package_name, model):
    """
    Ensures the specific integration library for a provider is installed before use.
    """
    # Uses importlib to check if the required library is available in the current environment.
    if not importlib.util.find_spec(package_name):
        raise ImportError(
            f"The package required to run model '{model}' is missing: '{package_name}'."
        )

def _make_llm(model, temp, streaming):
    """
    Creates and returns a LangChain LLM object based on the provided model name and parameters.
    """
    # Logic for OpenAI: Checks for 'gpt' prefixes and initializes ChatOpenAI.
    if model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
        from langchain_openai import ChatOpenAI
        # temp (Temperature) acts as a control for randomness in the model's output.
        llm = ChatOpenAI(
            temperature=temp,
            model_name=model,
            request_timeout=1000,
            streaming=streaming,
            # StreamingStdOutCallbackHandler displays the Agent's reasoning process in real-time.
            callbacks=[StreamingStdOutCallbackHandler()] if streaming else None,
        )
    
    # Logic for Fireworks AI: Validates 'langchain_fireworks' exists before importing.
    elif model.startswith("accounts/fireworks"):
        check_package_exists("langchain_fireworks", model)
        from langchain_fireworks import ChatFireworks
        llm = ChatFireworks(
            temperature=temp,
            model_name=model,
            request_timeout=1000,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()] if streaming else None,
        )

    # Logic for Together AI: Requires a 'together/' prefix to distinguish the provider.
    elif model.startswith("together/"):
        check_package_exists("langchain_together", model)
        from langchain_together import ChatTogether
        llm = ChatTogether(
            temperature=temp,
            model=model.replace("together/", ""), # Removes prefix for the API call.
            request_timeout=1000,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()] if streaming else None,
        )

    # Logic for Anthropic (Claude): Validates 'langchain_anthropic' installation.
    elif model.startswith("claude"):
        check_package_exists("langchain_anthropic", model)
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            temperature=temp,
            model_name=model,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()] if streaming else None,
        )

    # Default case: Raises an error if the model string does not match any supported provider.
    else:
        raise ValueError(f"Unrecognized or unsupported model name: {model}")
        
    return llm
