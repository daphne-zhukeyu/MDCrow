# This module acts as a factory to instantiate Large Language Model (LLM) objects.
import importlib.util
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def check_package_exists(package_name, model):
    """
    Ensures the specific integration library for a provider is installed before use.
    """
    # Uses importlib to check if a package is available in the current environment.
    if not importlib.util.find_spec(package_name):
        raise ImportError(
            f"The package required to run model '{model}' is missing: '{package_name}'."
        )

def _make_llm(model, temp, streaming):
    """
    Creates and returns a LangChain LLM object based on the provided model name.
    """
    # Logic for OpenAI models (e.g., GPT-4).
    if model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
        from langchain_openai import ChatOpenAI
        # Returns a configured ChatOpenAI instance with specified temperature and streaming.
        llm = ChatOpenAI(
            temperature=temp,
            model_name=model,
            request_timeout=1000,
            streaming=streaming,
            # StreamingStdOutCallbackHandler allows the user to see the Agent's "thoughts" in real-time.
            callbacks=[StreamingStdOutCallbackHandler()] if streaming else None,
        )
    
    # Logic for Fireworks AI models.
    elif model.startswith("accounts/fireworks"):
        check_package_exists("langchain_fireworks", model)
        from langchain_fireworks import ChatFireworks
        llm = ChatFireworks(...)

    # Logic for Together AI models (requires 'together/' prefix).
    elif model.startswith("together/"):
        check_package_exists("langchain_together", model)
        from langchain_together import ChatTogether
        llm = ChatTogether(...)

    # Logic for Anthropic models (e.g., Claude 3).
    elif model.startswith("claude"):
        check_package_exists("langchain_anthropic", model)
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(...)

    else:
        # Error handling for unsupported or mistyped model names.
        raise ValueError(f"Unrecognized or unsupported model name: {model}")
        
    return llm
