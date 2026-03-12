"""
Author: Pranay Hedau
Purpose: Hybrid LLM factory — OpenAI + Ollama switchable via config.

Instead of hardcoding ChatOpenAI() in every agent, agents call get_llm()
and receive whatever provider is configured. This is the Factory Pattern:
one function that creates different objects based on configuration.

Usage:
    from src.models.llm import get_llm
    llm = get_llm()                          # default settings
    llm = get_llm(temperature=0.1)           # precise output (for evaluation)
    llm = get_llm(temperature=0.7)           # creative output (for outreach)
    Date Created: 03-07-2026
"""

from langchain_core.language_models.chat_models import BaseChatModel
from src.config import settings #config settings


"""Purpose: Create and return an LLM instance based on the configured provider.

    Args:
        temperature: Controls randomness (0.0 = deterministic, 1.0 = creative).
                     Default 0.3 is a good balance for structured tasks.
        streaming: If True, returns tokens as they're generated.
                   Used in the Streamlit UI for real-time output.
        json_mode: If True, constrains the model to output valid JSON.
                   Ollama uses format='json' (token-level constraint).
                   OpenAI uses response_format={'type': 'json_object'}.

    Returns:
        A LangChain chat model (either ChatOpenAI or ChatOllama).
        Both implement the same BaseChatModel interface, so agents
        don't care which one they get — they call .invoke() the same way.

    Raises:
        ValueError: If LLM_PROVIDER in .env is not "openai" or "ollama"
    """
def get_llm(temperature: float = 0.3, streaming: bool = False, json_mode: bool = False) -> BaseChatModel:
    
    provider = settings.llm_provider.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs = dict(
            model=settings.openai_model,
            temperature=temperature,
            streaming=streaming,
            api_key=settings.openai_api_key,
        )
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return ChatOpenAI(**kwargs)

    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = dict(
            model=settings.ollama_model,
            temperature=temperature,
            base_url=settings.ollama_base_url,
        )
        if json_mode:
            kwargs["format"] = "json"
        return ChatOllama(**kwargs)

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. "
            f"Set LLM_PROVIDER to 'openai' or 'ollama' in your .env file."
        )


"""Purpose: Separate LLM instance for evaluation (LLM-as-judge).

    Uses temperature=0.1 for maximum consistency — when the eval LLM
    rates an outreach message as 4/5, you want the same input to
    produce the same rating every time. Higher temperature would
    give different scores on each run, making evaluation unreliable.

    Why I created a separate function? Because agents might use temperature=0.5
    for creative output, but evaluation must always be near-deterministic.
    """
def get_eval_llm() -> BaseChatModel:
    
    return get_llm(temperature=0.1, streaming=False)#temp kept 0.1 as we want deterministic evaluation