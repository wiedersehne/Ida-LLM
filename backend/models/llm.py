import os
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from langchain.llms import OpenAI
from langchain.llms import Cohere
from langchain.llms import Anthropic
from langchain.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

def get_llm(provider="openai", temperature=0):
    """
    Returns an instance of an LLM based on the specified provider.

    Parameters:
    - model_name (str): The name of the model to use.
    - provider (str): The provider of the model. Options: "openai", "huggingface", "cohere", "anthropic".
    - temperature (float): Sampling temperature.

    Returns:
    - An instance of the selected LLM.
    """
    if provider == "openai":
        model_name = os.getenv("OPENAI_LLM_MODEL")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OpenAI API key. Set OPENAI_API_KEY in environment variables.")
        return ChatOpenAI(model_name=model_name, temperature=temperature, openai_api_key=api_key)
    
    elif provider == "huggingface":
        model_name = os.getenv("HUGGINGFACE_LLM_MODEL")
        api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not api_key:
            raise ValueError("Missing Hugging Face API token. Set HUGGINGFACEHUB_API_TOKEN in environment variables.")
        return HuggingFaceHub(repo_id=model_name, model_kwargs={"temperature": temperature}, huggingfacehub_api_token=api_key)
    
    elif provider == "ollama":
        model_name = os.getenv("OLLAMA_LLM_MODEL")
        return Ollama(model=model_name)
    
    elif provider == "anthropic":
        model_name = os.getenv("ANTHROPIC_LLM_MODEL")   
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Missing Anthropic API key. Set ANTHROPIC_API_KEY in environment variables.")
        return Anthropic(model=model_name, temperature=temperature, anthropic_api_key=api_key)
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    

