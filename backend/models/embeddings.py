from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain.embeddings import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()

def get_embeddings(provider="openai"):
    """
    Retrieve embeddings based on the specified provider.

    Parameters:
    - provider (str): The chosen provider for embeddings. Options are 'openai' and 'huggingface'.

    Returns:
    - embeddings: An instance of embeddings from the selected provider.

    Raises:
    - ValueError: If the required API key is missing from environment variables.
    """
    
    if provider == "openai":
        # Retrieve the OpenAI API key and model name from environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_EMBEDDING_MODEL")
        # Check if the API key is not set and raise an error if missing
        if not api_key:
            raise ValueError("Missing OpenAI API key. Set OPENAI_API_KEY in environment variables.")
        
        # Instantiate OpenAIEmbeddings with the specified model and API key
        embeddings = OpenAIEmbeddings(model=model, openai_api_key=api_key)
    
    elif provider == "huggingface":
        # Retrieve the Hugging Face API token and model repository ID from environment variables
        api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        model = os.getenv("HUGGINGFACE_EMBEDDINGS_MODEL")
        
        # Check if the API token is not set and raise an error if missing
        if not api_key:
            raise ValueError("Missing Hugging Face API token. Set HUGGINGFACEHUB_API_TOKEN in environment variables.")
        
        # Instantiate HuggingFaceEmbeddings with the specified model repository ID and API token
        model_name = os.getenv("HUGGINGFACE_EMBEDDINGS_MODEL")
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    
    elif provider == "ollama":
        model_name = os.getenv("OLLAMA_LLM_MODEL")
        return OllamaEmbeddings(model=model_name)

    return embeddings
