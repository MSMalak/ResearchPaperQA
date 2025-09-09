# rag_chatbot/embedder.py
import os
from dotenv import load_dotenv
import logging
from typing import List, Union, Optional
from abc import ABC, abstractmethod
from openai import OpenAI
from langchain.embeddings.base import Embeddings

# Charger les variables depuis config.env
load_dotenv('./config.env')

# Import with fallbacks for different embedding providers
try:
    from langchain_openai import OpenAIEmbeddings
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI embeddings not available. Install with: pip install langchain-openai")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("Sentence Transformers not available. Install with: pip install sentence-transformers")

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)

class BaseEmbeddingWrapper(Embeddings):
    """Abstract base class for embedding wrappers to ensure consistent interface."""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        pass

class OpenAIEmbeddingWrapper(BaseEmbeddingWrapper):
    """Wrapper for OpenAI embeddings."""
    
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI embeddings not available. Install with: pip install langchain-openai")
        
        # Check for API key
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        try:
            self.embeddings = OpenAIEmbeddings(
                model=model_name,
                openai_api_key=api_key
            )
            # Test the connection
            self.embeddings.embed_query("test")
            logger.info(f"OpenAI embeddings initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embeddings: {str(e)}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

class SentenceTransformerWrapper(BaseEmbeddingWrapper):
    """Wrapper for Sentence Transformers embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("Sentence Transformers not available. Install with: pip install sentence-transformers")
        
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Sentence Transformer initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Sentence Transformer: {str(e)}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text])
        return embedding[0].tolist()

class HuggingFaceEmbeddingWrapper(BaseEmbeddingWrapper):
    """Wrapper for HuggingFace embeddings via LangChain."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("HuggingFace embeddings not available. Install with: pip install langchain-huggingface")
        
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
            logger.info(f"HuggingFace embeddings initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace embeddings: {str(e)}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

def get_embeddings(
    provider: str = "huggingface", 
    model_name: Optional[str] = None,
    **kwargs
) -> BaseEmbeddingWrapper:
    """
    Returns an embedding function based on selected provider and model.
    
    Args:
        provider (str): "openai", "sentence-transformers", or "huggingface"
        model_name (str, optional): Specific model name to use
        **kwargs: Additional arguments passed to the embedding wrapper
    
    Returns:
        BaseEmbeddingWrapper: Embedding wrapper with consistent interface
        
    Raises:
        ValueError: If provider is not supported or requirements not met
    """
    provider = provider.lower()
    
    # Default model names for each provider
    default_models = {
        "openai": "text-embedding-3-small",
        "sentence-transformers": "all-MiniLM-L6-v2", 
        "huggingface": "sentence-transformers/all-MiniLM-L6-v2"
    }
    
    if model_name is None:
        model_name = default_models.get(provider)
    
    try:
        if provider == "openai":
            return OpenAIEmbeddingWrapper(model_name=model_name, **kwargs)
        elif provider == "sentence-transformers":
            return SentenceTransformerWrapper(model_name=model_name)
        elif provider == "huggingface":
            return HuggingFaceEmbeddingWrapper(model_name=model_name)
        else:
            available_providers = []
            if OPENAI_AVAILABLE: available_providers.append("openai")
            if SENTENCE_TRANSFORMERS_AVAILABLE: available_providers.append("sentence-transformers") 
            if HUGGINGFACE_AVAILABLE: available_providers.append("huggingface")
            
            raise ValueError(
                f"Provider '{provider}' not supported. "
                f"Available providers: {available_providers}"
            )
    
    except Exception as e:
        logger.error(f"Failed to initialize {provider} embeddings: {str(e)}")
        
        # Fallback to huggingface if available
        if provider != "huggingface" and HUGGINGFACE_AVAILABLE:
            logger.info("Falling back to huggingface...")
            return HuggingFaceEmbeddingWrapper()
        
        raise

def get_available_providers() -> List[str]:
    """Returns list of available embedding providers."""
    providers = []
    if OPENAI_AVAILABLE:
        providers.append("openai")
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        providers.append("sentence-transformers")
    if HUGGINGFACE_AVAILABLE:
        providers.append("huggingface")
    return providers

def benchmark_embeddings(
    texts: List[str], 
    providers: Optional[List[str]] = None
) -> dict:
    """
    Benchmark different embedding providers on a sample of texts.
    
    Args:
        texts: Sample texts to embed
        providers: List of providers to test (defaults to all available)
    
    Returns:
        dict: Benchmark results with timing and dimension info
    """
    import time
    
    if providers is None:
        providers = get_available_providers()
    
    results = {}
    
    for provider in providers:
        try:
            logger.info(f"Benchmarking {provider}...")
            start_time = time.time()
            
            embedder = get_embeddings(provider=provider)
            embeddings = embedder.embed_documents(texts[:5])  # Test with first 5 texts
            
            end_time = time.time()
            
            results[provider] = {
                "success": True,
                "time_seconds": end_time - start_time,
                "embedding_dimension": len(embeddings[0]) if embeddings else 0,
                "texts_processed": len(embeddings)
            }
            
        except Exception as e:
            results[provider] = {
                "success": False,
                "error": str(e)
            }
    
    return results

# Example usage and testing
if __name__ == "__main__":
    # Sample texts for testing
    sample_texts = [
        "Natural language processing is a fascinating field.",
        "Machine learning models can process text efficiently.", 
        "Embeddings capture semantic meaning in vector space."
    ]
    
    print("Available providers:", get_available_providers())
    
    try:
        # Test with default provider
        embedder = get_embeddings()
        
        # Test document embedding
        doc_embeddings = embedder.embed_documents(sample_texts)
        print(f"\nDocument embeddings shape: {len(doc_embeddings)} x {len(doc_embeddings[0])}")
        
        # Test query embedding
        query_embedding = embedder.embed_query("What is machine learning?")
        print(f"Query embedding dimension: {len(query_embedding)}")
        
        # Benchmark available providers
        print("\nBenchmarking providers...")
        benchmark_results = benchmark_embeddings(sample_texts)
        for provider, result in benchmark_results.items():
            if result["success"]:
                print(f"{provider}: {result['time_seconds']:.3f}s, dim={result['embedding_dimension']}")
            else:
                print(f"{provider}: Failed - {result['error']}")
                
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")