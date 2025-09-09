# rag_chatbot/generator.py
import os
import logging
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

# LangChain imports with fallbacks
try:
    from langchain_openai import ChatOpenAI
    OPENAI_CHAT_AVAILABLE = True
except ImportError:
    try:
        from langchain.chat_models import ChatOpenAI
        OPENAI_CHAT_AVAILABLE = True
    except ImportError:
        OPENAI_CHAT_AVAILABLE = False

try:
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain.schema import BaseRetriever
    LANGCHAIN_CHAINS_AVAILABLE = True
except ImportError:
    LANGCHAIN_CHAINS_AVAILABLE = False

try:
    from langchain_community.llms import HuggingFacePipeline
    HF_PIPELINE_AVAILABLE = True
except ImportError:
    try:
        from langchain.llms import HuggingFacePipeline
        HF_PIPELINE_AVAILABLE = True
    except ImportError:
        HF_PIPELINE_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from rag_chatbot.vectorstore import VectorStoreManager

# Set up logging
logger = logging.getLogger(__name__)

class BaseGenerator(ABC):
    """Abstract base class for answer generators."""
    
    @abstractmethod
    def generate_answer(self, query: str, **kwargs) -> str:
        """Generate an answer for the given query."""
        pass

class OpenAIGenerator(BaseGenerator):
    """Generator using OpenAI models."""
    
    def __init__(
        self, 
        manager: VectorStoreManager,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None
    ):
        if not OPENAI_CHAT_AVAILABLE or not LANGCHAIN_CHAINS_AVAILABLE:
            raise ImportError("OpenAI or LangChain dependencies not available")
            
        if manager.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_or_load_index first.")
        
        # Check for API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.manager = manager
        self.retriever = manager.vectorstore.as_retriever()
        
        # Initialize LLM
        llm_kwargs = {
            "model": model_name,
            "temperature": temperature,
            "api_key": api_key
        }
        if max_tokens:
            llm_kwargs["max_tokens"] = max_tokens
            
        self.llm = ChatOpenAI(**llm_kwargs)
        
        # Create custom prompt for research papers
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a research assistant analyzing academic papers. Use the following context from research papers to answer the question accurately and comprehensively.

Context from papers:
{context}

Question: {question}

Instructions:
- Base your answer primarily on the provided context
- If the context doesn't contain enough information, state this clearly
- Include relevant details, methodologies, and findings when available
- If applicable, mention which paper or section the information comes from

Answer:"""
        )
        
        # Build RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )
        
        logger.info(f"OpenAI generator initialized with model: {model_name}")

    def generate_answer(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Generate an answer using OpenAI model.
        
        Args:
            query: User question
            **kwargs: Additional arguments
            
        Returns:
            Dict containing answer and source documents
        """
        try:
            logger.info(f"Generating answer for query: {query[:100]}...")
            result = self.qa_chain({"query": query})
            
            return {
                "answer": result["result"],
                "source_documents": result.get("source_documents", []),
                "model": "openai"
            }
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise

class HuggingFaceGenerator(BaseGenerator):
    """Generator using local HuggingFace models."""
    
    def __init__(
        self,
        manager: VectorStoreManager,
        model_name: str = "google/flan-t5-small",
        temperature: float = 0.5,
        max_length: int = 512,
        device: Optional[str] = None
    ):
        if not HF_PIPELINE_AVAILABLE or not TRANSFORMERS_AVAILABLE or not LANGCHAIN_CHAINS_AVAILABLE:
            raise ImportError("HuggingFace or Transformers dependencies not available")
            
        if manager.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_or_load_index first.")
        
        self.manager = manager
        self.retriever = manager.vectorstore.as_retriever()
        self.model_name = model_name
        
        # Determine device
        if device is None:
            import torch
            if torch.cuda.is_available():
                device = 0  # Use first GPU
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = -1  # CPU
        
        try:
            # Setup HuggingFace pipeline
            self.hf_pipeline = pipeline(
                "text2text-generation",
                model=model_name,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                device=device,
                #return_full_text=False
            )
            
            # Create LangChain wrapper
            self.llm = HuggingFacePipeline(pipeline=self.hf_pipeline)
            
            logger.info(f"HuggingFace generator initialized with model: {model_name} on device: {device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace model: {str(e)}")
            # Fallback to CPU
            try:
                logger.info("Trying CPU fallback...")
                self.hf_pipeline = pipeline(
                    "text2text-generation",
                    model=model_name,
                    max_length=max_length,
                    do_sample=True,
                    temperature=temperature,
                    device=-1,
                    return_full_text=False
                )
                self.llm = HuggingFacePipeline(pipeline=self.hf_pipeline)
                logger.info("CPU fallback successful")
            except Exception as fallback_e:
                logger.error(f"CPU fallback also failed: {str(fallback_e)}")
                raise
        
        # Create prompt template optimized for smaller models
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Context: {context}

Question: {question}

Based on the context above, provide a concise and accurate answer:"""
        )
        
        # Build RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )

    def generate_answer(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Generate an answer using HuggingFace model.
        
        Args:
            query: User question
            **kwargs: Additional arguments
            
        Returns:
            Dict containing answer and source documents
        """
        try:
            logger.info(f"Generating answer with {self.model_name} for query: {query[:100]}...")
            result = self.qa_chain({"query": query})
            
            return {
                "answer": result["result"],
                "source_documents": result.get("source_documents", []),
                "model": self.model_name
            }
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise

class RAGChatbot:
    """Main RAG Chatbot class that manages the complete pipeline."""
    
    def __init__(
        self,
        documents_path: str = "data/sample_papers",
        index_path: str = "rag_index",
        embedding_provider: str = "huggingface",
        generator_type: str = "local"  # "openai" or "local"
    ):
        self.documents_path = documents_path
        self.index_path = index_path
        self.embedding_provider = embedding_provider
        self.generator_type = generator_type
        
        self.manager = None
        self.generator = None
        
        logger.info("RAG Chatbot initialized")
    
    def setup(
        self,
        force_recreate_index: bool = False,
        **generator_kwargs
    ) -> None:
        """
        Setup the complete RAG pipeline.
        
        Args:
            force_recreate_index: Force recreation of vector index
            **generator_kwargs: Arguments for generator initialization
        """
        # Load documents
        logger.info("Loading documents...")
        from rag_chatbot.loader import load_papers
        documents = load_papers(folder_path=self.documents_path)
        
        # Initialize vector store
        logger.info("Setting up vector store...")
        self.manager = VectorStoreManager(store_type="faiss")
        self.manager.create_or_load_index(
            documents=documents,
            embedding_provider=self.embedding_provider,
            index_path=self.index_path,
            force_recreate=force_recreate_index
        )
        
        # Initialize generator
        logger.info(f"Setting up {self.generator_type} generator...")
        if self.generator_type == "openai":
            self.generator = OpenAIGenerator(self.manager, **generator_kwargs)
        elif self.generator_type == "local":
            self.generator = HuggingFaceGenerator(self.manager, **generator_kwargs)
        else:
            raise ValueError(f"Unsupported generator type: {self.generator_type}")
        
        logger.info("RAG pipeline setup complete!")
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question to the RAG system.
        
        Args:
            question: User question
            
        Returns:
            Dict containing answer and metadata
        """
        if not self.generator:
            raise ValueError("Chatbot not setup. Call setup() first.")
        
        return self.generator.generate_answer(question)
    
    def get_relevant_documents(self, query: str, k: int = 5) -> List:
        """Get relevant documents for a query without generating an answer."""
        if not self.manager or not self.manager.vectorstore:
            raise ValueError("Vector store not initialized. Call setup() first.")
        
        return self.manager.search_similar(query, k=k)

# Legacy functions for backward compatibility
def generate_answer(
    manager: VectorStoreManager,
    query: str,
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0.0
) -> str:
    """
    Legacy function for generating answers with OpenAI (backward compatibility).
    
    Args:
        manager: Initialized vector store manager
        query: User question
        model_name: OpenAI LLM model name
        temperature: LLM randomness
        
    Returns:
        Generated answer string
    """
    generator = OpenAIGenerator(
        manager=manager,
        model_name=model_name,
        temperature=temperature
    )
    result = generator.generate_answer(query)
    return result["answer"]

def generate_answer_local(
    manager: VectorStoreManager,
    query: str,
    hf_model: str = "google/flan-t5-small",
    temperature: float = 0.5,
    max_length: int = 256
) -> str:
    """
    Legacy function for generating answers with local models (backward compatibility).
    
    Args:
        manager: Initialized vector store manager
        query: User question
        hf_model: HuggingFace model name
        temperature: LLM randomness
        max_length: Max tokens for generation
        
    Returns:
        Generated answer string
    """
    generator = HuggingFaceGenerator(
        manager=manager,
        model_name=hf_model,
        temperature=temperature,
        max_length=max_length
    )
    result = generator.generate_answer(query)
    return result["answer"]

def get_available_generators() -> List[str]:
    """Get list of available generator types."""
    generators = []
    if OPENAI_CHAT_AVAILABLE and LANGCHAIN_CHAINS_AVAILABLE:
        generators.append("openai")
    if HF_PIPELINE_AVAILABLE and TRANSFORMERS_AVAILABLE and LANGCHAIN_CHAINS_AVAILABLE:
        generators.append("local")
    return generators

# Example usage
if __name__ == "__main__":
    try:
        print("Available generators:", get_available_generators())
        
        # Method 1: Using RAGChatbot class (recommended)
        print("\n=== Testing RAGChatbot class ===")
        chatbot = RAGChatbot(
            documents_path="data/sample_papers",
            index_path="test_generator_index",
            embedding_provider="huggingface",
            generator_type="local"
        )
        
        # Setup the pipeline
        chatbot.setup(
            force_recreate_index=True,
            model_name="google/flan-t5-small",
            max_length=512
        )
        
        # Ask questions
        questions = [
            "What is the main contribution of this research?",
            "What methodology was used?",
            "What are the key findings?"
        ]
        
        for question in questions:
            print(f"\nQuestion: {question}")
            result = chatbot.ask(question)
            print(f"Answer: {result['answer']}")
            print(f"Model: {result['model']}")
            print(f"Sources: {len(result['source_documents'])} documents")
            print("-" * 60)
        
        # Method 2: Legacy approach (for backward compatibility)
        print("\n=== Testing legacy functions ===")
        from rag_chatbot.loader import load_papers
        from rag_chatbot.vectorstore import VectorStoreManager
        
        documents = load_papers()
        manager = VectorStoreManager(store_type="faiss")
        manager.create_or_load_index(
            documents=documents,
            embedding_provider="huggingface",
            index_path="legacy_test_index",
            force_recreate=True
        )
        
        query = "What is the main contribution?"
        answer = generate_answer_local(manager, query)
        print(f"Legacy answer: {answer}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")