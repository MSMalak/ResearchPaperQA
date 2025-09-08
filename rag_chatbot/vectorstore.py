# rag_chatbot/vectorstore.py
import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Install with: pip install faiss-cpu or faiss-gpu")

try:
    from langchain_community.vectorstores import FAISS
    from langchain.schema import Document
    LANGCHAIN_FAISS_AVAILABLE = True
except ImportError:
    LANGCHAIN_FAISS_AVAILABLE = False
    logging.warning("LangChain FAISS not available. Install with: pip install langchain-community")

try:
    from langchain_community.vectorstores import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

from rag_chatbot.embedder import get_embeddings, BaseEmbeddingWrapper

# Set up logging
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Enhanced vector store manager with multiple backend support."""
    
    def __init__(self, store_type: str = "faiss"):
        self.store_type = store_type.lower()
        self.vectorstore = None
        self.embeddings = None
        self.metadata = {}
        
    def create_or_load_index(
        self,
        documents: List[Document],
        embedding_provider: str = "sentence-transformers",
        embedding_model: Optional[str] = None,
        index_path: str = "vector_index",
        force_recreate: bool = False,
        **kwargs
    ) -> Any:
        """
        Create or load a vector store from document chunks.
        
        Args:
            documents: List of LangChain Document objects
            embedding_provider: Embedding provider ("openai", "sentence-transformers", etc.)
            embedding_model: Specific model name (optional)
            index_path: Path to save/load index
            force_recreate: Force recreation even if index exists
            **kwargs: Additional arguments for vector store creation
            
        Returns:
            Vector store object
        """
        try:
            # Initialize embeddings
            self.embeddings = get_embeddings(
                provider=embedding_provider, 
                model_name=embedding_model
            )
            
            # Convert to Path object
            index_path = Path(index_path)
            
            # Check if we should load existing index
            if not force_recreate and self._index_exists(index_path):
                logger.info(f"Loading existing {self.store_type} index from {index_path}")
                return self._load_index(index_path)
            else:
                logger.info(f"Creating new {self.store_type} index...")
                return self._create_index(documents, index_path, **kwargs)
                
        except Exception as e:
            logger.error(f"Error in create_or_load_index: {str(e)}")
            raise

    def _index_exists(self, index_path: Path) -> bool:
        """Check if index exists at given path."""
        if self.store_type == "faiss":
            return (index_path / "index.faiss").exists() and (index_path / "index.pkl").exists()
        elif self.store_type == "chroma":
            return index_path.exists() and any(index_path.iterdir())
        return False

    def _create_index(self, documents: List[Document], index_path: Path, **kwargs) -> Any:
        """Create new vector index."""
        if not documents:
            raise ValueError("No documents provided for indexing")
            
        # Validate documents
        self._validate_documents(documents)
        
        # Create index based on store type
        if self.store_type == "faiss":
            return self._create_faiss_index(documents, index_path, **kwargs)
        elif self.store_type == "chroma":
            return self._create_chroma_index(documents, index_path, **kwargs)
        else:
            raise ValueError(f"Unsupported store type: {self.store_type}")

    def _create_faiss_index(self, documents: List[Document], index_path: Path, **kwargs) -> FAISS:
        """Create FAISS vector store."""
        if not FAISS_AVAILABLE or not LANGCHAIN_FAISS_AVAILABLE:
            raise ImportError("FAISS dependencies not available")
            
        try:
            # Create FAISS index
            vectorstore = FAISS.from_documents(documents, self.embeddings)
            
            # Save index
            index_path.mkdir(parents=True, exist_ok=True)
            vectorstore.save_local(str(index_path))
            
            # Save metadata
            self._save_metadata(index_path, documents, **kwargs)
            
            logger.info(f"FAISS index created and saved at {index_path}")
            self.vectorstore = vectorstore
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error creating FAISS index: {str(e)}")
            raise

    def _create_chroma_index(self, documents: List[Document], index_path: Path, **kwargs) -> Chroma:
        """Create Chroma vector store."""
        if not CHROMA_AVAILABLE:
            raise ImportError("Chroma not available. Install with: pip install chromadb")
            
        try:
            # Create Chroma index
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(index_path)
            )
            
            # Save metadata
            self._save_metadata(index_path, documents, **kwargs)
            
            logger.info(f"Chroma index created and saved at {index_path}")
            self.vectorstore = vectorstore
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error creating Chroma index: {str(e)}")
            raise

    def _load_index(self, index_path: Path) -> Any:
        """Load existing vector index."""
        try:
            # Load metadata first
            self._load_metadata(index_path)
            
            if self.store_type == "faiss":
                vectorstore = FAISS.load_local(str(index_path), self.embeddings)
            elif self.store_type == "chroma":
                vectorstore = Chroma(
                    persist_directory=str(index_path),
                    embedding_function=self.embeddings
                )
            else:
                raise ValueError(f"Unsupported store type: {self.store_type}")
                
            self.vectorstore = vectorstore
            logger.info(f"Successfully loaded {self.store_type} index from {index_path}")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            raise

    def _validate_documents(self, documents: List[Document]) -> None:
        """Validate document format and content."""
        if not documents:
            raise ValueError("Document list is empty")
            
        empty_docs = [i for i, doc in enumerate(documents) if not doc.page_content.strip()]
        if empty_docs:
            logger.warning(f"Found {len(empty_docs)} empty documents at indices: {empty_docs[:10]}")
            
        logger.info(f"Validated {len(documents)} documents for indexing")

    def _save_metadata(self, index_path: Path, documents: List[Document], **kwargs) -> None:
        """Save index metadata."""
        metadata = {
            "created_at": datetime.now().isoformat(),
            "store_type": self.store_type,
            "num_documents": len(documents),
            "embedding_provider": getattr(self.embeddings, '__class__', {}).get('__name__', 'unknown'),
            "total_chars": sum(len(doc.page_content) for doc in documents),
            "avg_doc_length": sum(len(doc.page_content) for doc in documents) / len(documents),
            "kwargs": kwargs
        }
        
        # Save sample of source files
        source_files = set()
        for doc in documents[:100]:  # Sample first 100 docs
            if 'source_file' in doc.metadata:
                source_files.add(doc.metadata['source_file'])
        metadata["sample_source_files"] = list(source_files)
        
        with open(index_path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
            
        self.metadata = metadata
        logger.info("Metadata saved successfully")

    def _load_metadata(self, index_path: Path) -> None:
        """Load index metadata."""
        metadata_path = index_path / "metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            logger.info("Metadata loaded successfully")
        else:
            logger.warning("No metadata file found")
            self.metadata = {}

    def search_similar(
        self, 
        query: str, 
        k: int = 5, 
        score_threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score (optional)
            
        Returns:
            List of (Document, score) tuples
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call create_or_load_index first.")
            
        try:
            # Get similarity search with scores
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            # Filter by score threshold if provided
            if score_threshold is not None:
                results = [(doc, score) for doc, score in results if score >= score_threshold]
                
            logger.info(f"Found {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise

    def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to existing index."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
            
        try:
            self._validate_documents(documents)
            self.vectorstore.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to existing index")
            
            # Update metadata
            if 'num_documents' in self.metadata:
                self.metadata['num_documents'] += len(documents)
                
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        stats = {
            "store_type": self.store_type,
            "initialized": self.vectorstore is not None,
            "metadata": self.metadata
        }
        
        if self.vectorstore and self.store_type == "faiss":
            try:
                stats["index_size"] = self.vectorstore.index.ntotal
            except:
                pass
                
        return stats

# Convenience functions for backward compatibility
def create_faiss_index(
    documents: List[Document], 
    embedding_model: str = "sentence-transformers", 
    index_path: str = "faiss_index",
    **kwargs
) -> FAISS:
    """
    Legacy function for creating FAISS index (backward compatibility).
    
    Args:
        documents: Document chunks from loader.py
        embedding_model: Embedding provider name
        index_path: Path to save/load FAISS index
        **kwargs: Additional arguments
        
    Returns:
        FAISS vector store object
    """
    manager = VectorStoreManager(store_type="faiss")
    return manager.create_or_load_index(
        documents=documents,
        embedding_provider=embedding_model,
        index_path=index_path,
        **kwargs
    )

def get_available_stores() -> List[str]:
    """Get list of available vector store backends."""
    stores = []
    if FAISS_AVAILABLE and LANGCHAIN_FAISS_AVAILABLE:
        stores.append("faiss")
    if CHROMA_AVAILABLE:
        stores.append("chroma")
    return stores

# Example usage and testing
if __name__ == "__main__":
    try:
        # Import here to avoid circular imports
        from rag_chatbot.loader import load_papers
        
        print("Available vector stores:", get_available_stores())
        
        # Load sample documents
        print("Loading documents...")
        documents = load_papers()
        
        if not documents:
            print("No documents loaded. Please check your data folder.")
            exit(1)
        
        # Create vector store manager
        manager = VectorStoreManager(store_type="faiss")
        
        # Create index
        print("Creating vector index...")
        vectorstore = manager.create_or_load_index(
            documents=documents,
            embedding_provider="sentence-transformers",
            index_path="test_index"
        )
        
        # Test search
        print("Testing similarity search...")
        query = "machine learning algorithms"
        results = manager.search_similar(query, k=3)
        
        print(f"\nTop 3 results for '{query}':")
        for i, (doc, score) in enumerate(results):
            print(f"{i+1}. Score: {score:.4f}")
            print(f"   Content: {doc.page_content[:200]}...")
            print(f"   Source: {doc.metadata.get('source_file', 'Unknown')}")
            print()
        
        # Print stats
        print("Vector store stats:")
        stats = manager.get_stats()
        for key, value in stats.items():
            if key != 'metadata':
                print(f"  {key}: {value}")
        
        # Test legacy function
        print("\nTesting legacy create_faiss_index function...")
        legacy_store = create_faiss_index(
            documents[:10],  # Use subset for testing
            index_path="legacy_test_index"
        )
        print("Legacy function works correctly!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")