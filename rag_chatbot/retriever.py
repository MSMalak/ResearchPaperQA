# rag_chatbot/retriever.py
from typing import List, Optional, Tuple
from langchain.schema import Document
from rag_chatbot.vectorstore import VectorStoreManager

def get_relevant_docs(
    manager: VectorStoreManager, 
    query: str, 
    k: int = 5, 
    score_threshold: Optional[float] = None
) -> List[Tuple[Document, float]]:
    """
    Retrieve top-k relevant document chunks from a vector store managed by VectorStoreManager.

    Args:
        manager (VectorStoreManager): Initialized vector store manager
        query (str): User query
        k (int): Number of top results to retrieve
        score_threshold (float, optional): Minimum similarity score to include

    Returns:
        List[Tuple[Document, float]]: List of (document, similarity score) tuples
    """
    return manager.search_similar(query=query, k=k, score_threshold=score_threshold)


# Example usage
if __name__ == "__main__":
    from rag_chatbot.loader import load_papers
    from rag_chatbot.vectorstore import VectorStoreManager

    # Load documents
    documents = load_papers()

    # Initialize vector store manager and create/load index
    manager = VectorStoreManager(store_type="faiss")
    vectorstore = manager.create_or_load_index(
        documents=documents,
        embedding_provider="sentence-transformers",
        index_path="test_index",
        force_recreate=False  # Set True if you want to rebuild
    )

    # Query
    query = "What is the main contribution of the first paper?"
    top_docs = get_relevant_docs(manager, query, k=3)

    # Print top results
    for i, (doc, score) in enumerate(top_docs):
        print(f"\nResult {i+1} (score: {score:.4f}):")
        print(doc.page_content[:500])
        print(f"Source: {doc.metadata.get('source_file', 'Unknown')}")
