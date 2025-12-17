from langchain.schema import Document
from rag_chatbot.vectorstore import VectorStoreManager
from langchain_core.embeddings import Embeddings


class DummyEmbeddings(Embeddings):
    """Embeddings deterministes pour tests (pas de modèles, pas de torch)."""
    def __init__(self, dim: int = 8):
        self.dim = dim

    def _vec(self, text: str):
        # vecteur simple & déterministe (pas besoin d’être “bon”, juste stable)
        h = abs(hash(text))
        return [((h >> i) & 255) / 255.0 for i in range(self.dim)]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def test_faiss_roundtrip_create_and_load(tmp_path, monkeypatch):
    # Patch: vectorstore.get_embeddings(...) -> DummyEmbeddings()
    import rag_chatbot.vectorstore as vs
    monkeypatch.setattr(vs, "get_embeddings", lambda provider="x", model_name=None: DummyEmbeddings())

    docs = [
        Document(page_content="Transformers are sequence models.", metadata={"source_file": "a.pdf", "page": 1}),
        Document(page_content="FAISS is used for similarity search.", metadata={"source_file": "b.pdf", "page": 2}),
        Document(page_content="RAG combines retrieval with generation.", metadata={"source_file": "c.pdf", "page": 3}),
    ]

    index_dir = tmp_path / "index"
    mgr = VectorStoreManager(store_type="faiss")

    # Create
    mgr.create_or_load_index(
        documents=docs,
        embedding_provider="sentence-transformers",
        index_path=str(index_dir),
        force_recreate=True,
    )

    assert (index_dir / "index.faiss").exists()
    assert (index_dir / "index.pkl").exists()

    # Load (same manager ok)
    mgr2 = VectorStoreManager(store_type="faiss")
    monkeypatch.setattr(vs, "get_embeddings", lambda provider="x", model_name=None: DummyEmbeddings())

    mgr2.create_or_load_index(
        documents=docs,  # pas utilisé si load
        embedding_provider="sentence-transformers",
        index_path=str(index_dir),
        force_recreate=False,
    )

    results = mgr2.search_similar("What is FAISS?", k=2)
    assert len(results) == 2
    assert isinstance(results[0][0].page_content, str)
