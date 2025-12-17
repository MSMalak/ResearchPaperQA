from langchain.schema import Document
from rag_chatbot.vectorstore import VectorStoreManager
from langchain_core.embeddings import Embeddings


class DummyEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[0.1] * 8 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 8


def test_stats_after_index_creation(tmp_path, monkeypatch):
    import rag_chatbot.vectorstore as vs
    monkeypatch.setattr(vs, "get_embeddings", lambda provider="x", model_name=None: DummyEmbeddings())

    docs = [Document(page_content="hello", metadata={"source_file": "x.pdf"})]
    mgr = VectorStoreManager(store_type="faiss")

    mgr.create_or_load_index(
        documents=docs,
        embedding_provider="sentence-transformers",
        index_path=str(tmp_path / "idx"),
        force_recreate=True,
    )

    stats = mgr.get_stats()
    assert stats["store_type"] == "faiss"
    assert stats["initialized"] is True
    assert "metadata" in stats
