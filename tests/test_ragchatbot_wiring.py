from langchain.schema import Document
from rag_chatbot.generator import RAGChatbot
from langchain_core.embeddings import Embeddings


class DummyEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[0.2] * 8 for _ in texts]

    def embed_query(self, text):
        return [0.2] * 8


class DummyGenerator:
    def __init__(self, manager, **kwargs):
        self.manager = manager

    def generate_answer(self, query: str):
        return {"answer": "dummy", "source_documents": [], "model": "dummy"}


def test_chatbot_setup_and_ask(tmp_path, monkeypatch):
    # 1) Patch loader: pas de PDF
    import rag_chatbot.loader as loader
    monkeypatch.setattr(
        loader,
        "load_papers",
        lambda folder_path="x": [Document(page_content="RAG is retrieval + generation", metadata={"source_file": "x.pdf"})],
    )

    # 2) Patch embeddings
    import rag_chatbot.vectorstore as vs
    monkeypatch.setattr(vs, "get_embeddings", lambda provider="x", model_name=None: DummyEmbeddings())

    # 3) Patch generator backend pour éviter transformers/torch
    import rag_chatbot.generator as gen
    monkeypatch.setattr(gen, "HuggingFaceGenerator", DummyGenerator)

    bot = RAGChatbot(
        documents_path="does_not_matter",
        index_path=str(tmp_path / "idx"),
        embedding_provider="sentence-transformers",
        generator_type="local",
    )
    bot.setup(force_recreate_index=True)

    out = bot.ask("What is RAG?")
    assert out["answer"] == "dummy"
