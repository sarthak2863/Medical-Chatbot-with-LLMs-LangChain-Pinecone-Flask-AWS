from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from typing import List
from langchain_core.documents import Document


# ---------------- LOAD PDF FILES ----------------
def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()


# ---------------- CLEAN METADATA ----------------
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs = []
    for doc in docs:
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": doc.metadata.get("source")}
            )
        )
    return minimal_docs


# ---------------- SPLIT TEXT ----------------
def text_split(extracted_data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    return splitter.split_documents(extracted_data)


# ---------------- EMBEDDINGS (NO LANGCHAIN DEPENDENCY) ----------------
class HuggingFaceEmbeddingWrapper:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()


def download_hugging_face_embeddings():
    return HuggingFaceEmbeddingWrapper()