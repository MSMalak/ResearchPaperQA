# rag_chatbot/loader.py
import os
import logging
from pathlib import Path
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_papers(
    folder_path: str = "data/sample_papers", 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200,
    file_extensions: Optional[List[str]] = None
) -> List[Document]:
    """
    Load all PDFs in folder_path, split them into chunks, and return a list of Documents.
    
    Args:
        folder_path (str): Path to folder containing PDF files.
        chunk_size (int): Max characters per chunk.
        chunk_overlap (int): Overlap between chunks for context.
        file_extensions (List[str], optional): File extensions to process. Defaults to ['.pdf'].
    
    Returns:
        List[Document]: List of LangChain Document objects.
        
    Raises:
        FileNotFoundError: If folder_path doesn't exist.
        ValueError: If no valid files found in folder.
    """
    if file_extensions is None:
        file_extensions = ['.pdf']
    
    folder_path = Path(folder_path)
    
    # Check if folder exists
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder '{folder_path}' does not exist")
    
    # Initialize text splitter once
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]  # Better splitting for academic papers
    )
    
    all_docs = []
    processed_files = 0
    
    # Process all files in folder
    for file_path in folder_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in [ext.lower() for ext in file_extensions]:
            try:
                logger.info(f"Loading {file_path.name}...")
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
                
                # Add metadata about source file
                for doc in docs:
                    doc.metadata['source_file'] = file_path.name
                    doc.metadata['file_path'] = str(file_path)
                
                # Split documents into chunks
                split_docs = text_splitter.split_documents(docs)
                all_docs.extend(split_docs)
                processed_files += 1
                
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {str(e)}")
                continue
    
    if processed_files == 0:
        raise ValueError(f"No valid files found in '{folder_path}' with extensions {file_extensions}")
    
    logger.info(f"Successfully loaded {len(all_docs)} chunks from {processed_files} files")
    return all_docs

def validate_documents(documents: List[Document]) -> bool:
    """
    Validate loaded documents for basic quality checks.
    
    Args:
        documents (List[Document]): List of documents to validate.
        
    Returns:
        bool: True if documents pass validation.
    """
    if not documents:
        logger.warning("No documents to validate")
        return False
    
    empty_docs = [doc for doc in documents if not doc.page_content.strip()]
    if empty_docs:
        logger.warning(f"Found {len(empty_docs)} empty documents")
    
    avg_length = sum(len(doc.page_content) for doc in documents) / len(documents)
    logger.info(f"Average document length: {avg_length:.2f} characters")
    
    return True

# Example usage
if __name__ == "__main__":
    try:
        # Load documents
        documents = load_papers(
            folder_path="data/sample_papers",
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Validate documents
        if validate_documents(documents):
            print(f"\nFirst chunk preview:")
            print("-" * 50)
            print(documents[0].page_content[:500])
            print("-" * 50)
            print(f"Metadata: {documents[0].metadata}")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")