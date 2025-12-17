# rag_chatbot/main.py
import sys
import os
import logging
from typing import Optional, Dict, Any
import argparse

from rag_chatbot.generator import RAGChatbot, get_available_generators

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



def parse_args():
    parser = argparse.ArgumentParser(
        description="ResearchPaperQA — RAG chatbot over research PDFs"
    )

    parser.add_argument(
        "--documents",
        type=str,
        default="data/sample_papers",
        help="Path to directory containing PDF documents"
    )

    parser.add_argument(
        "--generator",
        type=str,
        choices=["local", "openai"],
        default="local",
        help="Answer generation backend"
    )

    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Rebuild vector index from documents"
    )

    return parser.parse_args()


def display_welcome():
    """Display welcome message and instructions."""
    print("=" * 60)
    print("🤖 Welcome to Research Paper QA Chatbot!")
    print("=" * 60)
    print("\nThis chatbot can answer questions about your research papers.")
    print("\nCommands:")
    print("  - Type your question and press Enter")
    print("  - 'sources' - Show sources for the last answer")
    print("  - 'stats' - Show chatbot statistics")
    print("  - 'help' - Show this help message")
    print("  - 'exit' or 'quit' - Exit the chatbot")
    print("\n" + "-" * 60 + "\n")

def display_help():
    """Display help information."""
    print("\n📖 Help:")
    print("- Ask questions about your research papers")
    print("- The bot will search through your documents and provide answers")
    print("- Use 'sources' to see which documents were used for the last answer")
    print("- Use 'stats' to see information about your document collection")
    print("- Questions work best when specific (e.g., 'What methodology was used?')")
    print()

def display_sources(source_documents, max_sources: int = 3):
    """Display source documents used for the answer."""
    if not source_documents:
        print("No source documents available for the last answer.")
        return
    
    print(f"\n📚 Sources used (showing top {min(len(source_documents), max_sources)}):")
    print("-" * 50)
    
    for i, doc in enumerate(source_documents[:max_sources]):
        print(f"\n{i+1}. Source: {doc.metadata.get('source_file', 'Unknown')}")
        if 'page' in doc.metadata:
            print(f"   Page: {doc.metadata['page']}")
        
        # Show snippet of content
        content = doc.page_content.strip()
        if len(content) > 200:
            content = content[:200] + "..."
        print(f"   Content: {content}")
    
    if len(source_documents) > max_sources:
        print(f"\n... and {len(source_documents) - max_sources} more sources")
    print()

def display_stats(chatbot: RAGChatbot):
    """Display chatbot statistics."""
    try:
        if chatbot.manager:
            stats = chatbot.manager.get_stats()
            print("\n📊 Chatbot Statistics:")
            print("-" * 30)
            print(f"Vector Store: {stats.get('store_type', 'Unknown')}")
            print(f"Index Status: {'✓ Loaded' if stats.get('initialized') else '✗ Not loaded'}")
            
            metadata = stats.get('metadata', {})
            if metadata:
                print(f"Documents: {metadata.get('num_documents', 'Unknown')}")
                print(f"Total Characters: {metadata.get('total_chars', 'Unknown'):,}")
                print(f"Avg Document Length: {metadata.get('avg_doc_length', 0):.0f} chars")
                print(f"Created: {metadata.get('created_at', 'Unknown')}")
                
                sample_files = metadata.get('sample_source_files', [])
                if sample_files:
                    print(f"Sample Files: {', '.join(sample_files[:3])}")
                    if len(sample_files) > 3:
                        print(f"              ... and {len(sample_files) - 3} more")
        else:
            print("📊 No statistics available - chatbot not initialized")
        print()
    except Exception as e:
        print(f"Error getting statistics: {str(e)}\n")

def setup_chatbot(
    documents_path: str = "data/sample_papers",
    generator_type: str = "auto",
    force_recreate: bool = False
) -> Optional[RAGChatbot]:
    """
    Set up the RAG chatbot with error handling.
    
    Args:
        documents_path: Path to documents folder
        generator_type: "openai", "local", or "auto"
        force_recreate: Force recreation of index
        
    Returns:
        Initialized RAGChatbot or None if setup failed
    """
    try:
        # Determine generator type
        available_generators = get_available_generators()
        
        if generator_type == "auto":
            # Try OpenAI first if API key is available, otherwise use local
            if "openai" in available_generators and os.getenv('OPENAI_API_KEY'):
                generator_type = "openai"
                print("🔑 OpenAI API key found - using OpenAI generator")
            elif "local" in available_generators:
                generator_type = "local"
                print("🏠 Using local HuggingFace generator")
            else:
                print("❌ No generators available. Please install required dependencies.")
                return None
        
        if generator_type not in available_generators:
            print(f"❌ Generator '{generator_type}' not available.")
            print(f"Available generators: {available_generators}")
            return None
        
        # Initialize chatbot
        print("🚀 Initializing RAG Chatbot...")
        chatbot = RAGChatbot(
            documents_path=documents_path,
            index_path="main_rag_index",
            embedding_provider="huggingface",
            generator_type=generator_type
        )
        
        # Setup with appropriate parameters
        setup_kwargs = {}
        if generator_type == "local":
            setup_kwargs.update({
                "model_name": "google/flan-t5-small",  # Small, fast model
                "max_length": 512,
                "temperature": 0.3
            })
        elif generator_type == "openai":
            setup_kwargs.update({
                "model_name": "gpt-3.5-turbo",
                "temperature": 0.0,
                "max_tokens": 500
            })
        
        print("📚 Loading documents and creating/loading vector index...")
        chatbot.setup(
            force_recreate_index=force_recreate,
            **setup_kwargs
        )
        
        print("✅ Chatbot setup complete!")
        return chatbot
        
    except FileNotFoundError as e:
        print(f"❌ Documents not found: {str(e)}")
        print("Please make sure your documents are in the 'data/sample_papers' folder")
        return None
    except ValueError as e:
        if "API key" in str(e):
            print("❌ OpenAI API key not found.")
            print("Please set your OPENAI_API_KEY environment variable or use local generator.")
            return None
        else:
            print(f"❌ Setup error: {str(e)}")
            return None
    except Exception as e:
        print(f"❌ Unexpected error during setup: {str(e)}")
        logger.error(f"Chatbot setup failed: {str(e)}", exc_info=True)
        return None

def interactive_chat(chatbot: RAGChatbot):
    """Run the interactive chat loop."""
    display_welcome()
    
    last_result = None  # Store last result for sources command
    
    while True:
        try:
            query = input("💬 Your question: ").strip()
            
            if not query:
                continue
                
            # Handle commands
            if query.lower() in ["exit", "quit", "q"]:
                print("\n👋 Thank you for using Research Paper QA Chatbot!")
                print("Goodbye!\n")
                break
                
            elif query.lower() == "help":
                display_help()
                continue
                
            elif query.lower() == "sources":
                if last_result and last_result.get("source_documents"):
                    display_sources(last_result["source_documents"])
                else:
                    print("No sources available. Ask a question first!\n")
                continue
                
            elif query.lower() == "stats":
                display_stats(chatbot)
                continue
            
            # Generate answer
            print("\n🔍 Searching for relevant information...")
            
            try:
                result = chatbot.ask(query)
                last_result = result  # Store for sources command
                
                print("\n💡 Answer:")
                print("-" * 30)
                print(result["answer"])
                
                # Show brief source info
                source_count = len(result.get("source_documents", []))
                if source_count > 0:
                    print(f"\n📄 Based on {source_count} source document(s)")
                    print("(Type 'sources' to see details)")
                
                print("\n" + "=" * 60 + "\n")
                
            except Exception as e:
                print(f"\n❌ Error generating answer: {str(e)}")
                logger.error(f"Answer generation failed: {str(e)}", exc_info=True)
                print("Please try rephrasing your question.\n")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break

def main():
    """Main function."""
    args = parse_args()

    chatbot = setup_chatbot(
        documents_path=args.documents,
        generator_type=args.generator,
        force_recreate=args.recreate
    )
   
    if chatbot is None:
        print("\n❌ Failed to initialize chatbot. Exiting...")
        sys.exit(1)
    
    # Start interactive chat
    try:
        interactive_chat(chatbot)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        logger.error(f"Main execution failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()