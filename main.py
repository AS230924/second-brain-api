"""
Second Brain - Personal Knowledge Agent

Main entry point and CLI.
"""

import os
import sys
from pathlib import Path
from typing import Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    import yaml
    
    if config_path is None:
        config_path = Path(__file__).parent / "config" / "config.yaml"
        if not config_path.exists():
            config_path = Path(__file__).parent / "config" / "config.example.yaml"
            
    if not Path(config_path).exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}
        
    with open(config_path) as f:
        return yaml.safe_load(f)


class SecondBrain:
    """
    Main Second Brain application.
    
    High-level API for:
    - Ingesting knowledge
    - Querying your knowledge base
    - Running evaluations
    """
    
    def __init__(self, config: dict = None):
        self.config = config or load_config()
        
        self._embedder = None
        self._vector_store = None
        self._metadata_store = None
        self._agent = None
        
    @property
    def embedder(self):
        """Lazy load embedder."""
        if self._embedder is None:
            from src.processing.embeddings import get_embedder
            
            emb_config = self.config.get("embeddings", {})
            self._embedder = get_embedder(
                provider=emb_config.get("provider", "sentence-transformers"),
                model=emb_config.get("model", "all-mpnet-base-v2"),
            )
        return self._embedder
        
    @property
    def vector_store(self):
        """Lazy load vector store."""
        if self._vector_store is None:
            from src.storage.vector_store import VectorStore
            
            storage_config = self.config.get("storage", {})
            self._vector_store = VectorStore(
                persist_path=storage_config.get("vector_db_path", "./data/chroma_db"),
            )
        return self._vector_store
        
    @property
    def metadata_store(self):
        """Lazy load metadata store."""
        if self._metadata_store is None:
            from src.storage.vector_store import MetadataStore
            
            storage_config = self.config.get("storage", {})
            self._metadata_store = MetadataStore(
                db_path=storage_config.get("metadata_db_path", "./data/metadata.db"),
            )
        return self._metadata_store
        
    @property
    def agent(self):
        """Lazy load agent."""
        if self._agent is None:
            from src.agent.agent import SecondBrainAgent
            
            self._agent = SecondBrainAgent(
                vector_store=self.vector_store,
                metadata_store=self.metadata_store,
                embedder=self.embedder,
                config=self.config.get("retrieval", {}),
            )
        return self._agent
        
    def ingest(
        self,
        source_type: str,
        path: str,
        **kwargs,
    ) -> dict:
        """
        Ingest content from a source.
        
        Args:
            source_type: Type of source (kindle, notes, pdf, etc.)
            path: Path to source file/directory
            
        Returns:
            Ingestion statistics
        """
        from src.models import SourceType
        from src.processing.chunking import get_chunker_for_source
        
        # Get the appropriate ingester
        ingesters = self._get_ingesters()
        
        source_type_enum = SourceType(source_type)
        
        if source_type_enum not in ingesters:
            raise ValueError(f"No ingester for source type: {source_type}")
            
        ingester = ingesters[source_type_enum]
        
        # Get the appropriate chunker
        chunker = get_chunker_for_source(source_type_enum)
        
        # Process items
        stats = {
            "items_ingested": 0,
            "chunks_created": 0,
            "errors": [],
        }
        
        for item in ingester.ingest(path):
            try:
                # Store metadata
                self.metadata_store.add_knowledge_item(item)
                stats["items_ingested"] += 1
                
                # Chunk the item
                chunks = list(chunker.chunk(item))
                
                if chunks:
                    # Embed chunks
                    contents = [c.content for c in chunks]
                    embeddings = self.embedder.embed(contents)
                    
                    # Store in vector database
                    self.vector_store.add_chunks(chunks, embeddings)
                    stats["chunks_created"] += len(chunks)
                    
            except Exception as e:
                logger.error(f"Error processing item: {e}")
                stats["errors"].append(str(e))
                
        logger.info(f"Ingestion complete: {stats}")
        return stats
        
    def _get_ingesters(self) -> dict:
        """Get all available ingesters."""
        from src.models import SourceType
        from src.ingestion.kindle import KindleIngester
        from src.ingestion.markdown import MarkdownIngester, ObsidianIngester
        from src.ingestion.documents import PDFIngester, ArticleIngester
        
        return {
            SourceType.KINDLE_HIGHLIGHT: KindleIngester(),
            SourceType.NOTE: MarkdownIngester(),
            SourceType.PDF: PDFIngester(),
            SourceType.ARTICLE: ArticleIngester(),
        }
        
    def query(self, question: str, **kwargs) -> dict:
        """
        Query your knowledge base.
        
        Args:
            question: Natural language question
            
        Returns:
            Response dict with answer and sources
        """
        return self.agent.query(question, **kwargs)
        
    def find_connections(self, content: str, **kwargs) -> list:
        """Find connections to existing knowledge."""
        return self.agent.find_connections(content, **kwargs)
        
    def summarize(self, topic: str) -> str:
        """Summarize everything you know about a topic."""
        return self.agent.summarize_topic(topic)
        
    def stats(self) -> dict:
        """Get knowledge base statistics."""
        return self.agent.get_stats()
        
    def evaluate(self, eval_path: str = None) -> dict:
        """Run evaluation suite."""
        from src.evaluation.evaluator import EvaluationSuite, create_sample_eval_queries
        
        suite = EvaluationSuite(self.agent)
        
        if eval_path:
            suite.load_eval_queries(eval_path)
        else:
            suite.eval_queries = create_sample_eval_queries()
            
        results = suite.run_evaluation()
        suite.print_report()
        
        return results


def create_cli():
    """Create CLI using Typer."""
    try:
        import typer
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        logger.error("CLI dependencies not installed. Run: pip install typer rich")
        return None
        
    app = typer.Typer(
        name="second-brain",
        help="Your Personal Knowledge Agent",
    )
    console = Console()
    
    @app.command()
    def ingest(
        source_type: str = typer.Argument(..., help="Source type: kindle, notes, pdf"),
        path: str = typer.Argument(..., help="Path to source file/directory"),
    ):
        """Ingest content into your knowledge base."""
        brain = SecondBrain()
        
        console.print(f"[blue]Ingesting {source_type} from {path}...[/blue]")
        
        stats = brain.ingest(source_type, path)
        
        console.print(f"[green]✓ Ingested {stats['items_ingested']} items[/green]")
        console.print(f"[green]✓ Created {stats['chunks_created']} chunks[/green]")
        
        if stats["errors"]:
            console.print(f"[yellow]⚠ {len(stats['errors'])} errors occurred[/yellow]")
            
    @app.command()
    def ask(
        question: str = typer.Argument(..., help="Your question"),
        no_synthesis: bool = typer.Option(False, "--raw", help="Return raw results without synthesis"),
    ):
        """Ask a question about your knowledge."""
        brain = SecondBrain()
        
        console.print(f"[blue]Searching your knowledge base...[/blue]\n")
        
        response = brain.query(question, synthesize=not no_synthesis)
        
        if response.get("answer"):
            console.print("[bold]Answer:[/bold]")
            console.print(response["answer"])
            console.print()
            
        if response.get("sources"):
            console.print("[bold]Sources:[/bold]")
            for source in response["sources"][:5]:
                console.print(f"  • {source['title']} ({source['type']})")
                
    @app.command()
    def connect(
        content: str = typer.Argument(..., help="Content to find connections for"),
    ):
        """Find connections between new content and your knowledge."""
        brain = SecondBrain()
        
        results = brain.find_connections(content)
        
        console.print("[bold]Related items in your knowledge base:[/bold]\n")
        
        for i, result in enumerate(results[:5]):
            console.print(f"[blue]{i+1}. {result.chunk.source_title or 'Untitled'}[/blue]")
            console.print(f"   {result.chunk.content[:200]}...")
            console.print(f"   [dim]Relevance: {result.score:.3f}[/dim]\n")
            
    @app.command()
    def summarize(
        topic: str = typer.Argument(..., help="Topic to summarize"),
    ):
        """Summarize everything you know about a topic."""
        brain = SecondBrain()
        
        console.print(f"[blue]Summarizing your knowledge about '{topic}'...[/blue]\n")
        
        summary = brain.summarize(topic)
        console.print(summary)
        
    @app.command()
    def stats():
        """Show knowledge base statistics."""
        brain = SecondBrain()
        
        s = brain.stats()
        
        table = Table(title="Knowledge Base Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Items", str(s.get("knowledge_items", 0)))
        table.add_row("Total Chunks", str(s.get("chunks", 0)))
        table.add_row("Total Entities", str(s.get("entities", 0)))
        
        console.print(table)
        
        if s.get("by_source"):
            console.print("\n[bold]By Source Type:[/bold]")
            for source, count in s["by_source"].items():
                console.print(f"  {source}: {count}")
                
    @app.command()
    def evaluate(
        eval_file: str = typer.Option(None, help="Path to evaluation queries JSON"),
    ):
        """Run evaluation suite."""
        brain = SecondBrain()
        
        console.print("[blue]Running evaluation...[/blue]\n")
        
        brain.evaluate(eval_file)
        
    @app.command()
    def chat():
        """Interactive chat with your knowledge base."""
        brain = SecondBrain()
        
        console.print("[bold blue]Second Brain Chat[/bold blue]")
        console.print("Ask questions about your knowledge. Type 'quit' to exit.\n")
        
        while True:
            try:
                question = console.input("[green]You:[/green] ")
                
                if question.lower() in ["quit", "exit", "q"]:
                    break
                    
                if not question.strip():
                    continue
                    
                response = brain.query(question)
                
                console.print(f"\n[blue]Brain:[/blue] {response.get('answer', 'No answer found.')}\n")
                
            except KeyboardInterrupt:
                break
                
        console.print("\n[dim]Goodbye![/dim]")
        
    return app


# CLI entry point
def main():
    """Main entry point."""
    cli = create_cli()
    if cli:
        cli()
    else:
        # Fallback if CLI deps not installed
        print("Second Brain")
        print("Install CLI dependencies: pip install typer rich")
        

if __name__ == "__main__":
    main()
