"""
Second Brain API

FastAPI backend that exposes the Second Brain functionality via REST API.
This can be deployed to Railway, Render, or any cloud provider.
"""

import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import Second Brain components
from main import SecondBrain
from src.models import SourceType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Second Brain API",
    description="Personal Knowledge Agent API",
    version="1.0.0",
)

# CORS - allow Lovable frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://*.lovable.app",
        "https://*.lovableproject.com",
        # Add your production domain here
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Second Brain (singleton)
brain: SecondBrain = None


def get_brain() -> SecondBrain:
    """Get or create the Second Brain instance."""
    global brain
    if brain is None:
        config = {
            "storage": {
                "vector_db_path": os.getenv("VECTOR_DB_PATH", "./data/chroma_db"),
                "metadata_db_path": os.getenv("METADATA_DB_PATH", "./data/metadata.db"),
            },
            "embeddings": {
                "provider": os.getenv("EMBEDDING_PROVIDER", "sentence-transformers"),
                "model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            },
            "retrieval": {
                "hybrid_weight": 0.3,
                "rerank": False,
            },
        }
        brain = SecondBrain(config)
    return brain


# ============== Request/Response Models ==============

class QueryRequest(BaseModel):
    """Request model for querying the knowledge base."""
    question: str = Field(..., description="Natural language question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")
    synthesize: bool = Field(default=True, description="Generate synthesized answer")


class QueryResponse(BaseModel):
    """Response model for queries."""
    answer: Optional[str] = None
    sources: list[dict] = []
    query_time_ms: float = 0


class ConnectRequest(BaseModel):
    """Request model for finding connections."""
    content: str = Field(..., description="Content to find connections for")
    top_k: int = Field(default=5, ge=1, le=20)


class ConnectionResult(BaseModel):
    """A single connection result."""
    title: Optional[str]
    author: Optional[str]
    source_type: str
    content_preview: str
    relevance_score: float


class IngestRequest(BaseModel):
    """Request model for ingestion."""
    source_type: str = Field(..., description="kindle_highlight, note, pdf, article")
    content: Optional[str] = Field(default=None, description="Text content to ingest")
    title: Optional[str] = Field(default=None)
    author: Optional[str] = Field(default=None)
    tags: list[str] = Field(default_factory=list)


class StatsResponse(BaseModel):
    """Response model for stats."""
    total_chunks: int = 0
    total_items: int = 0
    total_entities: int = 0
    by_source: dict = {}


class SourceItem(BaseModel):
    """A source in the knowledge base."""
    id: str
    title: Optional[str]
    author: Optional[str]
    source_type: str
    content_preview: str
    created_at: Optional[str]
    tags: list[str] = []


# ============== API Endpoints ==============

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Second Brain API",
        "version": "1.0.0",
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    brain = get_brain()
    stats = brain.stats()
    return {
        "status": "healthy",
        "knowledge_base": {
            "chunks": stats.get("chunks", 0),
            "items": stats.get("knowledge_items", 0),
        }
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the knowledge base with a natural language question.
    
    Returns a synthesized answer along with source citations.
    """
    import time
    start = time.time()
    
    try:
        brain = get_brain()
        response = brain.query(
            request.question,
            top_k=request.top_k,
            synthesize=request.synthesize,
        )
        
        query_time = (time.time() - start) * 1000
        
        return QueryResponse(
            answer=response.get("answer"),
            sources=response.get("sources", []),
            query_time_ms=round(query_time, 2),
        )
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/connect", response_model=list[ConnectionResult])
async def find_connections(request: ConnectRequest):
    """
    Find connections between new content and existing knowledge.
    
    Useful for "how does this relate to what I already know?"
    """
    try:
        brain = get_brain()
        results = brain.find_connections(request.content, top_k=request.top_k)
        
        connections = []
        for r in results:
            connections.append(ConnectionResult(
                title=r.chunk.source_title,
                author=r.chunk.source_author,
                source_type=r.chunk.source_type.value,
                content_preview=r.chunk.content[:300] + "..." if len(r.chunk.content) > 300 else r.chunk.content,
                relevance_score=round(r.score, 3),
            ))
            
        return connections
    except Exception as e:
        logger.error(f"Connection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize")
async def summarize(topic: str):
    """
    Summarize everything in the knowledge base about a topic.
    """
    try:
        brain = get_brain()
        summary = brain.summarize(topic)
        return {"topic": topic, "summary": summary}
    except Exception as e:
        logger.error(f"Summarize error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest_content(request: IngestRequest):
    """
    Ingest content directly (for small text content).
    
    For file uploads, use /ingest/file endpoint.
    """
    if not request.content:
        raise HTTPException(status_code=400, detail="Content is required")
        
    try:
        brain = get_brain()
        
        # Create a temporary file for ingestion
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(request.content)
            temp_path = f.name
            
        try:
            stats = brain.ingest(request.source_type, temp_path)
            return {
                "status": "success",
                "items_ingested": stats.get("items_ingested", 0),
                "chunks_created": stats.get("chunks_created", 0),
            }
        finally:
            os.unlink(temp_path)
            
    except Exception as e:
        logger.error(f"Ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/file")
async def ingest_file(
    file: UploadFile = File(...),
    source_type: str = "note",
    background_tasks: BackgroundTasks = None,
):
    """
    Ingest a file into the knowledge base.
    
    Supported formats:
    - .txt (My Clippings.txt for Kindle)
    - .md (Markdown notes)
    - .pdf (PDF documents)
    - .html (Saved articles)
    """
    # Validate file type
    allowed_extensions = {".txt", ".md", ".pdf", ".html", ".htm"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {allowed_extensions}"
        )
        
    # Map extensions to source types
    ext_to_source = {
        ".txt": "kindle_highlight" if "clipping" in file.filename.lower() else "note",
        ".md": "note",
        ".pdf": "pdf",
        ".html": "article",
        ".htm": "article",
    }
    
    if source_type == "auto":
        source_type = ext_to_source.get(file_ext, "note")
        
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
            
        # Ingest the file
        brain = get_brain()
        stats = brain.ingest(source_type, temp_path)
        
        # Clean up
        os.unlink(temp_path)
        
        return {
            "status": "success",
            "filename": file.filename,
            "source_type": source_type,
            "items_ingested": stats.get("items_ingested", 0),
            "chunks_created": stats.get("chunks_created", 0),
        }
        
    except Exception as e:
        logger.error(f"File ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/kindle")
async def ingest_kindle_highlights(highlights: list[dict]):
    """
    Ingest Kindle highlights directly from JSON.
    
    Each highlight should have:
    - content: The highlight text
    - title: Book title
    - author: Book author
    - location: (optional) Kindle location
    """
    try:
        brain = get_brain()
        
        # Create a clippings-format string
        clippings = []
        for h in highlights:
            entry = f"""{h.get('title', 'Unknown')} ({h.get('author', 'Unknown')})
- Your Highlight on Location {h.get('location', '0')} | Added on {datetime.now().strftime('%A, %B %d, %Y %I:%M:%S %p')}

{h.get('content', '')}
=========="""
            clippings.append(entry)
            
        clippings_text = "\n".join(clippings)
        
        # Save and ingest
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(clippings_text)
            temp_path = f.name
            
        try:
            stats = brain.ingest("kindle_highlight", temp_path)
            return {
                "status": "success",
                "highlights_processed": len(highlights),
                "chunks_created": stats.get("chunks_created", 0),
            }
        finally:
            os.unlink(temp_path)
            
    except Exception as e:
        logger.error(f"Kindle ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get knowledge base statistics."""
    try:
        brain = get_brain()
        stats = brain.stats()
        
        return StatsResponse(
            total_chunks=stats.get("chunks", 0),
            total_items=stats.get("knowledge_items", 0),
            total_entities=stats.get("entities", 0),
            by_source=stats.get("by_source", {}),
        )
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sources", response_model=list[SourceItem])
async def list_sources(
    source_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """
    List sources in the knowledge base.
    
    Optionally filter by source type.
    """
    try:
        brain = get_brain()
        
        # Query metadata store directly
        from sqlalchemy import text
        engine = brain.metadata_store._get_engine()
        
        with engine.connect() as conn:
            query = """
                SELECT id, source_title, source_author, source_type, 
                       content, created_at, tags
                FROM knowledge_items
            """
            params = {"limit": limit, "offset": offset}
            
            if source_type:
                query += " WHERE source_type = :source_type"
                params["source_type"] = source_type
                
            query += " ORDER BY created_at DESC LIMIT :limit OFFSET :offset"
            
            results = conn.execute(text(query), params).fetchall()
            
        sources = []
        for row in results:
            sources.append(SourceItem(
                id=row[0],
                title=row[1],
                author=row[2],
                source_type=row[3],
                content_preview=row[4][:200] + "..." if row[4] and len(row[4]) > 200 else (row[4] or ""),
                created_at=row[5],
                tags=row[6].split(",") if row[6] else [],
            ))
            
        return sources
        
    except Exception as e:
        logger.error(f"List sources error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sources/{source_id}")
async def delete_source(source_id: str):
    """Delete a source from the knowledge base."""
    try:
        brain = get_brain()
        
        # Delete from vector store
        deleted = brain.vector_store.delete_by_knowledge_item(source_id)
        
        # Delete from metadata store
        from sqlalchemy import text
        engine = brain.metadata_store._get_engine()
        
        with engine.connect() as conn:
            conn.execute(
                text("DELETE FROM knowledge_items WHERE id = :id"),
                {"id": source_id}
            )
            conn.commit()
            
        return {
            "status": "success",
            "deleted_id": source_id,
            "chunks_removed": deleted,
        }
        
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Run Server ==============

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("ENV", "development") == "development",
    )
