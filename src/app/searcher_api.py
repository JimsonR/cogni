from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import logging
import uvicorn
import concurrent.futures
from datetime import datetime
import tiktoken
import numpy as np
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from openai import AzureOpenAI
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AP Gov QA & Insights API",
    description="Unified API for semantic search of government Q&A documents and insights management",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
try:
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("OPENAI_API_VERSION", "2024-02-15-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_ENDPOINT")
    )
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or os.getenv("OPENAI_DEPLOYMENT_NAME")
    logger.info("Azure OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Azure OpenAI client: {e}")
    client = None
    deployment = None

# MongoDB Atlas configuration
MONGODB_CONNECTION_STRING = os.getenv("MONGODB_CONNECTION_STRING")
DATABASE_NAME = os.getenv("DATABASE_NAME", "crda")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "knowledge_bank")

# Initialize MongoDB client
try:
    mongo_client = MongoClient("mongodb+srv://cognivaultai_db_user:BFEpSuOj1fpjmhGn@cogni-vault.gd7fz06.mongodb.net/?retryWrites=true&w=majority&appName=cogni-vault")
    # Test connection
    mongo_client.admin.command('ping')
    logger.info("MongoDB Atlas connection established successfully")
    db = mongo_client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    
    # Setup insights collection for the new insights API
    crda_db = mongo_client.crda
    insights_collection = crda_db.insights
    
except Exception as e:
    logger.error(f"Failed to connect to MongoDB Atlas: {e}")
    mongo_client = None
    db = None
    collection = None
    insights_collection = None

class SearchRequest(BaseModel):
    """Request body model for search queries only (responses returned as plain dicts)."""
    # Keep config minimal; avoid altering protected_namespaces to prevent underscore errors
    model_config = {"extra": "ignore"}
    query: str = Field(..., description="Search query text", min_length=1)
    limit: int = Field(default=5, description="Number of results to return", ge=1, le=50)
    threshold: float = Field(default=0.7, description="Similarity threshold (0-1)", ge=0.0, le=1.0)
    include_embeddings: bool = Field(default=False, description="Include embeddings in response")


class Insight(BaseModel):
    """Pydantic model for insights data validation"""
    model_config = {"extra": "ignore"}
    insight: str
    summary_answer: List[str]
    follow_up_questions_and_answers: List[Dict[str, Any]]
    tags: List[str]
    ingestion_timestamp: Optional[datetime] = None


# Helper functions
def get_openai_embedding(text, timeout=15):
    """Get embeddings using Azure OpenAI's text-embedding model with context window truncation and timeout."""
    if client is None:
        raise Exception("OpenAI client not initialized. Check your API credentials.")
    
    # Truncate text to fit within model context window (e.g., 8000 tokens for text-embedding-3-small)
    max_tokens = 8000
    try:
        encoding = tiktoken.get_encoding("cl100k_base")  # Use explicit encoding for text-embedding-3-small
    except Exception:
        # Fallback to a simple character-based truncation if tiktoken fails
        if len(text) > max_tokens * 4:  # Rough approximation: 4 chars per token
            text = text[:max_tokens * 4]
    else:
        tokens = encoding.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            text = encoding.decode(tokens)
    
    def call():
        return client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(call)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.error(f"OpenAI embedding call timed out for text: {text[:50]}")
            raise TimeoutError("OpenAI embedding call timed out.")


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    
    dot_product = np.dot(vec1_np, vec2_np)
    norm1 = np.linalg.norm(vec1_np)
    norm2 = np.linalg.norm(vec2_np)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def search_documents(query_embedding: List[float], limit: int, threshold: float) -> List[Dict[str, Any]]:
    """Search documents using MongoDB Atlas Vector Search."""
    if collection is None:
        raise HTTPException(status_code=500, detail="MongoDB collection not available")
    
    try:
        # First, let's check if we have documents with embeddings
        sample_doc = collection.find_one({"question_embedding": {"$exists": True, "$ne": None}})
        if not sample_doc:
            logger.warning("No documents found with question_embedding field!")
            # Let's check what documents exist
            total_docs = collection.count_documents({})
            logger.info(f"Total documents in collection: {total_docs}")
            if total_docs > 0:
                sample = collection.find_one()
                logger.info(f"Sample document fields: {list(sample.keys()) if sample else 'None'}")
        else:
            logger.info(f"Found document with embedding. Sample fields: {list(sample_doc.keys())}")
            embedding_length = len(sample_doc.get("question_embedding", []))
            logger.info(f"Embedding vector length: {embedding_length}")
        
        # MongoDB Atlas Vector Search aggregation pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "queryVector": query_embedding,
                    "path": "question_embedding",
                    "numCandidates": 10000,
                    "index": "questions_index",
                    "limit": limit
                }
            },
            {
                "$addFields": {
                    "similarity_score": {"$meta": "vectorSearchScore"}
                }
            },
            {
                '$project': {
                    'user_question': 1,
                    'detailed_answer': 1,
                    'follow_up_questions.question': 1,
                    'similarity_score': 1,

                }
            }
        ]
        
        logger.info(f"Executing vector search with query embedding length: {len(query_embedding)}")
        raw = list(collection.aggregate(pipeline))
        logger.info(f"Vector search returned {len(raw)} raw results")
        
        # Debug: Log the first result to see what fields are available
        if raw:
            logger.info(f"Sample result keys: {list(raw[0].keys())}")
            if 'similarity_score' in raw[0]:
                logger.info(f"Sample similarity_score: {raw[0]['similarity_score']}")
        else:
            logger.warning("Vector search returned no results - this might indicate an index issue")

        # Filter by threshold and prepare results
        results = []
        for doc in raw:
            # Clean up any MongoDB-specific underscore fields that might cause Pydantic issues
            cleaned_doc = {}
            for key, value in doc.items():
                if not key.startswith('__'):  # Skip fields with double underscores
                    cleaned_doc[key] = value
            
            score = cleaned_doc.get("similarity_score")
            if score is not None and score >= threshold:
                results.append(cleaned_doc)
            elif threshold <= 0:
                # If threshold is 0 or negative, include all results
                cleaned_doc["similarity_score"] = score if score is not None else 0.0
                results.append(cleaned_doc)
        
        logger.info(f"After threshold filtering ({threshold}): {len(results)} results")
        return results
        
    except PyMongoError as e:
        logger.error(f"MongoDB Vector Search error: {e}")
        # Fallback to simple search if vector search fails
        logger.info("Falling back to simple similarity search...")
        return search_documents_fallback(query_embedding, limit, threshold)


def search_documents_fallback(query_embedding: List[float], limit: int, threshold: float) -> List[Dict[str, Any]]:
    """Fallback search using Python-based cosine similarity calculation."""
    try:
        # Simple MongoDB query to get documents with embeddings
        documents = list(collection.find(
            {"question_embedding": {"$exists": True, "$ne": None}},
            limit=limit * 3  # Get more documents to filter by threshold
        ))
        
        # Calculate similarity scores in Python
        scored_documents = []
        for doc in documents:
            if "question_embedding" in doc and doc["question_embedding"]:
                similarity = cosine_similarity(query_embedding, doc["question_embedding"])
                if similarity >= threshold:
                    # Clean up any problematic underscore fields
                    cleaned_doc = {}
                    for key, value in doc.items():
                        if not key.startswith('__'):  # Skip fields with double underscores
                            cleaned_doc[key] = value
                    cleaned_doc["similarity_score"] = similarity
                    scored_documents.append(cleaned_doc)
        
        # Sort by similarity score (highest first) and limit results
        scored_documents.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored_documents[:limit]
        
    except PyMongoError as e:
        logger.error(f"Fallback search error: {e}")
        raise HTTPException(status_code=500, detail="Database search failed")


@app.get("/health")
async def health_check():
    """Health check endpoint returning plain dict (no Pydantic response model)."""
    mongodb_connected = collection is not None
    openai_connected = client is not None
    collection_count = None
    if mongodb_connected:
        try:
            collection_count = collection.count_documents({})
        except Exception:
            mongodb_connected = False
    status = "healthy" if mongodb_connected and openai_connected else "degraded"
    return {
        "status": status,
        "mongodb_connected": mongodb_connected,
        "openai_connected": openai_connected,
        "collection_count": collection_count
    }


@app.get("/debug/collection-info")
async def debug_collection_info():
    """Debug endpoint to check collection state."""
    if collection is None:
        raise HTTPException(status_code=500, detail="MongoDB collection not available")
    
    try:
        # Count total documents
        total_docs = collection.count_documents({})
        
        # Count documents with embeddings
        docs_with_embeddings = collection.count_documents({"question_embedding": {"$exists": True, "$ne": None}})
        
        # Get a sample document
        sample_doc = collection.find_one()
        sample_fields = list(sample_doc.keys()) if sample_doc else []
        
        # Check if we have embedding field in sample
        has_embedding = "question_embedding" in sample_fields if sample_doc else False
        embedding_length = len(sample_doc.get("question_embedding", [])) if sample_doc and has_embedding else 0
        
        return {
            "total_documents": total_docs,
            "documents_with_embeddings": docs_with_embeddings,
            "sample_document_fields": sample_fields,
            "has_embedding_field": has_embedding,
            "embedding_vector_length": embedding_length,
            "database": DATABASE_NAME,
            "collection": COLLECTION_NAME
        }
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        raise HTTPException(status_code=500, detail=f"Error checking collection: {str(e)}")


@app.post("/search")
async def search_questions(request: SearchRequest) -> Dict[str, Any]:
    """Perform semantic vector search on the Q&A collection."""
    start_time = datetime.now()
    
    if collection is None:
        raise HTTPException(status_code=500, detail="MongoDB collection not available")
    
    if client is None:
        raise HTTPException(status_code=500, detail="OpenAI client not available")
    
    try:
        # Generate embedding for the query
        query_embedding = get_openai_embedding(request.query)
        
        # Search documents
        raw_results = search_documents(query_embedding, request.limit, request.threshold)
        
        # Format results as simple dictionaries (avoid Pydantic issues)
        results = []
        for doc in raw_results:
            try:
                # Handle different field naming conventions
                question = (
                    doc.get("question") or 
                    doc.get("user_question") or 
                    doc.get("summary_question", "")
                )
                answer = (
                    doc.get("answer") or 
                    doc.get("detailed_answer") or
                    doc.get("summary_answer", "")
                )
                
                # Debug logging to see what fields are available
                if not answer:
                    logger.warning(f"No answer found for document {doc.get('_id')}. Available fields: {list(doc.keys())}")
                
                # Safely extract follow-up questions, handling different formats
                # follow_ups = doc.get("follow_up_questions_and_answers")
                # if not follow_ups:
                follow_ups = doc.get("follow_up_questions")
                
                # Create result as plain dictionary with safe field handling
                result_data = {
                    "id": str(doc["_id"]),
                    "question": question,
                    "answer": answer,
                    "similarity_score": round(float(doc.get("similarity_score", 0.0)), 4),
                    "follow_up_questions": follow_ups,
                    "question_embedding": doc.get("question_embedding") if request.include_embeddings else None,
                    "created_at": str(doc.get("created_at")) if doc.get("created_at") else None,
                    "updated_at": str(doc.get("updated_at")) if doc.get("updated_at") else None
                }
                
                # Remove any MongoDB-specific fields that might cause issues
                for key in list(result_data.keys()):
                    if key.startswith('_') and key != 'id':
                        del result_data[key]
                
                results.append(result_data)
                
            except Exception as e:
                logger.error(f"Error formatting search result: {e}")
                logger.error(f"Document keys: {list(doc.keys())}")
                # Skip this result but continue with others
                continue
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds() * 1000
        
        # Return as plain dictionary to avoid Pydantic issues
        return {
            "results": results,
            "total_results": len(results),
            "query": request.query,
            "execution_time_ms": round(execution_time, 2)
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/test-embedding")
async def test_embedding(text: str = "test query"):
    """Test endpoint to isolate embedding generation issues."""
    try:
        logger.info(f"Testing embedding for text: {text}")
        embedding = get_openai_embedding(text)
        return {
            "status": "success",
            "text": text,
            "embedding_length": len(embedding),
            "embedding_sample": embedding[:5] if len(embedding) > 5 else embedding
        }
    except Exception as e:
        logger.error(f"Test embedding error: {e}")
        return {
            "status": "error",
            "text": text,
            "error": str(e),
            "error_type": str(type(e))
        }


@app.get("/search")
async def vector_search_get(
    q: Optional[str] = Query(None, description="Search query (alias of 'query')"),
    query: Optional[str] = Query(None, description="Search query (alternative param)"),
    limit: int = Query(default=5, description="Number of results", ge=1, le=50),
    threshold: float = Query(default=0.7, description="Similarity threshold", ge=0.0, le=1.0),
    include_embeddings: bool = Query(default=False, description="Include embeddings in response")
):
    """GET endpoint for vector search accepting either ?q= or ?query=."""
    final_query = q or query
    if not final_query:
        raise HTTPException(status_code=400, detail={
            "error": "Missing required query parameter",
            "message": "Provide either 'q' or 'query' parameter, e.g. /search?q=your+text",
            "examples": [
                "/search?q=what+is+federalism",
                "/search?query=separation+of+powers"
            ]
        })
    request = SearchRequest(
        query=final_query,
        limit=limit,
        threshold=threshold,
        include_embeddings=include_embeddings
    )
    return await search_questions(request)


# Insights API endpoints
@app.get("/insights/")
async def get_all_insights():
    """Get all insights from the database"""
    if insights_collection is None:
        raise HTTPException(status_code=500, detail="Insights collection not available")
    
    try:
        insights = list(insights_collection.find({}))
        # Convert ObjectId to string for JSON serialization
        for insight in insights:
            insight["_id"] = str(insight["_id"])
        return {"insights": insights, "count": len(insights)}
    except Exception as e:
        logger.error(f"Error fetching insights: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch insights")


@app.get("/insights/search/")
async def search_insights(query: str):
    """Search insights by text query using MongoDB text search"""
    if insights_collection is None:
        raise HTTPException(status_code=500, detail="Insights collection not available")
    
    try:
        # Use text search if available, otherwise regex search
        search_results = list(insights_collection.find({
            "$or": [
                {"insight": {"$regex": query, "$options": "i"}},
                {"summary_answer": {"$regex": query, "$options": "i"}},
                {"tags": {"$regex": query, "$options": "i"}}
            ]
        }))
        
        # Convert ObjectId to string
        for insight in search_results:
            insight["_id"] = str(insight["_id"])
            
        return {"insights": search_results, "count": len(search_results), "query": query}
    except Exception as e:
        logger.error(f"Error searching insights: {e}")
        raise HTTPException(status_code=500, detail="Failed to search insights")


@app.get("/insights/tags/{tag}")
async def get_insights_by_tag(tag: str):
    """Get insights by specific tag"""
    if insights_collection is None:
        raise HTTPException(status_code=500, detail="Insights collection not available")
    
    try:
        insights = list(insights_collection.find({"tags": {"$regex": tag, "$options": "i"}}))
        
        # Convert ObjectId to string
        for insight in insights:
            insight["_id"] = str(insight["_id"])
            
        return {"insights": insights, "count": len(insights), "tag": tag}
    except Exception as e:
        logger.error(f"Error fetching insights by tag: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch insights by tag")


@app.get("/insights/{insight_id}")
async def get_insight_by_id(insight_id: str):
    """Get a specific insight by its ID"""
    if insights_collection is None:
        raise HTTPException(status_code=500, detail="Insights collection not available")
    
    try:
        from bson import ObjectId
        insight = insights_collection.find_one({"_id": ObjectId(insight_id)})
        if insight:
            insight["_id"] = str(insight["_id"])
            return insight
        raise HTTPException(status_code=404, detail="Insight not found")
    except Exception as e:
        logger.error(f"Error fetching insight by ID: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/insights/latest/{limit}")
async def get_latest_insights(limit: int = 5):
    """Get the latest insights, limited by the specified number"""
    if insights_collection is None:
        raise HTTPException(status_code=500, detail="Insights collection not available")
    
    try:
        insights = list(insights_collection.find().sort("ingestion_timestamp", -1).limit(limit))
        
        # Convert ObjectId to string
        for insight in insights:
            insight["_id"] = str(insight["_id"])
            
        return {"insights": insights, "count": len(insights), "limit": limit}
    except Exception as e:
        logger.error(f"Error fetching latest insights: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch latest insights")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AP Gov QA Vector Search & Insights API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "search_post": "/search (POST)",
            "search_get": "/search?q=your_query",
            "insights_all": "/insights/",
            "insights_search": "/insights/search/?query=your_query",
            "insights_by_tag": "/insights/tags/{tag}",
            "insights_by_id": "/insights/{insight_id}",
            "insights_latest": "/insights/latest/{limit}",
            "docs": "/docs",
            "openapi": "/openapi.json"
        }
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, reload=True)
