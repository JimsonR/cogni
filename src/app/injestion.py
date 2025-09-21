import json
import logging
import concurrent.futures
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional
import tiktoken
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = None
try:
    # Basic client initialization without additional parameters that might cause issues
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    logger.info("Azure OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Azure OpenAI client: {e}")
    logger.error("Please check your OpenAI library version and API credentials")

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
except Exception as e:
    logger.error(f"Failed to connect to MongoDB Atlas: {e}")
    mongo_client = None
    db = None
    collection = None


# Helper functions for embeddings and similarity
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


def create_embeddings_for_document(document: Dict[str, Any]) -> Dict[str, Any]:
    """Create embeddings for the user_question field only."""
    try:
        # Use user_question as the primary field for embeddings
        question_text = document.get("user_question", "")
        
        # Clean and validate question text
        if question_text:
            question_text = question_text.strip()
            if question_text:  # Only proceed if we have valid text after stripping
                document["question_embedding"] = get_openai_embedding(question_text)
                logger.info(f"Created question embedding for: {question_text[:50]}...")
            else:
                logger.warning("Empty question text after cleaning, skipping embedding generation")
        else:
            logger.warning("No question text found, skipping embedding generation")
        
        # Process follow-up questions for the new schema
        if "follow_up_questions" in document and isinstance(document["follow_up_questions"], list):
            for follow_up in document["follow_up_questions"]:
                if isinstance(follow_up, dict) and "question" in follow_up:
                    follow_up_text = follow_up["question"].strip() if follow_up["question"] else ""
                    if follow_up_text:
                        try:
                            follow_up["question_embedding"] = get_openai_embedding(follow_up_text)
                            logger.info(f"Created follow-up embedding for: {follow_up_text[:50]}...")
                        except Exception as e:
                            logger.warning(f"Failed to create embedding for follow-up question: {e}")
        
        # Add metadata
        document["embedding_created_at"] = datetime.utcnow()
        document["embedding_model"] = "text-embedding-3-small"
        
        return document
        
    except Exception as e:
        # Use user_question for error logging
        question_ref = document.get("user_question", "") or document.get("question_id", "Unknown")
        logger.error(f"Failed to create embeddings for document {str(question_ref)[:50]}: {e}")
        return document


def upsert_document_to_mongodb(document: Dict[str, Any]) -> bool:
    """Upsert a single document to MongoDB Atlas. Uses auto-generated question_id as primary key."""
    if collection is None:
        logger.error("MongoDB collection not initialized")
        return False
    
    try:
        # Use question_id as the primary key (auto-generated)
        question_id = document.get("question_id")
        user_question = document.get("user_question", "")
        
        # Clean up question text - remove null values and strip whitespace
        if user_question:
            user_question = user_question.strip()
            if user_question:
                document["user_question"] = user_question
        
        # Always use question_id as the unique key since it's auto-generated
        if question_id:
            unique_key = {"question_id": question_id}
            identifier = f"question_id: {question_id}"
        else:
            logger.warning("Document missing question_id, skipping...")
            return False
        
        # Add metadata
        document["updated_at"] = datetime.utcnow()
        if "created_at" not in document:
            document["created_at"] = datetime.utcnow()
        
        # Perform upsert operation
        result = collection.update_one(
            unique_key,
            {"$set": document},
            upsert=True
        )
        
        if result.upserted_id:
            logger.info(f"Inserted new document with {identifier}")
        elif result.modified_count > 0:
            logger.info(f"Updated existing document with {identifier}")
        else:
            logger.info(f"No changes needed for document with {identifier}")
        
        return True
        
    except PyMongoError as e:
        question_ref = document.get("question_id", "Unknown")
        logger.error(f"MongoDB error upserting document {question_ref}: {e}")
        return False
    except Exception as e:
        question_ref = document.get("question_id", "Unknown")
        logger.error(f"Unexpected error upserting document {question_ref}: {e}")
        return False


def process_and_upsert_batch(documents: List[Dict[str, Any]], batch_size: int = 10) -> Dict[str, int]:
    """Process documents in batches and upsert to MongoDB with embeddings. Auto-increments question IDs."""
    results = {"success": 0, "failed": 0, "total": len(documents)}
    
    logger.info(f"Starting to process {len(documents)} documents in batches of {batch_size}")
    
    # Auto-increment question ID counter
    question_id_counter = 1
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
        
        for document in batch:
            try:
                # Remove existing question_id and assign auto-incremented one
                document_copy = document.copy()
                document_copy["question_id"] = f"Q{question_id_counter}"
                question_id_counter += 1
                
                # Create embeddings
                document_with_embeddings = create_embeddings_for_document(document_copy)
                
                # Upsert to MongoDB
                if upsert_document_to_mongodb(document_with_embeddings):
                    results["success"] += 1
                else:
                    results["failed"] += 1
                    
            except Exception as e:
                # Use user_question for error logging
                doc_ref = document.get("user_question", "") or f"Q{question_id_counter-1}"
                logger.error(f"Failed to process document {str(doc_ref)[:50]}: {e}")
                results["failed"] += 1
    
    logger.info(f"Batch processing completed. Success: {results['success']}, Failed: {results['failed']}")
    return results


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON data from file, handling multiple JSON arrays."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Split content by empty lines and try to parse each section as JSON
        all_documents = []
        json_sections = content.split('\n\n\n')
        
        for i, section in enumerate(json_sections):
            section = section.strip()
            if not section:
                continue
                
            try:
                # Try to parse this section as JSON
                data = json.loads(section)
                if isinstance(data, list):
                    all_documents.extend(data)
                    logger.info(f"Loaded {len(data)} documents from section {i+1}")
                elif isinstance(data, dict):
                    all_documents.append(data)
                    logger.info(f"Loaded 1 document from section {i+1}")
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON section {i+1}: {e}")
                continue
        
        logger.info(f"Total loaded {len(all_documents)} documents from {file_path}")
        return all_documents
            
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        return []


def create_mongodb_indexes():
    """Create indexes for optimal query performance."""
    if collection is None:
        logger.error("MongoDB collection not initialized")
        return
    
    try:
        # Drop existing problematic indexes if they exist
        try:
            collection.drop_index("question_1")
            logger.info("Dropped existing question index")
        except PyMongoError:
            pass  # Index might not exist
        
        # Create text index for full-text search - updated for normalized schema
        collection.create_index([
            ("user_question", "text"),
            ("answer", "text"),
            ("follow_up_questions.question", "text")
        ])
        logger.info("Created text search index")
        
        # Create unique index on question_id (auto-generated, so should be unique)
        collection.create_index("question_id", unique=True)
        logger.info("Created unique index on question_id")
        
        # Create index on created_at for temporal queries
        collection.create_index("created_at")
        logger.info("Created index on created_at")
        
        # Create vector search index for embeddings (if supported)
        try:
            collection.create_index([("question_embedding", "2dsphere")])
            logger.info("Created vector index on question_embedding")
        except PyMongoError as e:
            logger.warning(f"Could not create vector index (may require Atlas setup): {e}")
        
        logger.info("All indexes created successfully")
        
    except PyMongoError as e:
        logger.error(f"Error creating indexes: {e}")


class MongoDBIngestion:
    """MongoDB ingestion class for insights data"""
    def __init__(self, connection_string: str):
        """Initialize MongoDB connection"""
        self.client = MongoClient(connection_string)
        self.db = self.client.crda
        self.collection = self.db.insights

    def ingest_insights(self, insights: List[Dict]) -> None:
        """Ingest insights into MongoDB"""
        try:
            # Add timestamp to each insight
            for insight in insights:
                insight['ingestion_timestamp'] = datetime.utcnow()
            
            # Insert insights
            result = self.collection.insert_many(insights)
            print(f"Successfully inserted {len(result.inserted_ids)} insights")
            
        except Exception as e:
            print(f"Error ingesting insights: {str(e)}")

    def close(self):
        """Close MongoDB connection"""
        self.client.close()


def ingest_insights_from_file(insights_file_path: str = "insights.json"):
    """Standalone function to ingest insights from JSON file"""
    # MongoDB Atlas connection string
    connection_string = "mongodb+srv://jyothika:Jyothika%40123@cluster.ollkbh1.mongodb.net/?retryWrites=true&w=majority&appName=Cluster"
    
    try:
        # Load insights from JSON file
        with open(insights_file_path, 'r') as f:
            insights = json.load(f)
        
        # Initialize MongoDB ingestion
        mongo_ingestion = MongoDBIngestion(connection_string)
        
        # Ingest insights
        mongo_ingestion.ingest_insights(insights)
        
        # Close connection
        mongo_ingestion.close()
        
    except FileNotFoundError:
        print(f"Error: {insights_file_path} file not found")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {insights_file_path}")
    except Exception as e:
        print(f"Error: {str(e)}")


def main():
    """Main function to run the ingestion process."""
    # Check if required services are available
    if client is None:
        logger.error("OpenAI client not available. Cannot proceed without embeddings.")
        return
    
    if collection is None:
        logger.error("MongoDB collection not available. Cannot proceed.")
        return
    
    # Create indexes
    create_mongodb_indexes()
    
    # Load data
    data_file_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "data-2.json")
    documents = load_json_data(data_file_path)
    
    if not documents:
        logger.error("No documents to process")
        return
    
    # Process and upsert documents
    results = process_and_upsert_batch(documents, batch_size=5)  # Smaller batch size since we're creating embeddings for main questions and follow-ups
    
    logger.info(f"Ingestion completed. Total: {results['total']}, Success: {results['success']}, Failed: {results['failed']}")
    
    # Close MongoDB connection
    if mongo_client:
        mongo_client.close()
        logger.info("MongoDB connection closed")


if __name__ == "__main__":
    # Check command line arguments for different ingestion modes
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "insights":
            # Ingest insights mode
            insights_file = sys.argv[2] if len(sys.argv) > 2 else "insights.json"
            print(f"Running insights ingestion from {insights_file}")
            ingest_insights_from_file(insights_file)
        elif mode == "qa" or mode == "questions":
            # QA ingestion mode (default)
            print("Running Q&A document ingestion with embeddings")
            main()
        else:
            print("Usage:")
            print("  python injestion.py                    # Run Q&A ingestion (default)")
            print("  python injestion.py qa                 # Run Q&A ingestion")
            print("  python injestion.py insights [file]    # Run insights ingestion")
            print("                                         # file defaults to 'insights.json'")
    else:
        # Default mode: Q&A ingestion
        print("Running Q&A document ingestion with embeddings (default mode)")
        main()
