from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import os
import openai
from openai import OpenAI
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchAny, PayloadSchemaType
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Structured RAG Hotel Search API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from environment variables
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
STRUCTURED_COLLECTION_NAME = os.getenv("STRUCTURED_COLLECTION_NAME", "hotels_structured_rag")
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8001"))

# Initialize clients
qdrant_client = None
openai_client = None
hotels_data = []

def initialize_clients():
    """Initialize Qdrant and OpenAI clients from environment variables."""
    global qdrant_client, openai_client
    
    # Check if required environment variables are set
    if not QDRANT_URL or not QDRANT_API_KEY or not OPENAI_API_KEY:
        missing = []
        if not QDRANT_URL: missing.append("QDRANT_URL")
        if not QDRANT_API_KEY: missing.append("QDRANT_API_KEY")
        if not OPENAI_API_KEY: missing.append("OPENAI_API_KEY")
        
        print(f"Missing environment variables: {', '.join(missing)}")
        print("Please check your .env file and ensure all required variables are set.")
        return False
    
    try:
        # Initialize Qdrant client
        qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        
        # Test Qdrant connection
        collections = qdrant_client.get_collections()
        print(f"Successfully connected to Qdrant!")
        print(f"Available collections: {[col.name for col in collections.collections]}")
        
        # Initialize OpenAI client
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Test OpenAI connection
        test_embedding = get_embedding("test connection")
        if test_embedding:
            print("Successfully connected to OpenAI!")
        else:
            raise Exception("Failed to generate test embedding")
            
        return True
        
    except Exception as e:
        print(f"Failed to initialize clients: {e}")
        return False

# Pydantic models
class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    hotels: List[Dict[str, Any]]
    search_type: str
    query_analysis: Optional[Dict[str, Any]] = None

def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """Generate embeddings for given text using OpenAI's embedding model."""
    if not openai_client:
        return None
    
    try:
        text = text.replace("\n", " ")
        response = openai_client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def parse_query_to_structured(query: str) -> Optional[Dict]:
    """Parse natural language query into structured format using OpenAI."""
    if not openai_client:
        return None
    
    # Define the JSON schema for structured output
    schema = {
        "type": "object",
        "properties": {
            "region": {"type": ["string", "null"]},
            "country": {"type": ["string", "null"]},
            "type": {"type": ["string", "null"]},
            "atmosphere": {
                "type": ["string", "null"],
                "enum": [
                    "Luxury", "Historic", "Family-friendly", "Adventure", "Relaxing", "Business", "Romantic", "Vibrant", "Peaceful", "Rustic", "Tech", "Cultural", "Wellness", "Quirky", "Minimalist", "Party", "Traditional", "Modern", "Unique", "Tranquil", "Lively"
                ]
            },
            "activities": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "Swimming", "Hiking", "Skiing", "Diving", "Dancing", "Music", "Festivals", "Workshops", "Spa", "Cooking", "Sightseeing", "Shopping", "Coding", "Axe-throwing", "Yoga", "Meditation", "Wine tasting", "BBQ", "Samba", "Chocolate making", "Safari", "Beach", "Fitness", "Art", "Gaming", "Movie nights", "Milonga", "Yodeling", "Ski", "Golf", "Biking", "Relaxation", "Business", "Conference", "Family", "Tech", "Nature", "Culinary", "Cultural", "Dance", "Sports", "Music sessions"
                    ]
                }
            },
            "services": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "Spa", "Restaurant", "Pool", "WiFi", "Parking", "Bar", "Gym", "Conference room", "Breakfast", "Hammam", "Rooftop terrace", "DJ", "Soundproof rooms", "Coffee bar", "Mint tea", "Tagine", "Beach", "Fireplace", "Fiber optic internet", "Work pods", "Massage", "Workshops", "Beer garden", "Movie-themed suites", "Classic car service", "Onsen", "Tatami rooms", "Garden", "Wine cellar", "Bistro", "Ice lounge", "Casino", "Gelato shop", "Pizzeria", "Jazz bar", "Steakhouse", "BBQ", "Sub-zero lounge", "Conference facilities", "Meeting rooms", "Library", "Meditation garden", "Dance studios", "Cafe", "Soundproof work pods", "Red carpet entrance", "Saganaki", "Ouzo", "Animatronic animals", "Wedding venue", "Cognac collection", "Parisian-style bistro"
                    ]
                }
            },
            "best_months": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["region", "country", "type", "atmosphere", "activities", "services", "best_months"],
        "additionalProperties": False
    }
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are a hotel search query parser. Parse user queries into structured hotel criteria.

Return structured data with these fields:
- region: Geographic region (Europe, Asia, North America, South America, Africa, Oceania, Middle East)
- country: Specific country name
- type: Hotel type (Resort, Hotel, Lodge, Villa, Boutique, etc.)
- atmosphere: Atmosphere/vibe (Luxury, Romantic, Family-friendly, Adventure, Relaxing, etc.)
- activities: List of activities (Swimming, Hiking, Skiing, Diving, Gambling, etc.)
- services: List of services (Spa, casino, Restaurant, Pool, WiFi, etc.)
- best_months: List of best months to visit

Use null for fields that cannot be determined from the query.
Use empty arrays for list fields that cannot be determined."""
                },
                {
                    "role": "user",
                    "content": f"Parse this hotel search query: '{query}'"
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "hotel_query_structure",
                    "schema": schema,
                    "strict": True
                }
            },
            temperature=0
        )
        
        structured_query = json.loads(response.choices[0].message.content)
        return structured_query if isinstance(structured_query, dict) else None
        
    except Exception as e:
        print(f"Error parsing query: {e}")
        return None

def serialize_structured_data(structured_data: Dict) -> str:
    """Convert structured data into a labeled string for embedding."""
    parts = []
    
    if structured_data.get('region'):
        parts.append(f"Region: {structured_data['region']}")
    if structured_data.get('country'):
        parts.append(f"Country: {structured_data['country']}")
    if structured_data.get('type'):
        parts.append(f"Type: {structured_data['type']}")
    if structured_data.get('atmosphere'):
        parts.append(f"Atmosphere: {structured_data['atmosphere']}")
    if structured_data.get('activities'):
        activities = ', '.join(structured_data['activities'])
        parts.append(f"Activities: {activities}")
    if structured_data.get('services'):
        services = ', '.join(structured_data['services'])
        parts.append(f"Services: {services}")
    if structured_data.get('best_months'):
        months = ', '.join(structured_data['best_months'])
        parts.append(f"Best Months: {months}")
    
    return '. '.join(parts)

def create_filters_from_structure(structured_query: Dict) -> Optional[Filter]:
    """Create Qdrant filters from structured query data."""
    must_conditions = []
    should_conditions = []

    # Only region is required (must)
    if structured_query.get('region'):
        must_conditions.append(FieldCondition(key="region", match=MatchValue(value=structured_query['region'])))

    # Others are preferred (should)
    if structured_query.get('country'):
        should_conditions.append(FieldCondition(key="country", match=MatchValue(value=structured_query['country'])))

    if structured_query.get('type'):
        should_conditions.append(FieldCondition(key="type", match=MatchValue(value=structured_query['type'])))

    if structured_query.get('atmosphere'):
        should_conditions.append(FieldCondition(key="atmosphere", match=MatchValue(value=structured_query['atmosphere'])))

    if structured_query.get('activities'):
        should_conditions.append(FieldCondition(key="activities", match=MatchAny(any=structured_query['activities'])))

    if structured_query.get('services'):
        should_conditions.append(FieldCondition(key="services", match=MatchAny(any=structured_query['services'])))

    if structured_query.get('best_months') and structured_query['best_months'] != ['All']:
        should_conditions.append(FieldCondition(key="best_months", match=MatchAny(any=structured_query['best_months'])))

    if must_conditions or should_conditions:
        return Filter(
            must=must_conditions if must_conditions else None,
            should=should_conditions if should_conditions else None
        )
    else:
        return None

def traditional_rag_search(query: str, top_k: int = 5) -> List[Dict]:
    """Fallback to traditional RAG search using only description embeddings."""
    if not qdrant_client:
        return []
    
    try:
        desc_embedding = get_embedding(query)
        if not desc_embedding:
            return []
        
        search_results = qdrant_client.search(
            collection_name=STRUCTURED_COLLECTION_NAME,
            query_vector=("description", desc_embedding),
            limit=top_k,
            with_payload=True,
            score_threshold=0.0
        )
        
        results = []
        for result in search_results:
            hotel_info = {
                'id': result.id,
                'name': result.payload.get('name', f'Hotel {result.id}'),
                'description': result.payload.get('description', ''),
                'region': result.payload.get('region'),
                'country': result.payload.get('country'),
                'type': result.payload.get('type'),
                'atmosphere': result.payload.get('atmosphere'),
                'activities': result.payload.get('activities', []),
                'services': result.payload.get('services', []),
                'best_months': result.payload.get('best_months', []),
                'score': round(result.score, 4),
                'description_score': round(result.score, 4),
                'structure_score': 0.0
            }
            results.append(hotel_info)
        
        return results
        
    except Exception as e:
        print(f"Traditional search failed: {e}")
        return []

def structured_rag_search(query: str, top_k: int = 5) -> tuple[List[Dict], str, Optional[Dict]]:
    """Perform structured RAG search with fallback to traditional RAG."""
    if not qdrant_client:
        return [], "error", None
    
    # Step 1: Parse query to structured format
    structured_query = parse_query_to_structured(query)
    
    if structured_query:
        # Step 2: Create filters and embeddings
        filters = create_filters_from_structure(structured_query)
        serialized_structure = serialize_structured_data(structured_query)
        
        # Get embeddings
        desc_embedding = get_embedding(query)
        structure_embedding = get_embedding(serialized_structure)
        
        if desc_embedding and structure_embedding:
            try:
                # Step 3: Perform dual vector search with filters
                search_results_desc = qdrant_client.search(
                    collection_name=STRUCTURED_COLLECTION_NAME,
                    query_vector=("description", desc_embedding),
                    query_filter=filters,
                    limit=top_k * 2,
                    with_payload=True,
                    score_threshold=0.0
                )
                
                search_results_struct = qdrant_client.search(
                    collection_name=STRUCTURED_COLLECTION_NAME,
                    query_vector=("structure", structure_embedding),
                    query_filter=filters,
                    limit=top_k * 2,
                    with_payload=True,
                    score_threshold=0.0
                )
                
                # Step 4: Combine and rank results
                score_map = {}
                
                # Add description scores
                for result in search_results_desc:
                    score_map[result.id] = {
                        'description_score': result.score,
                        'structure_score': 0.0,
                        'payload': result.payload
                    }
                
                # Add structure scores
                for result in search_results_struct:
                    if result.id in score_map:
                        score_map[result.id]['structure_score'] = result.score
                    else:
                        score_map[result.id] = {
                            'description_score': 0.0,
                            'structure_score': result.score,
                            'payload': result.payload
                        }
                
                # Calculate combined scores and format results
                final_results = []
                for hotel_id, scores in score_map.items():
                    combined_score = (scores['description_score'] * 0.6) + (scores['structure_score'] * 0.4)
                    
                    hotel_info = {
                        'id': hotel_id,
                        'name': scores['payload'].get('name', f'Hotel {hotel_id}'),
                        'description': scores['payload'].get('description', ''),
                        'region': scores['payload'].get('region'),
                        'country': scores['payload'].get('country'),
                        'type': scores['payload'].get('type'),
                        'atmosphere': scores['payload'].get('atmosphere'),
                        'activities': scores['payload'].get('activities', []),
                        'services': scores['payload'].get('services', []),
                        'best_months': scores['payload'].get('best_months', []),
                        'score': round(combined_score, 4),
                        'description_score': round(scores['description_score'], 4),
                        'structure_score': round(scores['structure_score'], 4)
                    }
                    final_results.append(hotel_info)
                
                # Sort by combined score and return top results
                final_results.sort(key=lambda x: x['score'], reverse=True)
                return final_results[:top_k], "structured", structured_query
                
            except Exception as e:
                print(f"Structured search failed: {e}")
    
    # Fallback to traditional RAG
    return traditional_rag_search(query, top_k), "traditional", structured_query

def generate_response(query: str, hotels: List[Dict], search_type: str) -> str:
    """Generate a natural language response based on search results."""
    if not hotels:
        return "I couldn't find any hotels matching your criteria. Please try a different search."
    
    response_parts = []
    
    if search_type == "structured":
        response_parts.append("Based on your specific requirements, I found these hotels:")
    else:
        response_parts.append("Here are some hotels that match your search:")
    
    for i, hotel in enumerate(hotels[:3], 1):
        hotel_desc = f"\n{i}. **{hotel['name']}**"
        if hotel['country']:
            hotel_desc += f" in {hotel['country']}"
        if hotel['region']:
            hotel_desc += f" ({hotel['region']})"
        
        if hotel['type']:
            hotel_desc += f"\n   - Type: {hotel['type']}"
        if hotel['atmosphere']:
            hotel_desc += f"\n   - Atmosphere: {hotel['atmosphere']}"
        if hotel['activities']:
            hotel_desc += f"\n   - Activities: {', '.join(hotel['activities'][:3])}"
        if hotel['services']:
            hotel_desc += f"\n   - Services: {', '.join(hotel['services'][:3])}"
        
        hotel_desc += f"\n   - Match Score: {hotel['score']:.2f}"
        response_parts.append(hotel_desc)
    
    if len(hotels) > 3:
        response_parts.append(f"\n...and {len(hotels) - 3} more hotels found.")
    
    return "\n".join(response_parts)

# Define routes AFTER all functions are defined
@app.get("/")
async def root():
    """Root endpoint for testing."""
    print("DEBUG: Root endpoint called")
    return {"message": "Structured RAG Hotel Search API is running!", "status": "ok"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    print("DEBUG: Health endpoint called")
    return {
        "status": "healthy" if qdrant_client and openai_client else "not_configured",
        "qdrant_configured": qdrant_client is not None,
        "openai_configured": openai_client is not None,
        "hotels_loaded": len(hotels_data),
        "collection_name": STRUCTURED_COLLECTION_NAME
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Handle chat messages and return hotel search results."""
    if not qdrant_client or not openai_client:
        raise HTTPException(status_code=503, detail="Service not configured. Please check your .env file and restart the server.")
    
    try:
        # Perform structured RAG search
        hotels, search_type, query_analysis = structured_rag_search(message.message)
        
        # Generate natural language response
        response_text = generate_response(message.message, hotels, search_type)
        
        return ChatResponse(
            response=response_text,
            hotels=hotels,
            search_type=search_type,
            query_analysis=query_analysis
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load hotel data and initialize clients on startup"""
    global hotels_data
    
    print("Starting Structured RAG Hotel Search API...")
    # Initialize API clients
    if initialize_clients():
        print("Application startup completed successfully!")
        print(f"Server will be available at http://localhost:{BACKEND_PORT}")
        print("Frontend should be available at http://localhost:3000")
    else:
        print("Application startup completed with errors. Please check your .env configuration.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=BACKEND_HOST, port=BACKEND_PORT)
