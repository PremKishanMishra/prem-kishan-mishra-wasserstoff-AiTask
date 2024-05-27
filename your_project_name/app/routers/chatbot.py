from fastapi import APIRouter, HTTPException
from app.models import Query, Response
from services.data_fetcher import fetch_posts
from services.embedding import generate_embeddings
from services.faiss_index import add_embeddings_to_index

router = APIRouter()

@router.post("/chat/", response_model=Response)
def chat(query: Query):
    try:
        posts = fetch_posts()
        for post in posts["data"]["posts"]["nodes"]:
            post_id = post["id"]
            post_content = post["content"]
            embeddings = generate_embeddings(post_content)
            add_embeddings_to_index(post_id, embeddings)
        
        response_text = "Responding to your query: " + query.text
        return Response(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
