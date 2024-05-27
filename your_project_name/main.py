from fastapi import FastAPI
from app.routers import chatbot

app = FastAPI()

app.include_router(chatbot.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
