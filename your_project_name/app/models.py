from pydantic import BaseModel

class Query(BaseModel):
    text: str

class Response(BaseModel):
    response: str
