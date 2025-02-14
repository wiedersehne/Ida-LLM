from fastapi import FastAPI
from pydantic import BaseModel
from backend.models.llm import get_llm
from backend.agents.pdf_retriever import BaseRetriever

app = FastAPI()

class Request(BaseModel):
    query : str

class Response(BaseModel):
    response : str

@app.post("/",response_model=Response)
async def predict_api(query:Request):
    retriever = BaseRetriever()
    response = retriever.run(Request.query)
    return response