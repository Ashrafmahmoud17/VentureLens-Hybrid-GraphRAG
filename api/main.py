from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.hybrid_retrieval import ask_groq


app = FastAPI(
    title="VentureLens API",
    description="Hybrid GraphRAG API for career and skill intelligence.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="User question",
    )
    top_k: int = Field(default=5, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    retrieval: dict[str, Any]


@app.get("/")
def root():
    return {
        "message": "Welcome to VentureLens API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "VentureLens API",
    }


@app.post("/api/v1/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    try:
        result = ask_groq(
            question=request.question,
            top_k=request.top_k,
        )

        return AskResponse(
            answer=result["answer"],
            retrieval=result["retrieval"],
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Question processing failed: {str(exc)}",
        ) from exc
