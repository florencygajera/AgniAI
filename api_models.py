"""
api_models.py
=============
Shared JSON response shapes for AgniAI's REST API.

Every route in app.py uses these helpers so the .NET ChatController.cs
always receives a predictable, consistent JSON structure — no surprises
during deserialization on their side.

.NET team: map these to C# record/class with these exact field names.
"""

from __future__ import annotations
from typing import Any, Optional


def ok_chat(
    answer: str,
    style: str,
    session_id: Optional[str] = None,
) -> dict:
    """
    Successful chat response.

    C# mapping:
        public record ChatResponse(
            string Answer,
            string Style,
            string? SessionId
        );
    """
    payload: dict[str, Any] = {
        "success": True,
        "answer": answer,
        "style": style,        # "short" | "elaborate" | "detail"
    }
    if session_id:
        payload["session_id"] = session_id
    return payload


def ok_ingest(message: str, chunks: int, source: str) -> dict:
    """
    Successful ingest response.

    C# mapping:
        public record IngestResponse(
            bool Success,
            string Message,
            int Chunks,
            string Source
        );
    """
    return {
        "success": True,
        "message": message,
        "chunks": chunks,
        "source": source,
    }


def ok_health(vectors: int, chunks: int, model: str, status: str = "ok") -> dict:
    """
    Health-check response — used by .NET to verify the Python service is up.

    C# mapping:
        public record HealthResponse(
            string Status,
            int Vectors,
            int Chunks,
            string Model
        );
    """
    return {
        "status": status,
        "vectors": vectors,
        "chunks": chunks,
        "model": model,
    }


def err(message: str, code: int = 400) -> tuple[dict, int]:
    """
    Error response — always the same shape so C# never hits a
    deserialization exception on error paths.

    C# mapping:
        public record ErrorResponse(bool Success, string Error);
    """
    return {"success": False, "error": message}, code


def ok_sources(sources: list) -> dict:
    """
    List of ingested sources.
    """
    return {
        "success": True,
        "count": len(sources),
        "sources": sources,
    }


def ok_stats(vectors: int, chunks: int) -> dict:
    return {
        "success": True,
        "vectors": vectors,
        "chunks": chunks,
    }


def ok_message(message: str) -> dict:
    return {
        "success": True,
        "message": message,
    }