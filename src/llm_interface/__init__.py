"""
LLM Client Interface

Provider-agnostic abstractions for interacting with LLMs.

Usage:
    from src.llm_interface import LLMClient, LLMResponse, ImagePart, GenerateConfig
    from src.llm_interface import GoogleGenAIClient

    llm = GoogleGenAIClient(model="gemini-2.0-flash", api_key="...")
    # or locally:
    # llm = HuggingFaceClient(model="Qwen/Qwen3-VL-4B-Thinking")
    response = llm.generate("What is 2+2?")
    print(response.text)
"""

from src.llm_interface.base import (
    LLMClient,
    LLMResponse,
    ImagePart,
    GenerateConfig,
    ContentPart,
)
from src.llm_interface.google_genai import GoogleGenAIClient
from src.llm_interface.qwen3vl import Qwen3VLClient

__all__ = [
    "LLMClient",
    "LLMResponse",
    "ImagePart",
    "GenerateConfig",
    "ContentPart",
    "GoogleGenAIClient",
    "Qwen3VLClient",
]
