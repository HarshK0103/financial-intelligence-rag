"""Inference backends."""

from app.inference.inference_engine import InferenceEngine
from app.inference.ollama_inference_engine import OllamaInferenceEngine

TemplateInferenceEngine = InferenceEngine

__all__ = [
    "InferenceEngine",
    "OllamaInferenceEngine",
    "TemplateInferenceEngine",
]
