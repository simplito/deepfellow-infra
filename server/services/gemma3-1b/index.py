"""Gemma service."""

from server.ollama import OllamaOptions, ollama

service = ollama(OllamaOptions(model="gemma3:1b"))
