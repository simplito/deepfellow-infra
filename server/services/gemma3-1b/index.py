from server.ollama import ollama, OllamaOptions

service = ollama(OllamaOptions(model="gemma3:1b"))
