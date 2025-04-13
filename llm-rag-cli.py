import requests
import json
import os
from typing import List, Dict, Any


class LMStudioEmbeddings:
    """LM Studio embeddings wrapper."""

    def __init__(
        self,
        model: str = "text-embedding-medical-10-10-1-jinaai_jina-embeddings-v2-small-en-50-gpt-3.5-turbo-01_9062874564-i1",
        base_url: str = "http://127.0.0.1:1234"
    ):
        self.model = model
        self.base_url = base_url
        # Test connection to LM Studio
        try:
            self.embed_query("test")
            print("Successfully connected to LM Studio embeddings API")
        except Exception as e:
            print(f"Warning: Could not connect to LM Studio: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using LM Studio API."""
        embeddings = []
        # Process in batches to avoid overwhelming the API
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self._get_embeddings(batch)
            embeddings.extend(batch_embeddings)
            print(
                f"Processed {min(i+batch_size, len(texts))}/{len(texts)} documents")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using LM Studio API."""
        return self._get_embeddings([text])[0]

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from LM Studio API."""
        url = f"{self.base_url}/v1/embeddings"

        headers = {
            "Content-Type": "application/json"
        }

        # Handle both single string and list of strings
        if isinstance(texts, str):
            texts = [texts]

        data = {
            "model": self.model,
            "input": texts
        }

        try:
            response = requests.post(
                url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            result = response.json()
            # Extract the embedding data from the response
            embeddings = [item["embedding"] for item in result["data"]]
            return embeddings
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error getting embeddings from LM Studio: {e}")
        except (KeyError, IndexError) as e:
            raise ValueError(
                f"Error processing LM Studio response: {e}, Response: {response.text}")


def chat_with_lm_studio(messages, model="lmstudio-community/gemma-3-27b-it-GGUF/gemma-3-27b-it-Q4_K_M.gguf", max_tokens=2000, temperature=0.7):
    """Sends a list of messages to LM Studio's API and returns the response in Model Context Protocol format."""
    url = "http://localhost:1234/v1/chat/completions"

    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "messages": messages,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]
    except requests.exceptions.RequestException as e:
        return {"role": "assistant", "content": f"Error: {e}"}
    except (KeyError, IndexError) as e:
        return {"role": "assistant", "content": f"Error processing response: {e}, Response: {response.text}"}


def main():
    print("LMStudio RAG Chatbot")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.")

    conversation_history = []

    # Initial system message
    system_message = {
        "role": "system",
        "content": "You are a helpful assistant. Provide detailed and accurate responses to user questions."
    }
    conversation_history.append(system_message)

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            break

        # Add user message to conversation history
        user_message = {"role": "user", "content": user_input}
        conversation_history.append(user_message)

        # Get response from LM Studio
        response = chat_with_lm_studio(conversation_history)

        # Add assistant response to conversation history
        conversation_history.append(response)

        # Print the response
        print(f"Assistant: {response['content']}")


if __name__ == "__main__":
    main()
