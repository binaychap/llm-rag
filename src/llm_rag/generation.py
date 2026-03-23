"""LLM-based response generation."""
from typing import List, Dict, Any
from openai import OpenAI
from .config import config


class ResponseGenerator:
    """Generate responses using LLM based on retrieved documents."""

    def __init__(self):
        """Initialize response generator."""
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.OPENAI_MODEL

    def generate(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        system_prompt: str = None
    ) -> str:
        """Generate response using query and context documents."""
        if system_prompt is None:
            system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
If you cannot find the answer in the context, say so clearly."""

        # Build context from documents
        context = "\n\n".join([
            f"Document {i+1}:\n{doc['text']}"
            for i, doc in enumerate(context_docs)
        ])

        user_message = f"""Context:
{context}

Question: {query}

Please answer the question based on the provided context."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        return response.choices[0].message.content

    def generate_with_history(
        self,
        messages: List[Dict[str, str]],
        context_docs: List[Dict[str, Any]] = None
    ) -> str:
        """Generate response with conversation history."""
        if context_docs:
            # Add context to system message
            context = "\n\n".join([
                f"Document {i+1}:\n{doc['text']}"
                for i, doc in enumerate(context_docs)
            ])
            
            system_message = f"""You are a helpful assistant. Use the following context to answer questions:

{context}

If you cannot find the answer in the context, say so clearly."""
        else:
            system_message = "You are a helpful assistant."

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )

        return response.choices[0].message.content