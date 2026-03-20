"""
Generation engine using Groq API.
"""

import logging
import os
from groq import Groq

logger = logging.getLogger(__name__)


class GenerationEngine:

    def __init__(self, api_key: str, model_name: str = "llama-3.1-8b-instant"):
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        logger.info(f"GenerationEngine initialized with model: {model_name}")

    def generate_answer(self, query: str, context: str) -> str:
        if not context.strip():
            return "No relevant content found in the documents."

        prompt = f"""You are a precise document analysis assistant.

Answer the question using ONLY the context provided below.
Do not use any outside knowledge.
If the answer is not clearly present in the context, say:
"The documents do not contain a clear answer to this question."
Be concise and direct. Cite which document the answer came from if possible.

---CONTEXT START---
{context}
---CONTEXT END---

Question: {query}

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return f"Generation failed: {str(e)}"
