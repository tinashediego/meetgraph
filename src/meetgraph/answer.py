"""Grounded answers from Claude over GraphRAG context."""

import anthropic

from .config import settings

SYSTEM_PROMPT = """You answer questions about a team's meetings using only the meeting records provided.
Cite the meeting for each claim as [Meeting title, YYYY-MM-DD].
If the records don't contain the answer, say so plainly rather than guessing."""


def answer(question: str, context: str, client: anthropic.Anthropic | None = None) -> str:
    client = client or anthropic.Anthropic()
    response = client.beta.messages.create(
        model=settings.claude_model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"<meeting_records>\n{context}\n</meeting_records>\n\nQuestion: {question}",
        }],
        output_config={"effort": "medium"},
        # If a safety classifier declines, the server retries on a fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        return "The model declined to answer this question."
    return "".join(block.text for block in response.content if block.type == "text")
