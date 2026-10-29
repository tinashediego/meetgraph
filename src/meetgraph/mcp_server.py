"""MCP server exposing the meeting knowledge graph as tools.

The MCP client (e.g. Claude Desktop) is the LLM, so these tools return retrieved context and
graph facts rather than generated answers; no Anthropic API key is needed here.
"""

from functools import cache

from mcp.server.mcpserver import MCPServer

from . import graph, retriever
from .context import build_context
from .embeddings import LocalEmbeddings

mcp = MCPServer("meetgraph")


@cache
def _driver():
    return graph.connect()


@cache
def _store():
    return retriever.build_store(LocalEmbeddings())


@mcp.tool()
def search_meetings(query: str, k: int = 5) -> str:
    """Semantic search over meeting transcripts. Returns the best-matching excerpts grouped by
    meeting, with each meeting's date, attendees, decisions and action items from the graph."""
    return build_context(retriever.search(_store(), query, k=k))


@mcp.tool()
def list_meetings() -> list[dict]:
    """All meetings in date order, with attendees and topics."""
    return graph.list_meetings(_driver())


@mcp.tool()
def get_action_items(person: str | None = None, topic: str | None = None) -> list[dict]:
    """Action items, optionally filtered by owner name and/or meeting topic (partial, case-insensitive)."""
    return graph.action_items(_driver(), person=person, topic=topic)


@mcp.tool()
def topic_timeline(topic: str) -> list[dict]:
    """How a topic evolved: every meeting that discussed it, in date order, with its decisions."""
    return graph.topic_timeline(_driver(), topic)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
