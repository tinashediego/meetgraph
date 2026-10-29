# MeetGraph

**GraphRAG over meeting transcripts** with Neo4j, LangChain, Claude, and a Model Context Protocol (MCP) server.

Ask *"How are we handling conflicting edits in offline mode?"* and get an answer grounded in the right meetings, with decisions, owners and due dates pulled from a knowledge graph rather than guessed from text.

## Why GraphRAG?

Plain RAG embeds text chunks and returns the closest ones. That works for "what was said", but meeting questions are often about **structure**: who owns what, what was decided, how a topic changed over time. That information is scattered across chunks or never stated in one place.

MeetGraph stores both:

```
(Person)-[:ATTENDED]->(Meeting)-[:DISCUSSED]->(Topic)
(Meeting)-[:DECIDED]->(Decision)
(Meeting)-[:CREATED]->(ActionItem)-[:ASSIGNED_TO]->(Person)
(Chunk {text, embedding})-[:PART_OF]->(Meeting)
```

Retrieval runs in two steps inside a single Cypher query:

1. **Vector search** over `Chunk.embedding` (a Neo4j vector index) finds the relevant transcript excerpts.
2. **Graph expansion** walks from each chunk to its `Meeting` and collects the attendees, decisions and action items.

The answer can then cite facts like *"Decision: per-field last-writer-wins"* even when the matching excerpt only discusses the options.

## Stack

| Piece | Choice |
|---|---|
| Graph + vector store | Neo4j 5 (vector index), via Docker |
| Retrieval | LangChain `Neo4jVector` with a custom graph-expansion `retrieval_query` |
| Chunking | LangChain `RecursiveCharacterTextSplitter`, split on speaker turns |
| Embeddings | `BAAI/bge-small-en-v1.5` run locally with FastEmbed (no API key) |
| Answers | Claude via the official Anthropic Python SDK |
| Tools for AI clients | MCP server (official Python SDK) |

## Quick start

```bash
docker compose up -d                       # Neo4j at http://localhost:7474 (neo4j / meetgraph-dev)
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env                       # add ANTHROPIC_API_KEY to use `ask`

meetgraph ingest                           # 4 sample meetings -> graph + embeddings
meetgraph search "conflicting edits on two devices"   # see the retrieved GraphRAG context
meetgraph ask "What did we decide about offline conflicts, and who is doing what?"
meetgraph actions --person marcus --topic offline
meetgraph timeline "offline mode"
```

`search`, `actions`, `timeline` and the MCP server work without an API key. Only `ask` calls Claude.

## Use it from Claude Desktop (MCP)

Add this to Claude Desktop's `claude_desktop_config.json`, using absolute paths:

```json
{
  "mcpServers": {
    "meetgraph": {
      "command": "/path/to/meetgraph/.venv/bin/python",
      "args": ["-m", "meetgraph.mcp_server"]
    }
  }
}
```

Tools exposed:

| Tool | What it does |
|---|---|
| `search_meetings(query, k)` | GraphRAG search: excerpts plus each meeting's decisions and action items |
| `list_meetings()` | All meetings with attendees and topics |
| `get_action_items(person?, topic?)` | Action items filtered by owner and/or meeting topic |
| `topic_timeline(topic)` | How a topic evolved across meetings, with decisions |

The MCP client is already an LLM, so these tools return retrieved context instead of generated answers.

## Tests

```bash
pytest
```

Unit tests cover chunking, context building and the Claude call (with a fake client). The integration tests ingest the sample data into Neo4j and check that semantic search lands on the right meeting and pulls in graph-only facts. They are skipped when Neo4j isn't running.

## Project layout

```
src/meetgraph/
  meetings.py    load meeting JSON, chunk transcripts
  embeddings.py  local FastEmbed embeddings (LangChain interface)
  graph.py       Neo4j schema, ingestion, graph queries
  retriever.py   vector search + graph expansion (LangChain Neo4jVector)
  context.py     group hits by meeting into citable context
  answer.py      grounded answers from Claude
  mcp_server.py  MCP tools
  cli.py         command-line interface
data/meetings/   sample meetings (fictional team)
```

## Bring your own meetings

Each meeting is one JSON file in `data/meetings/`:

```json
{
  "id": "2026-09-10-offline-design-review",
  "title": "Offline Mode Design Review",
  "date": "2026-09-10",
  "attendees": ["Priya Nair", "Sam Okafor"],
  "topics": ["Offline mode", "Conflict resolution"],
  "decisions": ["Use last-writer-wins per field"],
  "action_items": [{"owner": "Sam Okafor", "task": "Prototype SQLite storage", "due": "2026-09-24"}],
  "transcript": "Priya Nair: ...\nSam Okafor: ..."
}
```

Re-running `meetgraph ingest` is idempotent: nodes are merged on stable ids.

## Next steps

- Extract decisions and action items from raw transcripts with Claude structured outputs, so plain transcript files work too.
- Add a small React/TypeScript chat UI with citations.
- Add an evaluation set of questions with expected meetings and facts, to compare plain RAG against GraphRAG.
