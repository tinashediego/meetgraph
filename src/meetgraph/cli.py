import argparse
import json
from pathlib import Path

from . import graph, retriever
from .config import DEFAULT_DATA_DIR
from .context import build_context
from .embeddings import LocalEmbeddings
from .meetings import load_meetings


def main() -> None:
    parser = argparse.ArgumentParser(prog="meetgraph", description="GraphRAG over meeting transcripts")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Load meetings into Neo4j")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)

    p = sub.add_parser("search", help="Show the GraphRAG context retrieved for a query")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)

    p = sub.add_parser("ask", help="Answer a question with Claude (needs ANTHROPIC_API_KEY)")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=5)

    p = sub.add_parser("actions", help="List action items")
    p.add_argument("--person")
    p.add_argument("--topic")

    p = sub.add_parser("timeline", help="Show how a topic evolved across meetings")
    p.add_argument("topic")

    args = parser.parse_args()

    if args.command == "ingest":
        meetings = load_meetings(args.data_dir)
        with graph.connect() as driver:
            chunks = graph.ingest(driver, meetings, LocalEmbeddings())
        print(f"Ingested {len(meetings)} meetings ({chunks} chunks).")
    elif args.command in ("search", "ask"):
        store = retriever.build_store(LocalEmbeddings())
        query = args.query if args.command == "search" else args.question
        context = build_context(retriever.search(store, query, k=args.k))
        if args.command == "search":
            print(context)
        else:
            from .answer import answer

            print(answer(args.question, context))
    elif args.command == "actions":
        with graph.connect() as driver:
            print(json.dumps(graph.action_items(driver, args.person, args.topic), indent=2))
    elif args.command == "timeline":
        with graph.connect() as driver:
            print(json.dumps(graph.topic_timeline(driver, args.topic), indent=2))


if __name__ == "__main__":
    main()
