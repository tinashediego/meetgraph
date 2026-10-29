"""Loading meeting records and splitting transcripts into chunks."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 400
CHUNK_OVERLAP = 60


@dataclass
class ActionItem:
    owner: str
    task: str
    due: str


@dataclass
class Meeting:
    id: str
    title: str
    date: str
    attendees: list[str]
    topics: list[str]
    decisions: list[str]
    action_items: list[ActionItem]
    transcript: str

    @classmethod
    def from_dict(cls, raw: dict) -> "Meeting":
        return cls(
            id=raw["id"],
            title=raw["title"],
            date=raw["date"],
            attendees=raw.get("attendees", []),
            topics=raw.get("topics", []),
            decisions=raw.get("decisions", []),
            action_items=[ActionItem(**item) for item in raw.get("action_items", [])],
            transcript=raw["transcript"],
        )


@dataclass
class Chunk:
    id: str
    meeting_id: str
    seq: int
    text: str
    embedding: list[float] = field(default_factory=list)


def load_meetings(data_dir: Path) -> list[Meeting]:
    paths = sorted(Path(data_dir).glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No meeting files (*.json) found in {data_dir}")
    return [Meeting.from_dict(json.loads(p.read_text())) for p in paths]


_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    # Prefer breaking between speaker turns, then sentences.
    separators=["\n", ". ", " ", ""],
)


def chunk_meeting(meeting: Meeting) -> list[Chunk]:
    """Split a transcript into overlapping chunks with stable ids, so re-ingesting is idempotent."""
    return [
        Chunk(id=f"{meeting.id}#{seq}", meeting_id=meeting.id, seq=seq, text=text)
        for seq, text in enumerate(_splitter.split_text(meeting.transcript))
    ]
