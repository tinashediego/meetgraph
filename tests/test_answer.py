from types import SimpleNamespace

from meetgraph.answer import answer


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_sends_context_and_returns_text():
    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text="Offline mode [Roadmap, 2026-09-02]")],
    )
    client = FakeClient(response)
    assert answer("What is the Q4 priority?", "## Roadmap", client) == "Offline mode [Roadmap, 2026-09-02]"
    request = client.calls[0]
    assert "## Roadmap" in request["messages"][0]["content"]
    assert request["fallbacks"] == "default"


def test_refusal_is_reported():
    client = FakeClient(SimpleNamespace(stop_reason="refusal", content=[]))
    assert "declined" in answer("q", "ctx", client)
