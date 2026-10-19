"""
Start the real web UI (app.py under Chainlit) with a scripted model, for the browser tests.

Owns: pointing Mackie's data and checkpoints at a scratch folder and replacing
every LLM call with a script, so the browser tests drive the real server and
frontend without API calls or real org data. Does not own the tests
(test_web_ui.py).

Usage: python tests/browser/serve.py DATA_DIR PORT

The scripted model, by the user's last message:
- "save <title>": calls save_post, then says it saved the post.
- "stream slowly": sends one sentence, then the next only once DATA_DIR/release exists.
- anything else: echoes the message and how many user messages it can see,
  which shows whether another session's history leaked in.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

REPO_ROOT: Path = Path(__file__).parent.parent.parent
RELEASE_FILE_NAME: str = "release"
FIRST_SENTENCE: str = "First sentence, streamed straight away."
SECOND_SENTENCE: str = " Second sentence, after the release."
# Bounds the gated stream so a test that never releases it can't hang the server.
RELEASE_WAIT_S: float = 30.0
RELEASE_POLL_S: float = 0.05


async def _wait_for_release(data_dir: Path) -> None:
    for _ in range(int(RELEASE_WAIT_S / RELEASE_POLL_S)):
        if (data_dir / RELEASE_FILE_NAME).exists():
            return
        await asyncio.sleep(RELEASE_POLL_S)


def _scripted_model(data_dir: Path):
    """An acompletion stand-in that answers each session from its own messages."""
    from tests.test_assistant import _text_stream, _tool_call_stream

    async def gated_stream():
        async for chunk in _text_stream(FIRST_SENTENCE):
            if chunk.choices[0].finish_reason:
                break
            yield chunk
        await _wait_for_release(data_dir)
        async for chunk in _text_stream(SECOND_SENTENCE):
            yield chunk

    async def model(**kwargs):
        messages = kwargs["messages"]
        if messages[-1]["role"] == "tool":
            return _text_stream("Saved it to the ledger.")
        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        said = user_messages[-1]
        if said.startswith("save "):
            post = {"platform": "general", "content": said, "title": said.removeprefix("save ")}
            return _tool_call_stream("tc-save", "save_post", json.dumps(post))
        if said == "stream slowly":
            return gated_stream()
        return _text_stream(f"You said: {said}. Messages I can see: {len(user_messages)}.")

    return model


def main() -> None:
    """Patch Mackie's storage and model, then run Chainlit headless on PORT."""
    data_dir, port = Path(sys.argv[1]), sys.argv[2]
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))

    import agent.graph as graph
    import agent.memory as mem
    import agent.tools as tools

    mem.DATA_DIR = data_dir
    graph.CHECKPOINTS_DB = data_dir / "checkpoints.db"
    graph.litellm.acompletion = tools.litellm.acompletion = _scripted_model(data_dir)

    os.environ.update(CHAINLIT_HOST="127.0.0.1", CHAINLIT_PORT=port)
    from chainlit.cli import run_chainlit
    from chainlit.config import config

    config.run.headless = True
    run_chainlit(str(REPO_ROOT / "app.py"))


if __name__ == "__main__":
    main()
