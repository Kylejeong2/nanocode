# nanocode-memory

A small extension of [1rgs/nanocode](https://github.com/1rgs/nanocode): compact the working conversation, then grep the original history when the summary lacks detail. Python standard library only.

Based on upstream commit `b009d3dbedf14795a5c10804a5455386563f4b5b`. The original coding tools are retained; all model requests use OpenRouter. Upstream declares MIT in its README.

## Run

Create a `.env` file beside `nanocode.py`:

```dotenv
OPENROUTER_API_KEY=your-key
MODEL=anthropic/claude-opus-4.5
TYPESAFE_API_KEY=your-jev-key
# JEV_MODEL=jev-latest
```

Then run:

```sh
python3 nanocode.py
```

`.env` loads automatically at startup. Exported environment variables take precedence, followed by the working directory's `.env`, then the `.env` beside the script. Single-line assignments, quoted values, comments, and optional `export` prefixes are supported; shell commands and variable interpolation are not executed. Restart the agent after changing `.env`.

Set `MODEL` to an OpenRouter model ID available to your account (default: `anthropic/claude-opus-4.5`). `OPENROUTER_API_KEY` is required for agent requests. `TYPESAFE_API_KEY` optionally enables Jev memory classification; without it, compaction falls back to the OpenRouter summary path. `JEV_MODEL` is optional and defaults to `jev-latest`.

```sh
python3 nanocode.py --compact-at 24000
python3 nanocode.py --session "memory/<session-id>"
```

Each session ID combines a timestamp and a random suffix. Its folder contains `history.md` and, after the first message, `state.json`. The `memory/` folder is ignored by Git. Existing sessions in `~/.nanocode/sessions/` can still be resumed by passing their full path to `--session`.

Commands: `/compact` forces a checkpoint, `/history` prints the transcript path, `/c` starts a fresh session without deleting the old one, `/q` exits.

Responses stream directly into the terminal as text arrives. Tool arguments are assembled before execution; completed responses are saved to history once. Compaction runs without streaming its internal summary. If a stream fails, partial text may remain visible, but the incomplete response is not saved or executed.

## What happens

1. Each user message, assistant response, tool call, and tool result is written to `history.md` in a private session directory under `memory/<session-id>/` beside `nanocode.py`.
2. Before each agent request, the working conversation size is estimated. At the threshold, Jev classifies older tool calls and their results for durable value.
3. Jev keeps valuable calls and full results, truncates results whose call matters but whose output does not, and drops calls that are not needed. The first message and recent messages are pinned. The full original transcript remains searchable in `history.md`.
4. The agent uses the compacted memory first. For missing specifics, `history_search` performs case-insensitive literal grep across the entire transcript and returns line numbers. `history_read` retrieves surrounding lines. Both tools paginate, including character offsets for very long lines.
5. Resuming restores the compacted working state while keeping the full transcript available. Interrupted tool calls receive an “execution status unknown” result so mutations are not automatically replayed.

The transcript is append-only during normal agent operation: compaction never rewrites or deletes it. Jev pruning only changes the working memory in `state.json`; it does not delete the durable `history.md` record. It includes compaction checkpoints and system prompts, but cannot contain provider-internal reasoning or information the API never returned. It is an ordinary editable file, not a tamper-proof audit log. History is session-scoped.

`memory.py` contains storage, compaction, and retrieval. `nanocode.py` contains the original coding agent and the integration. There is no vector database, embedding index, or retrieval service.

## Verification and limits

```sh
python3 -m unittest -v
```

Offline regression tests cover classification-based compaction and exact-detail recovery, repeated compaction and resume, tool boundaries, failed compaction, pagination, interrupted calls, malformed classifier responses, and a scripted agent loop that searches for a fact omitted from working memory. The API is mocked in these tests; they do not demonstrate live model classification quality.

The default threshold estimates message tokens as serialized UTF-8 bytes divided by three. It is not a tokenizer or a hard context guarantee; allow room for system/tools, response tokens, and the classification request. A single huge tool output can still exceed a provider's context window. Failed or oversized compactions retain the original working context and surface an error.

The Jev integration is a Python port of [fast-jev-compaction](https://github.com/typesafe-ai/fast-jev-compaction), released under the MIT license.

Use one process per session directory. Working state is saved with atomic replacement; a crash between appending the transcript and saving state can leave extra transcript entries not present in resumed working state. Like upstream nanocode, this executes shell commands and edits files with your user's permissions. Session logs contain the prompts and tool outputs you give it; keep them private.
