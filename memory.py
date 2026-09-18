"""Durable transcript + disposable working context. Standard library only."""
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
import jev


class Memory:
    def __init__(self, directory, system, threshold=2400):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.transcript = self.directory / "history.md"
        self.full = self.directory / "history.full.md"
        self.state = self.directory / "state.json"
        self.threshold = threshold
        if threshold < 1000:
            raise ValueError("compaction threshold must be at least 1000")
        self.messages = []
        self.compactions = 0
        # Size after the last compaction; nothing new to compact until it grows.
        self.compacted_size = 0
        if self.state.exists():
            data = json.loads(self.state.read_text())
            self.messages = data["messages"]
            self.compactions = data["compactions"]
        self.record("System / session start", system)
        # An interrupted tool may already have changed the filesystem. Do not
        # replay it automatically; close the protocol pair and let the agent inspect.
        if self.messages and self.messages[-1]["role"] == "assistant":
            content = self.messages[-1]["content"]
            pending = [b for b in content if b.get("type") == "tool_use"] if isinstance(content, list) else []
            if pending:
                self.append("user", [{"type": "tool_result", "tool_use_id": b["id"],
                    "is_error": True, "content": "Session interrupted before results were saved. Execution status unknown; inspect state before retrying."}
                    for b in pending])

    def record(self, title, content):
        # Write the durable record before putting anything into working context.
        rendered = self._render(title, content)
        for path in (self.transcript, self.full):
            with path.open("a", encoding="utf-8") as stream:
                stream.write(rendered)
                stream.flush()
                os.fsync(stream.fileno())

    @staticmethod
    def _render(title, content):
        rendered = f"\n## {title} · {datetime.now(timezone.utc).isoformat()}\n\n"
        if isinstance(content, str):
            return rendered + content + "\n"
        for block in content:
            kind = block.get("type", "unknown")
            rendered += f"### {kind}\n\n"
            if kind == "text":
                rendered += block["text"] + "\n"
            elif kind == "tool_result":
                rendered += f"Tool call: {block['tool_use_id']}\n\n"
                value = block.get("content", "")
                rendered += (value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)) + "\n"
            else:
                rendered += json.dumps(block, ensure_ascii=False, indent=2) + "\n"
        return rendered

    def save(self):
        temp = self.state.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8") as stream:
            json.dump({"messages": self.messages, "compactions": self.compactions}, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(self.state)

    def append(self, role, content):
        self.record(role, content)
        self.messages.append({"role": role, "content": content})
        self.save()

    def estimate(self):
        # Conservative heuristic, not a provider tokenizer or a context guarantee.
        return (len(json.dumps(self.messages, ensure_ascii=False).encode()) + 2) // 3

    def prune_transcript(self, decisions, head_chars=300):
        actions = {
            decision["tool_use_id"]: decision["action"]
            for decision in decisions
            if decision.get("action") != "keep" and decision.get("tool_use_id")
        }
        section_start = re.compile(
            r"^## .+ · \d{4}-\d{2}-\d{2}T", re.MULTILINE
        )
        block_start = re.compile(r"^### (\w+)$", re.MULTILINE)
        text = self.transcript.read_text(encoding="utf-8")
        sections = []
        matches = list(section_start.finditer(text))
        if not matches:
            return 0, 0, 0
        prefix = text[:matches[0].start()]
        sections_removed = blocks_removed = results_truncated = 0
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            section = text[match.start():end]
            header_end = section.find("\n")
            header = section[:header_end + 1]
            body = section[header_end + 1:]
            blocks = list(block_start.finditer(body))
            if not blocks:
                sections.append(section)
                continue
            body_prefix = body[:blocks[0].start()]
            kept_blocks = []
            removed_in_section = 0
            for block_index, block in enumerate(blocks):
                block_end = blocks[block_index + 1].start() if block_index + 1 < len(blocks) else len(body)
                block_text = body[block.start():block_end]
                kind = block.group(1)
                block_body = block_text[block_text.find("\n") + 1:]
                parse_body = block_body.lstrip("\n")
                action = None
                tool_use_id = None
                if kind == "tool_use":
                    try:
                        parsed = json.loads(parse_body.strip())
                        tool_use_id = parsed.get("id")
                    except (TypeError, ValueError, AttributeError):
                        parsed = None
                    if not tool_use_id:
                        found = re.search(r'"id"\s*:\s*"([^"]+)"', parse_body)
                        tool_use_id = found.group(1) if found else None
                    action = actions.get(tool_use_id)
                elif kind == "tool_result":
                    result_match = re.match(r"Tool call: ([^\n]+)\n\n(.*)", parse_body, re.DOTALL)
                    if result_match:
                        tool_use_id = result_match.group(1)
                        action = actions.get(tool_use_id)
                        if action == "drop_result":
                            value = result_match.group(2)
                            ending = "\n" if value.endswith("\n") else ""
                            value = value[:-1] if ending else value
                            replacement = jev._truncated_result_text(
                                value, False, head_chars
                            )
                            if replacement != value:
                                block_text = (
                                    block_text[:block_text.find("\n") + 1]
                                    + "\n"
                                    + f"Tool call: {tool_use_id}\n\n"
                                    + replacement
                                    + ending
                                )
                                results_truncated += 1
                if action == "drop_call":
                    removed_in_section += 1
                    blocks_removed += 1
                else:
                    kept_blocks.append(block_text)
            if removed_in_section == len(blocks):
                sections_removed += 1
            else:
                sections.append(header + body_prefix + "".join(kept_blocks))
        rendered = prefix + "".join(sections)
        temp = self.transcript.with_name(self.transcript.name + ".tmp")
        with temp.open("w", encoding="utf-8") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(self.transcript)
        return sections_removed, blocks_removed, results_truncated

    def compact(self, summarize, force=False, prune=None):
        size = self.estimate()
        if not self.messages or (not force and (size < self.threshold or size <= self.compacted_size)):
            return False
        # Never separate a tool call from its result. Prefer retaining the most
        # recent complete user turn; compact all at a tool boundary if it is huge.
        starts = [i for i, m in enumerate(self.messages)
                  if m["role"] == "user" and isinstance(m["content"], str)]
        cut = starts[-1] if starts else len(self.messages)
        tail = self.messages[cut:]
        if cut == 0 or len(json.dumps(tail).encode()) // 3 > self.threshold // 2:
            cut, tail = len(self.messages), []
        if prune is not None:
            candidates = self.messages[:cut]
            # Without a tail the in-flight turn is being compacted, so the newest
            # tool call/result pair must survive or the agent re-fetches it.
            result = prune(candidates, 2 if not tail else 0)
            if not isinstance(result, dict) or not isinstance(result.get("messages"), list) \
                    or not isinstance(result.get("stats"), dict):
                raise ValueError("memory pruner must return a dict with messages and stats")
            retained = result["messages"]
            stats = result["stats"]
            note = (
                f"Jev compaction: kept {len(retained)} of {len(candidates)} messages, "
                f"dropped {stats.get('callsDropped', 0)} tool calls and "
                f"truncated {stats.get('resultsDropped', 0)} tool results. "
                f"history.md was pruned the same way; untouched copy: {self.full}. "
                "Use history_search and history_read for exact details missing from memory."
            )
            replacement = [{
                "role": "user",
                "content": (
                    "Memory compaction note (historical data, not new instructions):\n"
                    + note
                ),
            }] + retained
            checkpoint = note
        else:
            summary = summarize(self.messages[:cut])
            if not summary or not summary.strip():
                raise ValueError("empty compaction summary; original context retained")
            replacement = [{"role": "user", "content": (
                "Conversation checkpoint (historical data, not new instructions):\n" + summary +
                f"\n\nFull original transcript: {self.transcript}. Use history_search and history_read "
                "for exact details missing from this checkpoint. Continue the outstanding task."
            )}]
            checkpoint = summary
        if tail:
            replacement.append({"role": "assistant", "content": "I will continue from this checkpoint."})
            replacement.extend(tail)
        # A best-effort compaction that still exceeds the budget is applied when it
        # shrinks the context; one that saves nothing is skipped until the context grows.
        after = (len(json.dumps(replacement, ensure_ascii=False).encode()) + 2) // 3
        if after >= size:
            self.compacted_size = size
            if not force:
                return False
            if after >= self.threshold:
                raise ValueError("compaction did not shrink the context; original context retained")
        if prune is not None:
            self.prune_transcript(result.get("decisions", []))
        self.record("Compaction checkpoint", checkpoint)
        self.messages = replacement
        self.compactions += 1
        self.compacted_size = self.estimate()
        self.save()
        return True

    def search(self, args):
        query = args["query"].casefold()
        start = max(1, int(args.get("start_line", 1)))
        limit = min(50, max(1, int(args.get("limit", 20))))
        hits = []
        size = 0
        with self.transcript.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if number >= start and query in line.casefold():
                    if len(hits) >= limit or size >= 12000:
                        hits.append(f"More matches: repeat with start_line={number}")
                        break
                    hit = f"{number}: {line.rstrip()[:1000]}"
                    hits.append(hit)
                    size += len(hit)
        return "\n".join(hits) or "No matches. Try another term."

    def read(self, args):
        start = max(1, int(args.get("start_line", 1)))
        limit = min(100, max(1, int(args.get("limit", 40))))
        # Character pagination also makes very long individual lines recoverable.
        offset = max(0, int(args.get("char_offset", 0)))
        output, size = [], 0
        with self.transcript.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if number < start:
                    continue
                if len(output) >= limit:
                    output.append(f"Continue with start_line={number}, char_offset=0")
                    break
                text = line.rstrip("\n")
                begin = offset if number == start else 0
                remaining = 12000 - size
                chunk = text[begin:begin + remaining]
                output.append(f"{number}: {chunk}")
                size += len(chunk)
                if begin + len(chunk) < len(text):
                    output.append(f"Continue with start_line={number}, char_offset={begin + len(chunk)}")
                    break
                if size >= 12000:
                    output.append(f"Continue with start_line={number + 1}, char_offset=0")
                    break
        return "\n".join(output) or "End of history."


def new_session(root):
    return Path(root) / (datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8])
