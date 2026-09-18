"""Jev-backed tool history compaction, ported from fast-jev-compaction."""

import json
import math
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


SYSTEM_ONE_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
STATE_CONTEXT = (
    "A coding assistant conversation is being compacted to free context. `history` is the whole conversation so far, oldest "
    "first; tool outputs are replaced by a short `result` note and long texts may be abridged. Each question asks "
    "whether one tool call, or the full output of that call, still needs to stay in the history verbatim. Whatever is not "
    "kept is deleted permanently, but the assistant can always re-run a tool or re-read a file."
)

REQUEST_OVERHEAD_TOKENS = 20
_TOKEN_PIECES = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")
_INPUT_CHARS = (1000, 200, 60)
_TEXT_HEAD = 400
_TEXT_TAIL = 150


class JevClient:
    def __init__(self, api_key, model=DEFAULT_MODEL, base_url=SYSTEM_ONE_URL, opener=urllib.request.urlopen):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.opener = opener

    def ask(self, state, questions):
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps({"model": self.model, "state": state, "questions": questions}).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            response = self.opener(request)
            status = getattr(response, "status", None)
            if status is None:
                status = response.getcode()
            raw = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read(200).decode("utf-8", "replace")
            raise ValueError(f"Jev request failed ({error.code}): {detail[:200]}") from error
        finally:
            if "response" in locals() and hasattr(response, "close"):
                response.close()
        if not 200 <= status < 300:
            detail = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
            raise ValueError(f"Jev request failed ({status}): {detail[:200]}")
        try:
            parsed = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        except (TypeError, ValueError) as error:
            raise ValueError("Jev returned malformed JSON") from error
        if (
            not isinstance(parsed, dict)
            or not isinstance(parsed.get("answers"), dict)
        ):
            raise ValueError("Jev response is missing answers")
        return parsed


def noul_answer(answers, name):
    answer = answers.get(name) if isinstance(answers, dict) else None
    value = answer.get("noul") if isinstance(answer, dict) else None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"Invalid Jev answer for {name}")
    return float(value)


def estimate_tokens(text):
    tokens = 0
    for piece in _TOKEN_PIECES.findall(text):
        first = ord(piece[0])
        if 48 <= first <= 57:
            tokens += len(piece) / 2
        elif 65 <= first <= 90 or 97 <= first <= 122:
            tokens += 1 + (len(piece) - 1) // 6
        else:
            tokens += 0.9
    return math.ceil(tokens)


def truncate(text, limit):
    return text if len(text) <= limit else text[:max(0, limit - 1)] + "…"


def abridge(text, head, tail):
    if len(text) <= head + tail + 40:
        return text
    omitted = len(text) - head - tail
    return f"{text[:head]}\n[… {omitted} chars omitted …]\n{text[-tail:]}"


def _content_blocks(message):
    content = message.get("content", [])
    return content if isinstance(content, list) else []


def _message_text(message):
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


@dataclass
class ToolCall:
    id: str
    tool_use_id: str
    tool: str
    input: object
    call_index: int
    result_index: int
    result_chars: int
    is_error: bool
    pinned: bool


def _is_pinned(index, total, preserve_recent):
    return index == 0 or index >= total - preserve_recent


def collect_tool_calls(messages, preserve_recent):
    results = {}
    for index, message in enumerate(messages):
        for block in _content_blocks(message):
            if block.get("type") == "tool_result" and block.get("tool_use_id") is not None:
                results[block["tool_use_id"]] = (index, block)
    calls = []
    for call_index, message in enumerate(messages):
        for block in _content_blocks(message):
            if block.get("type") != "tool_use":
                continue
            tool_use_id = block.get("id")
            found = results.get(tool_use_id)
            if not found:
                continue
            result_index, result = found
            calls.append(ToolCall(
                id=f"t{len(calls) + 1}",
                tool_use_id=tool_use_id,
                tool=block.get("name", ""),
                input=block.get("input", {}),
                call_index=call_index,
                result_index=result_index,
                result_chars=len(result.get("content", "") if isinstance(result.get("content", ""), str)
                                 else json.dumps(result.get("content", ""), ensure_ascii=False)),
                is_error=bool(result.get("is_error", False)),
                pinned=(
                    _is_pinned(call_index, len(messages), preserve_recent)
                    or _is_pinned(result_index, len(messages), preserve_recent)
                ),
            ))
    return calls


def _input_text(value, limit):
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        encoded = "[unserializable input]"
    return truncate(encoded, limit)


def _result_note(call):
    return f"{'error' if call.is_error else 'ok'}, {call.result_chars} chars (omitted)"


def _compact_call(call):
    pieces = []
    values = call.input.items() if isinstance(call.input, dict) else []
    for key, value in values:
        text = value if isinstance(value, str) else _input_text({key: value}, 200)
        text = re.sub(r"\s+", " ", text)
        pieces.append(f"{key}={text}")
    return f"{call.id} {call.tool} {truncate(' '.join(pieces), 60)} → {'error' if call.is_error else 'ok'} {call.result_chars}ch"


def _calls_by_message(calls):
    grouped = {}
    for call in calls:
        grouped.setdefault(call.call_index, []).append(call)
    return grouped


def _history_entries(messages, calls, input_chars):
    grouped = _calls_by_message(calls)
    entries = []
    for index, message in enumerate(messages):
        tool_calls = [
            {
                "id": call.id,
                "tool": call.tool,
                "input": _input_text(call.input, input_chars),
                "result": _result_note(call),
            }
            for call in grouped.get(index, [])
        ]
        text = _message_text(message)
        if not text.strip() and not tool_calls:
            continue
        entry = {"i": index, "role": message["role"], "text": text}
        if tool_calls:
            entry["tool_calls"] = tool_calls
        entries.append(entry)
    return entries


def _goal_from_messages(messages):
    prompts = [
        _message_text(message)
        for message in messages
        if message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and message["content"].strip()
    ]
    return "\n".join(truncate(text, 500) for text in prompts[-3:])


def fit_state(messages, calls, goal="", max_state_tokens=25000, preserve_recent=6):
    goal = goal or _goal_from_messages(messages)
    def state_of(history):
        return {"context": STATE_CONTEXT, "goal": goal, "history": history}

    base_tokens = estimate_tokens(json.dumps(state_of([]), ensure_ascii=False))

    def rebuild(input_chars):
        history = _history_entries(messages, calls, input_chars)
        per_entry = [estimate_tokens(json.dumps(entry, ensure_ascii=False)) + 1 for entry in history]
        return history, per_entry, base_tokens + sum(per_entry)

    history, per_entry, tokens = rebuild(_INPUT_CHARS[0])
    if tokens <= max_state_tokens:
        return state_of(history), tokens, "full"
    for limit in _INPUT_CHARS[1:]:
        history, per_entry, tokens = rebuild(limit)
        if tokens <= max_state_tokens:
            return state_of(history), tokens, f"inputs<={limit}"

    def pinned(entry):
        return _is_pinned(entry["i"], len(messages), preserve_recent)

    order = [i for i, entry in enumerate(history) if not pinned(entry)]
    order += [i for i, entry in enumerate(history) if pinned(entry)]
    for index in order:
        entry = history[index]
        if len(entry["text"]) <= _TEXT_HEAD + _TEXT_TAIL + 40:
            continue
        old = estimate_tokens(json.dumps(entry, ensure_ascii=False)) + 1
        entry["text"] = abridge(entry["text"], _TEXT_HEAD, _TEXT_TAIL)
        new = estimate_tokens(json.dumps(entry, ensure_ascii=False)) + 1
        tokens += new - old
        per_entry[index] = new
        if tokens <= max_state_tokens:
            return state_of(history), tokens, "texts abridged"
    for index in order:
        entry = history[index]
        if pinned(entry) or not entry["text"]:
            continue
        old = estimate_tokens(json.dumps(entry, ensure_ascii=False)) + 1
        original = len(_message_text(messages[entry["i"]]))
        entry["text"] = f"[… {original} chars omitted …]"
        new = estimate_tokens(json.dumps(entry, ensure_ascii=False)) + 1
        tokens += new - old
        per_entry[index] = new
        if tokens <= max_state_tokens:
            return state_of(history), tokens, "old messages collapsed"
    grouped = _calls_by_message(calls)
    for index in order:
        entry = history[index]
        own = grouped.get(entry["i"])
        if pinned(entry) or not own:
            continue
        old = estimate_tokens(json.dumps(entry, ensure_ascii=False)) + 1
        entry["tool_calls"] = [_compact_call(call) for call in own]
        new = estimate_tokens(json.dumps(entry, ensure_ascii=False)) + 1
        tokens += new - old
        per_entry[index] = new
        if tokens <= max_state_tokens:
            return state_of(history), tokens, "old calls compacted"
    left = set()
    for index in order:
        entry = history[index]
        if pinned(entry) or "tool_calls" in entry:
            continue
        left.add(index)
        tokens -= per_entry[index]
        if tokens <= max_state_tokens:
            return state_of([entry for i, entry in enumerate(history) if i not in left]), tokens, "old messages left out"

    filtered = [entry for i, entry in enumerate(history) if i not in left]
    merged = []
    for entry in filtered:
        previous = merged[-1] if merged else None
        foldable = lambda item: (
            not pinned(item) and not item["text"] and isinstance(item.get("tool_calls", [None])[0], str)
        )
        if previous and foldable(previous) and foldable(entry) and previous["role"] == entry["role"]:
            previous["tool_calls"].extend(entry["tool_calls"])
        else:
            merged.append(dict(entry))
    tokens = base_tokens + sum(estimate_tokens(json.dumps(entry, ensure_ascii=False)) + 1 for entry in merged)
    if tokens <= max_state_tokens:
        return state_of(merged), tokens, "old calls merged"
    raise ValueError(f"history too large for Jev (~{tokens} tokens after truncation, limit {max_state_tokens})")


def questions_for(call):
    return {
        f"call_{call.id}": {
            "type": "noul",
            "instructions": (
                f"Tool call {call.id} ({call.tool}) should stay in the history: knowing this call was made, "
                "with its input, still matters for what the assistant does next"
            ),
        },
        f"result_{call.id}": {
            "type": "noul",
            "instructions": (
                f"The full output of tool call {call.id} ({call.tool}, {call.result_chars} chars) should stay "
                "in the history verbatim: the assistant still needs its contents and re-running the tool would not do"
            ),
        },
    }


def batch_calls(candidates, state_tokens, max_request_tokens):
    budget = max_request_tokens - state_tokens - REQUEST_OVERHEAD_TOKENS
    batches, current, current_tokens = [], [], 0
    for call in candidates:
        tokens = estimate_tokens(json.dumps(questions_for(call), ensure_ascii=False))
        if current and current_tokens + tokens > budget:
            batches.append(current)
            current, current_tokens = [], 0
        if not current and tokens > budget:
            raise ValueError(f"state leaves no room for questions (~{state_tokens} of {max_request_tokens} tokens)")
        current.append(call)
        current_tokens += tokens
    if current:
        batches.append(current)
    return batches


def decide_call(call, keep_call, keep_result, keep_threshold):
    result = {
        "id": call.id,
        "tool_use_id": call.tool_use_id,
        "tool": call.tool,
        "keep_call": keep_call,
        "keep_result": keep_result,
    }
    if call.pinned:
        result.update(action="keep", reason="pinned")
    elif keep_result >= keep_threshold:
        result.update(action="keep", reason="kept")
    elif keep_call >= keep_threshold:
        result.update(action="drop_result", reason="result_dropped")
    else:
        result.update(action="drop_call", reason="call_dropped")
    return result


def _truncated_result_text(text, is_error, head_chars):
    if len(text) <= head_chars + 120:
        return text
    head = f"{text[:head_chars]}\n" if head_chars > 0 else ""
    suffix = " (error)" if is_error else ""
    return f"{head}[fast-jev-compaction truncated {len(text) - head_chars} chars of this tool result{suffix}; re-run the tool if needed]"


def apply_decisions(messages, decisions, calls, head_chars):
    by_id = {call.id: call for call in calls}
    actions = {
        by_id[decision["id"]].tool_use_id: decision["action"]
        for decision in decisions
        if decision.get("id") in by_id and decision.get("action") != "keep"
    }
    output = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            output.append(message)
            continue
        touched = any(
            block.get("type") in ("tool_use", "tool_result")
            and (block.get("id") or block.get("tool_use_id")) in actions
            for block in content
        )
        if not touched:
            output.append(message)
            continue
        rebuilt = []
        changed = False
        for block in content:
            identifier = block.get("id") or block.get("tool_use_id")
            action = actions.get(identifier)
            if action == "drop_call":
                changed = True
                continue
            if action == "drop_result" and block.get("type") == "tool_result":
                text = block.get("content", "")
                if not isinstance(text, str):
                    text = json.dumps(text, ensure_ascii=False)
                replacement = _truncated_result_text(text, bool(block.get("is_error")), head_chars)
                if replacement != text:
                    block = dict(block)
                    block["content"] = replacement
                    changed = True
            rebuilt.append(block)
        if not changed:
            output.append(message)
        elif not rebuilt or all(block.get("type") == "text" and not block.get("text", "").strip() for block in rebuilt):
            continue
        else:
            copy = dict(message)
            copy["content"] = rebuilt
            output.append(copy)
    return output


def _message_chars(message):
    content = message.get("content", "")
    if isinstance(content, str):
        return len(content)
    total = 0
    for block in content:
        if block.get("type") == "tool_use":
            try:
                total += len(json.dumps(block.get("input", {}), ensure_ascii=False))
            except (TypeError, ValueError):
                total += 20
        elif block.get("type") == "tool_result":
            value = block.get("content", "")
            total += len(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
        elif block.get("type") == "text":
            total += len(block.get("text", ""))
    return total


def compact(
    messages,
    asker,
    goal="",
    keep_threshold=0.5,
    preserve_recent=6,
    max_state_tokens=25000,
    max_request_tokens=30000,
    truncate_head_chars=300,
):
    started = time.time()
    calls = collect_tool_calls(messages, preserve_recent)
    candidates = [call for call in calls if not call.pinned]
    chars_before = sum(_message_chars(message) for message in messages)
    fitted = (0, "")
    batches = []
    answers = {}
    if candidates:
        state, state_tokens, stage = fit_state(
            messages, calls, goal, max_state_tokens, preserve_recent
        )
        fitted = (state_tokens, stage)
        batches = batch_calls(candidates, state_tokens, max_request_tokens)
        for batch in batches:
            questions = {}
            for call in batch:
                questions.update(questions_for(call))
            response = asker.ask(state, questions)
            response_answers = response.get("answers") if isinstance(response, dict) else None
            if not isinstance(response_answers, dict):
                raise ValueError("Jev response is missing answers")
            for call in batch:
                answers[call.id] = (
                    noul_answer(response_answers, f"call_{call.id}"),
                    noul_answer(response_answers, f"result_{call.id}"),
                )
    decisions = []
    for call in calls:
        keep_call, keep_result = answers.get(call.id, (1.0, 1.0))
        decisions.append(decide_call(call, keep_call, keep_result, keep_threshold))
    kept = apply_decisions(messages, decisions, calls, truncate_head_chars)
    reasons = [decision["reason"] for decision in decisions]
    return {
        "messages": kept,
        "decisions": decisions,
        "stats": {
            "messagesBefore": len(messages),
            "messagesAfter": len(kept),
            "charsBefore": chars_before,
            "charsAfter": sum(_message_chars(message) for message in kept),
            "calls": len(calls),
            "kept": reasons.count("kept"),
            "resultsDropped": reasons.count("result_dropped"),
            "callsDropped": reasons.count("call_dropped"),
            "pinned": reasons.count("pinned"),
            "stateTokens": fitted[0],
            "stateStage": fitted[1],
            "requests": len(batches),
            "ms": int((time.time() - started) * 1000),
        },
    }
