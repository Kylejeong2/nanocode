import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from memory import Memory
import nanocode
import jev


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.memory = Memory(self.temp.name, "system", 2000)

    def test_compaction_preserves_searchable_original_and_latest_request(self):
        m = self.memory
        m.append("user", "Exact release code: ORCHID-729\n" + "old context " * 600)
        m.append("assistant", [{"type": "text", "text": "Done"}])
        m.append("user", "What was the release code?")
        before = m.transcript.read_text()
        self.assertTrue(m.compact(lambda _: "We discussed a release; exact code omitted."))
        self.assertTrue(m.transcript.read_text().startswith(before))
        self.assertEqual(m.messages[-1]["content"], "What was the release code?")
        self.assertNotIn("ORCHID-729", str(m.messages))
        self.assertIn("ORCHID-729", m.search({"query": "orchid"}))
        number = int(m.search({"query": "orchid"}).split(":")[0])
        self.assertIn("ORCHID-729", m.read({"start_line": number, "limit": 1}))
        resumed = Memory(self.temp.name, "system", 2000)
        self.assertEqual(resumed.messages, m.messages)
        self.assertTrue(resumed.compact(lambda _: "Another checkpoint", force=True))
        self.assertIn("ORCHID-729", resumed.search({"query": "orchid"}))

    def test_compaction_at_tool_boundary_keeps_call_and_result_together(self):
        m = self.memory
        m.append("user", "Run a task")
        m.append("assistant", [{"type": "tool_use", "id": "t1", "name": "read", "input": {"path": "x"}}])
        m.append("user", [{"type": "tool_result", "tool_use_id": "t1", "content": "x" * 9000}])
        seen = []
        self.assertTrue(m.compact(lambda messages: seen.extend(messages) or "Continue task"))
        self.assertEqual(len(seen), 3)
        self.assertEqual(len(m.messages), 1)
        self.assertIn('"id": "t1"', m.transcript.read_text())

    def test_failure_retains_working_context(self):
        m = self.memory
        m.append("user", "hello")
        original = list(m.messages)
        for result in ("", "x" * 9000):
            with self.assertRaises(ValueError):
                m.compact(lambda _: result, force=True)
            self.assertEqual(m.messages, original)
        with self.assertRaises(RuntimeError):
            m.compact(lambda _: (_ for _ in ()).throw(RuntimeError("API failed")), force=True)
        self.assertEqual(m.messages, original)

    def test_jev_compaction_drops_low_value_memory_and_keeps_tail(self):
        m = self.memory
        m.append("user", "Durable constraint: never edit generated files.\n" + "context " * 600)
        m.append("assistant", [{"type": "text", "text": "Transient progress update."}])
        m.append("user", "What should we do next?")
        def prune(messages, preserve_recent):
            self.assertEqual(preserve_recent, 0)
            return {"messages": [messages[0]], "stats": {
                "callsDropped": 0, "resultsDropped": 0,
            }}
        self.assertTrue(m.compact(lambda _: "unused", prune=prune, force=True))
        contents = str(m.messages)
        self.assertIn("never edit generated files", contents)
        self.assertNotIn("Transient progress update", contents)
        self.assertIn("What should we do next?", contents)
        self.assertIn("Jev compaction: kept", m.transcript.read_text())

    def test_jev_compaction_keeps_tool_call_and_result_together(self):
        m = self.memory
        m.append("user", "Investigate the failure")
        m.append("assistant", [{"type": "tool_use", "id": "t1", "name": "read", "input": {"path": "x"}}])
        m.append("user", [{"type": "tool_result", "tool_use_id": "t1", "content": "important output"}])
        m.append("user", "Continue")
        def prune(messages, preserve_recent):
            return {"messages": [messages[0], messages[1], messages[2]], "stats": {}}
        self.assertTrue(m.compact(lambda _: "unused", prune=prune, force=True))
        self.assertIn("'id': 't1'", str(m.messages))
        self.assertIn("important output", str(m.messages))

    def test_jev_compaction_prunes_history(self):
        m = self.memory
        m.append("user", "Investigate the failure")
        m.append("assistant", [{"type": "tool_use", "id": "t1", "name": "read", "input": {"path": "x"}}])
        m.append("user", [{"type": "tool_result", "tool_use_id": "t1", "content": "important output"}])
        m.append("user", "Continue")

        def prune(messages, preserve_recent):
            return {
                "messages": [messages[0]],
                "decisions": [{"tool_use_id": "t1", "action": "drop_call"}],
                "stats": {"callsDropped": 1, "resultsDropped": 0},
            }

        self.assertTrue(m.compact(lambda _: "unused", prune=prune, force=True))
        self.assertNotIn("important output", m.transcript.read_text())
        self.assertIn("Compaction checkpoint", m.transcript.read_text())
        self.assertIn("important output", m.full.read_text())

    def test_search_pagination_and_long_line_read(self):
        m = self.memory
        m.append("user", "\n".join(["needle"] * 60) + "\n" + "A" * 15000 + "TAIL")
        first = m.search({"query": "needle", "limit": 2})
        self.assertIn("More matches", first)
        long_line = int(m.search({"query": "TAIL"}).split(":")[0])
        chunk = m.read({"start_line": long_line})
        self.assertIn("char_offset=12000", chunk)
        self.assertIn("TAIL", m.read({"start_line": long_line, "char_offset": 12000}))

    def test_interrupted_tool_is_not_replayed(self):
        self.memory.append("assistant", [{"type": "tool_use", "id": "t1", "name": "bash", "input": {"cmd": "some mutation"}}])
        resumed = Memory(self.temp.name, "system", 2000)
        self.assertEqual(resumed.messages[-1]["content"][0]["tool_use_id"], "t1")
        self.assertIn("unknown", resumed.messages[-1]["content"][0]["content"])

    def test_prune_transcript_removes_dropped_calls_and_truncates_results(self):
        m = self.memory
        m.append("user", "Keep this user text")
        m.append("assistant", [
            {"type": "tool_use", "id": "a", "name": "read", "input": {"path": "a"}},
            {"type": "tool_use", "id": "b", "name": "read", "input": {"path": "b"}},
        ])
        m.append("user", [
            {"type": "tool_result", "tool_use_id": "a", "content": "A" * 1000},
            {"type": "tool_result", "tool_use_id": "b", "content": "B" * 1000},
        ])
        m.append("assistant", [{"type": "text", "text": "Still here"}])
        m.prune_transcript([
            {"tool_use_id": "a", "action": "drop_call"},
            {"tool_use_id": "b", "action": "drop_result"},
        ], head_chars=10)
        history = m.transcript.read_text()
        full = m.full.read_text()
        self.assertNotIn('"id": "a"', history)
        self.assertNotIn("Tool call: a", history)
        self.assertNotIn("A" * 1000, history)
        self.assertIn("Tool call: b", history)
        self.assertIn("B" * 10, history)
        self.assertIn("truncated", history)
        self.assertNotIn("B" * 1000, history)
        self.assertIn("Keep this user text", history)
        self.assertIn("A" * 1000, full)
        self.assertIn("B" * 1000, full)

    def test_dotenv_loads_quotes_comments_and_literal_values(self):
        path = Path(self.temp.name) / ".env"
        path.write_text("# settings\nexport OPENROUTER_API_KEY='test-key' # comment\nMODEL = \"vendor/model\"\nLITERAL='$(echo nope) $HOME'\nEMPTY=\n")
        with patch.dict(nanocode.os.environ, {}, clear=True):
            nanocode.load_env(path)
            self.assertEqual(nanocode.os.environ["OPENROUTER_API_KEY"], "test-key")
            self.assertEqual(nanocode.os.environ["MODEL"], "vendor/model")
            self.assertEqual(nanocode.os.environ["LITERAL"], "$(echo nope) $HOME")
            self.assertEqual(nanocode.os.environ["EMPTY"], "")

    def test_dotenv_preserves_environment_and_missing_file_is_ok(self):
        path = Path(self.temp.name) / ".env"
        path.write_text("OPENROUTER_API_KEY=file-key\n")
        with patch.dict(nanocode.os.environ, {"OPENROUTER_API_KEY": "exported-key"}, clear=True):
            nanocode.load_env(path)
            nanocode.load_env(path.with_name("missing"))
            self.assertEqual(nanocode.os.environ["OPENROUTER_API_KEY"], "exported-key")

    def test_dotenv_errors_do_not_disclose_values(self):
        path = Path(self.temp.name) / ".env"
        path.write_text("OPENROUTER_API_KEY='secret-with-missing-quote\n")
        with patch.dict(nanocode.os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "^Invalid .env quoting on line 1$"):
                nanocode.load_env(path)

    def test_stream_emits_before_completion_and_reassembles_tools(self):
        emitted = []
        events = [
            {"type": "message_start", "message": {"role": "assistant", "content": []}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello "}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "world"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "t1", "name": "history_search", "input": {}}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"query":'}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '"hello"}'}},
            {"type": "content_block_stop", "index": 1},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
            {"type": "message_stop"},
        ]
        def stream():
            yield b": keepalive\n\n"
            for event in events:
                if event["type"] == "message_stop":
                    self.assertEqual(emitted, ["Hello ", "world"])
                yield ("data: " + json.dumps(event) + "\n").encode()
                yield b"\n"
        result = nanocode.read_stream(stream(), emitted.append)
        self.assertEqual(result["content"][0]["text"], "Hello world")
        self.assertEqual(result["content"][1]["input"], {"query": "hello"})
        self.assertEqual(result["stop_reason"], "tool_use")

    def test_stream_rejects_disconnects_and_provider_errors(self):
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            nanocode.read_stream(iter([]), lambda text: None)
        with self.assertRaisesRegex(RuntimeError, "overloaded"):
            nanocode.read_stream(iter([b'data: {"type":"error","error":{"message":"overloaded"}}\n', b'\n']), lambda text: None)

    def test_stream_request_and_response_cleanup(self):
        with patch.object(nanocode, "OPENROUTER_KEY", "test-key"), patch.object(nanocode.urllib.request, "urlopen") as send:
            send.return_value.__iter__.return_value = iter([b'data: {"type":"message_stop"}\n', b'\n'])
            nanocode.call_api([], "system", on_text=lambda text: None)
            self.assertTrue(json.loads(send.call_args.args[0].data)["stream"])
            send.return_value.close.assert_called_once()

    def test_api_requires_openrouter_key_without_fallback(self):
        with patch.object(nanocode, "OPENROUTER_KEY", None), patch.dict(nanocode.os.environ, {"ANTHROPIC_API_KEY": "unused"}), patch.object(nanocode.urllib.request, "urlopen") as send:
            for summary in (False, True):
                with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
                    nanocode.call_api([], "system", summary=summary)
            send.assert_not_called()

    def test_agent_and_compaction_use_openrouter(self):
        with patch.object(nanocode, "OPENROUTER_KEY", "test-key"), patch.object(nanocode.urllib.request, "urlopen") as send:
            send.return_value.read.return_value = b'{"content": []}'
            for summary in (False, True):
                nanocode.call_api([{"role": "user", "content": "hello"}], "system", summary=summary)
                request = send.call_args.args[0]
                self.assertEqual(request.full_url, "https://openrouter.ai/api/v1/messages")
                self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
                self.assertIsNone(request.get_header("X-api-key"))
                self.assertEqual("tools" in json.loads(request.data), not summary)

    def test_jev_client_request_and_validation(self):
        class Response:
            status = 200
            def read(self):
                return b'{"answers":{"call_t1":{"noul":0.8}}}'
            def close(self):
                pass
        requests = []
        def opener(request):
            requests.append(request)
            return Response()
        client = jev.JevClient("jev-key", "jev-test", "https://jev.test", opener)
        self.assertEqual(client.ask({"history": []}, {"call_t1": {"type": "noul"}})["answers"]["call_t1"]["noul"], 0.8)
        self.assertEqual(requests[0].full_url, "https://jev.test")
        self.assertEqual(requests[0].get_header("Authorization"), "Bearer jev-key")
        self.assertEqual(json.loads(requests[0].data)["model"], "jev-test")
        self.assertEqual(jev.noul_answer({"x": {"noul": 0.5}}, "x"), 0.5)
        with self.assertRaises(ValueError):
            jev.noul_answer({}, "x")

    def test_jev_client_rejects_http_malformed_and_missing_answers(self):
        class Response:
            def __init__(self, status, body):
                self.status, self.body = status, body
            def read(self):
                return self.body
            def close(self):
                pass
        for response, message in (
            (Response(500, b"provider exploded" * 20), r"Jev request failed \(500\)"),
            (Response(200, b"not json"), "malformed JSON"),
            (Response(200, b"{}"), "missing answers"),
        ):
            with self.subTest(message=message):
                client = jev.JevClient("key", opener=lambda _request, response=response: response)
                with self.assertRaisesRegex(ValueError, message):
                    client.ask({}, {})

    def test_fit_state_shrinks_and_rejects_impossible_history(self):
        messages = [
            {"role": "user", "content": "old " * 2000},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "read", "input": {"path": "x" * 2000}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "result"}]},
            {"role": "user", "content": "recent"},
        ]
        calls = jev.collect_tool_calls(messages, 0)
        state, tokens, stage = jev.fit_state(messages, calls, max_state_tokens=1000, preserve_recent=0)
        self.assertLessEqual(tokens, 1000)
        self.assertTrue(stage)
        with self.assertRaisesRegex(ValueError, "history too large for Jev"):
            jev.fit_state(messages, calls, max_state_tokens=1, preserve_recent=0)

    def test_jev_compact_applies_keep_drop_result_and_drop_call(self):
        messages = [
            {"role": "user", "content": "request"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "read", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "A" * 500}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "b", "name": "read", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "b", "content": "B" * 500}]},
        ]
        class Asker:
            def ask(self, state, questions):
                answers = {}
                for name in questions:
                    answers[name] = {"noul": 0.8 if name == "call_t1" else 0.2}
                return {"answers": answers}
        result = jev.compact(messages, Asker(), preserve_recent=0, truncate_head_chars=10)
        self.assertEqual(result["stats"]["calls"], 2)
        self.assertIn("truncated", result["messages"][2]["content"][0]["content"])
        self.assertEqual(len(result["messages"]), 3)

    def test_jev_prune_pins_newest_tool_pair(self):
        messages = [
            {"role": "user", "content": "request"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "read", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "A" * 500}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "b", "name": "read", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "b", "content": "B" * 500}]},
        ]
        class Asker:
            def ask(self, state, questions):
                return {"answers": {name: {"noul": 0.0} for name in questions}}
        with patch.object(nanocode.jev, "JevClient", return_value=Asker()), \
                patch.object(nanocode, "TYPESAFE_KEY", "k"):
            result = nanocode.jev_prune(messages, 2)
        self.assertEqual(result["stats"]["pinned"], 1)
        self.assertEqual(result["stats"]["callsDropped"], 1)
        self.assertEqual(result["decisions"][0]["tool_use_id"], "a")
        self.assertEqual(result["messages"][-1]["content"][0]["content"], "B" * 500)

    def test_best_effort_compaction_applies_shrink_and_skips_no_op_until_growth(self):
        m = self.memory
        m.append("user", "task")
        m.append("assistant", [{"type": "tool_use", "id": "t1", "name": "read", "input": {}}])
        m.append("user", [{"type": "tool_result", "tool_use_id": "t1", "content": "R" * 9000}])
        stats = {"callsDropped": 0, "resultsDropped": 0}
        shrink = lambda msgs, recent: {"messages": [msgs[0], msgs[1], {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "R" * 8000}]}], "stats": stats}
        self.assertTrue(m.compact(lambda _: "", prune=shrink))
        self.assertGreaterEqual(m.estimate(), m.threshold)
        calls = []
        def noop(msgs, recent):
            calls.append(recent)
            return {"messages": list(msgs), "stats": stats}
        self.assertFalse(m.compact(lambda _: "", prune=noop))
        self.assertFalse(m.compact(lambda _: "", prune=noop))
        self.assertEqual(len(calls), 0)
        m.append("assistant", [{"type": "text", "text": "more"}])
        self.assertFalse(m.compact(lambda _: "", prune=noop))
        self.assertEqual(calls, [2])
        with self.assertRaises(ValueError):
            m.compact(lambda _: "", prune=noop, force=True)

    def test_prune_malformed_shape_leaves_state_untouched(self):
        original = list(self.memory.messages)
        self.memory.append("user", "new")
        original = list(self.memory.messages)
        with self.assertRaises(ValueError):
            self.memory.compact(lambda _: "unused", prune=lambda _, __: [], force=True)
        self.assertEqual(self.memory.messages, original)

    def test_cli_compacts_then_retrieves_via_agent_tool_loop(self):
        session = str(Path(self.temp.name) / "cli")
        responses = iter([
            {"content": [{"type": "text", "text": "Recorded."}]},
            {"content": [{"type": "text", "text": "Checkpoint."}]},
            {"content": [{"type": "tool_use", "id": "s", "name": "history_search", "input": {"query": "ORCHID"}}]},
            {"content": [{"type": "text", "text": "The code was ORCHID-729."}]},
        ])
        calls = []
        def api(messages, system, summary=False, on_text=None):
            calls.append((str(messages), summary))
            return next(responses)
        with patch("sys.argv", ["nanocode.py", "--session", session]), patch("builtins.input", side_effect=["Code: ORCHID-729", "/compact", "What was the code?", "/q"]), patch.object(nanocode, "call_api", side_effect=api), patch.object(nanocode, "TYPESAFE_KEY", None), patch("builtins.print"):
            nanocode.main()
        self.assertTrue(calls[1][1])
        self.assertNotIn("ORCHID-729", calls[2][0])
        self.assertIn("ORCHID-729", calls[3][0])
        self.assertIn("history_search", (Path(session) / "history.md").read_text())


if __name__ == "__main__":
    unittest.main()
