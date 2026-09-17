import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from memory import Memory
import nanocode


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

    def test_classified_compaction_drops_low_value_memory_and_keeps_tail(self):
        m = self.memory
        m.append("user", "Durable constraint: never edit generated files.\n" + "context " * 600)
        m.append("assistant", [{"type": "text", "text": "Transient progress update."}])
        m.append("user", "What should we do next?")
        self.assertTrue(m.compact(
            lambda _: "unused",
            classify=lambda records: [0],
            force=True,
        ))
        contents = str(m.messages)
        self.assertIn("never edit generated files", contents)
        self.assertNotIn("Transient progress update", contents)
        self.assertIn("What should we do next?", contents)
        self.assertIn("Classified memory: kept", m.transcript.read_text())

    def test_classified_compaction_keeps_tool_call_and_result_together(self):
        m = self.memory
        m.append("user", "Investigate the failure")
        m.append("assistant", [{"type": "tool_use", "id": "t1", "name": "read", "input": {"path": "x"}}])
        m.append("user", [{"type": "tool_result", "tool_use_id": "t1", "content": "important output"}])
        m.append("user", "Continue")
        self.assertTrue(m.compact(
            lambda _: "unused",
            classify=lambda records: [2],
            force=True,
        ))
        self.assertIn("'id': 't1'", str(m.messages))
        self.assertIn("important output", str(m.messages))

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

    def test_classify_memory_parses_json_response(self):
        with patch.object(nanocode, "call_api", return_value={
            "content": [{"type": "text", "text": "```json\n{\"keep\":[0, 2]}\n```"}]
        }) as call:
            self.assertEqual(nanocode.classify_memory([{"index": 0}, {"index": 1}]), [0, 2])
            self.assertTrue(call.call_args.args[1].startswith("Classify historical"))

    def test_classify_memory_rejects_malformed_response(self):
        with patch.object(nanocode, "call_api", return_value={
            "content": [{"type": "text", "text": "not json"}]
        }):
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                nanocode.classify_memory([])

    def test_cli_compacts_then_retrieves_via_agent_tool_loop(self):
        session = str(Path(self.temp.name) / "cli")
        responses = iter([
            {"content": [{"type": "text", "text": "Recorded."}]},
            {"content": [{"type": "text", "text": "{\"keep\":[]}"}]},
            {"content": [{"type": "tool_use", "id": "s", "name": "history_search", "input": {"query": "ORCHID"}}]},
            {"content": [{"type": "text", "text": "The code was ORCHID-729."}]},
        ])
        calls = []
        def api(messages, system, summary=False, on_text=None):
            calls.append((str(messages), summary))
            return next(responses)
        with patch("sys.argv", ["nanocode.py", "--session", session]), patch("builtins.input", side_effect=["Code: ORCHID-729", "/compact", "What was the code?", "/q"]), patch.object(nanocode, "call_api", side_effect=api), patch("builtins.print"):
            nanocode.main()
        self.assertTrue(calls[1][1])
        self.assertNotIn("ORCHID-729", calls[2][0])
        self.assertIn("ORCHID-729", calls[3][0])
        self.assertIn("history_search", (Path(session) / "history.md").read_text())


if __name__ == "__main__":
    unittest.main()
