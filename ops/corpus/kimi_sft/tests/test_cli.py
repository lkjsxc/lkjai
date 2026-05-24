import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from kimi_sft import cli
from kimi_sft.api import call_kimi_cli_jobs
from kimi_sft.generate import apply_pilot_gate


def good_row(split="train", row_id="row-1", family="family-1"):
    return {
        "messages": [
            {"role": "user", "content": "Find release notes."},
            {
                "role": "assistant",
                "content": "<action>\n<reasoning>Search resources.</reasoning>\n<tool>resource.search</tool>\n<query>release notes</query>\n</action>",
            },
        ],
        "tags": ["sft", "language:en"],
        "meta": {
            "schema": cli.SCHEMA,
            "id": row_id,
            "split": split,
            "provenance": "kimi-generated",
            "mode": "sft",
            "prompt_contract": "agent-api",
            "scenario_family_id": family,
            "tool_sequence": ["resource.search"],
            "confirmation_required": False,
        },
    }


def write_row(root, split, row):
    path = Path(root) / split / "shard-000001.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row))
        handle.write("\n")


class KimiSftCliTest(unittest.TestCase):
    def test_load_api_key_from_file_without_logging_secret(self):
        with tempfile.TemporaryDirectory() as tmp:
            key_file = Path(tmp) / "key.md"
            key_file.write_text("# key\nsk-test_abcdefghijklmnopqrstuvwxyz\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"KIMI_API_KEY_FILE": str(key_file)}, clear=True):
                self.assertEqual(cli.load_api_key(), "sk-test_abcdefghijklmnopqrstuvwxyz")

    def test_missing_api_key_fails_clearly(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit) as raised:
                cli.load_api_key()
        self.assertIn("missing Kimi API key", str(raised.exception))

    def test_validate_reports_bad_jsonl_duplicate_and_split_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_row(tmp, "train", good_row("train", "dup", "shared"))
            write_row(tmp, "train", good_row("train", "dup", "unique"))
            write_row(tmp, "val", good_row("val", "val-1", "shared"))
            bad_path = Path(tmp) / "holdout" / "shard-000001.jsonl"
            bad_path.parent.mkdir(parents=True, exist_ok=True)
            bad_path.write_text("{bad\n", encoding="utf-8")
            report = cli.validate(tmp)
        joined = "\n".join(report["errors"])
        self.assertEqual(report["status"], "fail")
        self.assertIn("duplicate id", joined)
        self.assertIn("scenario family split leakage", joined)
        self.assertIn("bad JSONL", joined)

    def test_validate_rejects_disallowed_tool_and_multiple_action_blocks(self):
        row = good_row()
        row["messages"][1]["content"] = (
            "<action><tool>shell.exec</tool></action>"
            "<action><tool>agent.finish</tool></action>"
        )
        errors = cli.validate_row(row, "train", set())
        self.assertIn("assistant target must contain one action", errors)
        self.assertIn("missing assistant action target", errors)

    def test_validate_rejects_missing_confirmation(self):
        row = good_row()
        row["messages"][1]["content"] = "<action>\n<tool>resource.update_resource</tool>\n<ref>x</ref>\n</action>"
        row["meta"]["tool_sequence"] = ["resource.update_resource"]
        errors = cli.validate_row(row, "train", set())
        self.assertIn("mutation tool must be requested through confirmation", errors)

    def test_call_kimi_classifies_http_errors(self):
        for code, unauthorized, quota, retryable in [
            (401, True, False, False),
            (429, False, True, False),
            (500, False, False, True),
        ]:
            error = urllib_error(code)
            with mock.patch("urllib.request.urlopen", side_effect=error):
                result = cli.call_kimi("secret", {"timeout_seconds": 1}, [])
            self.assertEqual(result.unauthorized, unauthorized)
            self.assertEqual(result.quota_exhausted, quota)
            self.assertEqual(result.retryable, retryable)

    def test_call_kimi_classifies_access_terminated(self):
        error = urllib_error(403, b'{"error":{"type":"access_terminated_error"}}')
        with mock.patch("urllib.request.urlopen", side_effect=error):
            result = cli.call_kimi("secret", {"timeout_seconds": 1}, [])
        self.assertTrue(result.access_terminated)

    def test_promote_copies_only_after_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            quarantine = Path(tmp) / "quarantine"
            promoted = Path(tmp) / "promoted"
            write_row(quarantine, "train", good_row("train", "row-1", "family-1"))
            result = cli.promote(Namespace(quarantine=str(quarantine), promoted=str(promoted)))
            self.assertEqual(result["status"], "pass")
            self.assertTrue((promoted / "train" / "shard-000001.jsonl").is_file())
            self.assertTrue((promoted / "manifest.json").is_file())

    def test_validate_accepts_multi_turn_compaction_row(self):
        row = good_row()
        row["messages"] = [
            {"role": "system", "content": "Conversation summary: The user searched release notes and needs history next."},
            {"role": "user", "content": "Find release notes."},
            {
                "role": "assistant",
                "content": "<action>\n<reasoning>Search resources.</reasoning>\n<tool>resource.search</tool>\n<query>release notes</query>\n</action>",
            },
            {"role": "tool", "name": "resource.search", "content": "[{\"id\":\"release-notes\"}]"},
            {"role": "user", "content": "Show its history."},
            {
                "role": "assistant",
                "content": "<action>\n<reasoning>Fetch history for the matching resource.</reasoning>\n<tool>resource.history</tool>\n<ref>release-notes</ref>\n</action>",
            },
        ]
        row["tags"].append("compacted-context")
        row["meta"]["multi_turn"] = True
        row["meta"]["compaction"] = True
        row["meta"]["tool_sequence"] = ["resource.search", "resource.history"]
        self.assertEqual(cli.validate_row(row, "train", set()), [])

    def test_validate_rejects_incomplete_multi_turn_compaction_row(self):
        row = good_row()
        row["meta"]["multi_turn"] = True
        row["meta"]["compaction"] = True
        errors = cli.validate_row(row, "train", set())
        self.assertIn("multi-turn row requires at least two user turns", errors)
        self.assertIn("compaction row missing compacted-context tag", errors)
        self.assertIn("compaction row must start with system summary", errors)

    def test_call_kimi_cli_jobs_uses_runner_without_secret_argv(self):
        completed = subprocess_completed(
            '{"job_id":"job-1","status":"pass","text":"{\\"rows\\":[]}","error":"","attempts":1,"elapsed_ms":1}\n'
        )
        with mock.patch("subprocess.run", return_value=completed) as run:
            result = call_kimi_cli_jobs(
                "sk-secret_should_not_be_in_argv",
                {"api_provider": "kimi-cli", "kimi_cli_runner": "/bin/kimi-runner", "parallelism": 2},
                [{"job_id": "job-1", "ordinal": 1, "messages": [], "input_jsonl": ""}],
            )
        argv = run.call_args.args[0]
        self.assertNotIn("sk-secret_should_not_be_in_argv", " ".join(argv))
        self.assertEqual(run.call_args.kwargs["env"]["KIMI_API_KEY"], "sk-secret_should_not_be_in_argv")
        self.assertEqual(result[0].status, "pass")

    def test_pilot_gate_caps_full_target_until_approved(self):
        config = {"target_tokens": 60000000, "pilot_tokens": 1000000}
        with mock.patch.dict(os.environ, {}, clear=True):
            apply_pilot_gate(config)
        self.assertEqual(config["target_tokens"], 1000000)
        self.assertEqual(config["full_target_tokens"], 60000000)
        self.assertTrue(config["pilot_gate_required"])


def urllib_error(code, body=b'{"error":"redacted"}'):
    import urllib.error

    return urllib.error.HTTPError(
        "https://api.moonshot.ai/v1/chat/completions",
        code,
        "error",
        {},
        mock.Mock(read=lambda _n=-1: body),
    )


def subprocess_completed(stdout):
    return mock.Mock(returncode=0, stdout=stdout, stderr="")


if __name__ == "__main__":
    unittest.main()
