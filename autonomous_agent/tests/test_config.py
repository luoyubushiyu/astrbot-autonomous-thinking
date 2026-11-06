from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from autonomous_agent.config import (
    AppConfig,
    ConfigError,
    load_config,
    parse_quiet_window,
)


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.path = self.dir / "config.json"

    def write(self, data: dict) -> None:
        self.path.write_text(json.dumps(data), encoding="utf-8")

    def test_defaults_without_file(self) -> None:
        config = load_config(None, env={"AGENT_CONFIG": str(self.dir / "absent.json")})
        self.assertIsInstance(config, AppConfig)
        self.assertEqual(config.llm.provider, "mock")
        self.assertEqual(config.qq.adapter, "mock")
        self.assertTrue(config.database.store_events)

    def test_file_values(self) -> None:
        self.write(
            {
                "llm": {"provider": "mock", "model": "x-model"},
                "database": {"path": "sub/agent.db"},
                "behavior": {"quiet_hours": ["23:00-07:00"]},
            }
        )
        config = load_config(self.path, env={})
        self.assertEqual(config.llm.model, "x-model")
        self.assertTrue(config.database.path.endswith(str(Path("sub") / "agent.db")))
        self.assertEqual(config.behavior.quiet_hours, ["23:00-07:00"])

    def test_env_overrides_file(self) -> None:
        self.write({"llm": {"model": "file-model"}})
        config = load_config(
            self.path,
            env={
                "AGENT_LLM_MODEL": "env-model",
                "AGENT_DB_PATH": "env.db",
                "AGENT_SCHEDULER_ENABLED": "false",
            },
        )
        self.assertEqual(config.llm.model, "env-model")
        self.assertTrue(config.database.path.endswith("env.db"))
        self.assertFalse(config.scheduler.enabled)

    def test_deepseek_api_key_alias(self) -> None:
        config = load_config(
            None,
            env={
                "AGENT_CONFIG": str(self.dir / "absent.json"),
                "DEEPSEEK_API_KEY": "sk-test",
            },
        )
        self.assertEqual(config.llm.api_key, "sk-test")

    def test_missing_explicit_file(self) -> None:
        with self.assertRaises(ConfigError):
            load_config(self.dir / "nope.json", env={})

    def test_unknown_section_key(self) -> None:
        self.write({"llm": {"unknown": 1}})
        with self.assertRaises(ConfigError):
            load_config(self.path, env={})

    def test_bad_provider(self) -> None:
        self.write({"llm": {"provider": "anthropic"}})
        with self.assertRaises(ConfigError):
            load_config(self.path, env={})

    def test_bad_type(self) -> None:
        self.write({"scheduler": {"enabled": "yes"}})
        with self.assertRaises(ConfigError):
            load_config(self.path, env={})

    def test_provider_presets(self) -> None:
        self.write({"llm": {"provider": "openai"}})
        config = load_config(self.path, env={})
        self.assertEqual(config.llm.base_url, "https://api.openai.com/v1")
        self.assertEqual(config.llm.model, "gpt-4o-mini")

        self.write({"llm": {"provider": "deepseek"}})
        config = load_config(self.path, env={})
        self.assertEqual(config.llm.base_url, "https://api.deepseek.com/v1")
        self.assertEqual(config.llm.model, "deepseek-chat")

    def test_custom_base_url_survives_preset(self) -> None:
        self.write(
            {
                "llm": {
                    "provider": "openai",
                    "base_url": "http://localhost:8000/v1",
                    "model": "local-model",
                }
            }
        )
        config = load_config(self.path, env={})
        self.assertEqual(config.llm.base_url, "http://localhost:8000/v1")
        self.assertEqual(config.llm.model, "local-model")

    def test_json_mode_env(self) -> None:
        config = load_config(
            None,
            env={
                "AGENT_CONFIG": str(self.dir / "absent.json"),
                "AGENT_LLM_JSON_MODE": "false",
            },
        )
        self.assertFalse(config.llm.json_mode)

    def test_empty_base_url_rejected(self) -> None:
        self.write({"llm": {"provider": "openai_compatible", "base_url": ""}})
        with self.assertRaises(ConfigError):
            load_config(self.path, env={})

    def test_quiet_window_parse(self) -> None:
        self.assertEqual(parse_quiet_window("23:30-07:05"), (23, 30, 7, 5))
        with self.assertRaises(ConfigError):
            parse_quiet_window("25:00-26:00")
        with self.assertRaises(ConfigError):
            parse_quiet_window("nonsense")

    def test_mask_secrets(self) -> None:
        config = AppConfig()
        config.llm.api_key = "sk-secret"
        self.assertEqual(config.to_dict()["llm"]["api_key"], "***")
        self.assertEqual(config.to_dict(mask_secrets=False)["llm"]["api_key"], "sk-secret")


if __name__ == "__main__":
    unittest.main()