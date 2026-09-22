import pytest
import yaml

from xiaolinrag.config import ConfigError, load_config, redact

EMBED_KEY = "sk-testembed1234567890"
CHAT_KEY = "user_testchat1234567890"


def config_data(kb_dir) -> dict:
    return {
        "kb_dir": str(kb_dir),
        "index_dir": "data/index",
        "models": {
            "embedding_base_url": "https://api.siliconflow.cn/v1",
            "embedding_model": "Qwen/Qwen3-Embedding-8B",
            "rerank_base_url": "https://api.siliconflow.cn/v1",
            "rerank_model": "BAAI/bge-reranker-v2-m3",
            "chat_base_url": "https://opencode.ai/zen/go/v1",
            "chat_model": "deepseek/deepseek-v4-flash",
            "judge_model": "claude-haiku-4-5-20251001",
            "chat_temperature": 0.3,
        },
        "chunking": {
            "child_target": 300,
            "child_max": 500,
            "parent_target": 900,
            "parent_max": 1200,
            "context_max_chars": 100,
        },
        "contextual": {"enabled": True, "concurrency": 4, "interval_seconds": 0.5},
        "retrieval": {
            "vec_top_k": 20,
            "bm25_top_k": 20,
            "rrf_k": 60,
            "rerank_top_n": 40,
            "final_n": 5,
            "gate_threshold": -8.0,
        },
        "webui": {"server_name": "127.0.0.1", "server_port": 7860},
    }


@pytest.fixture
def project(tmp_path):
    """在临时目录搭一套 config.yaml + .env + kb 目录。"""
    kb = tmp_path / "kb"
    kb.mkdir()
    (tmp_path / ".env").write_text(
        f"SILICONFLOW_API_KEY={EMBED_KEY}\nCHAT_API_KEY={CHAT_KEY}\n",
        encoding="utf-8",
    )
    return tmp_path


def write_yaml(project, data) -> None:
    (project / "config.yaml").write_text(
        yaml.safe_dump(data, allow_unicode=True), encoding="utf-8"
    )


def test_load_valid_config(project, monkeypatch):
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.delenv("CHAT_API_KEY", raising=False)
    write_yaml(project, config_data(project / "kb"))

    cfg = load_config(project / "config.yaml")

    assert cfg.kb_dir == project / "kb"
    assert cfg.index_dir == project / "data" / "index"  # 相对路径按配置目录解析
    assert cfg.models.embedding_model == "Qwen/Qwen3-Embedding-8B"
    assert cfg.models.chat_temperature == 0.3
    assert cfg.models.embedding_api_key == EMBED_KEY
    assert cfg.chunking.child_target == 300
    assert cfg.contextual.enabled is True
    assert cfg.retrieval.final_n == 5
    assert cfg.webui.server_port == 7860


def test_missing_config_file(tmp_path):
    with pytest.raises(ConfigError, match="未找到配置文件"):
        load_config(tmp_path / "config.yaml")


def test_missing_api_key(project, monkeypatch):
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.delenv("CHAT_API_KEY", raising=False)
    (project / ".env").write_text("", encoding="utf-8")
    write_yaml(project, config_data(project / "kb"))

    with pytest.raises(ConfigError, match="SILICONFLOW_API_KEY"):
        load_config(project / "config.yaml")


def test_env_var_overrides_dotenv(project, monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-from-env")
    write_yaml(project, config_data(project / "kb"))

    cfg = load_config(project / "config.yaml")

    assert cfg.models.embedding_api_key == "sk-from-env"


def test_kb_dir_missing(project, monkeypatch):
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.delenv("CHAT_API_KEY", raising=False)
    data = config_data(project / "not-created")
    write_yaml(project, data)

    with pytest.raises(ConfigError, match="目录不存在"):
        load_config(project / "config.yaml")


def test_child_max_must_be_less_than_parent_max(project, monkeypatch):
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.delenv("CHAT_API_KEY", raising=False)
    data = config_data(project / "kb")
    data["chunking"]["child_max"] = 1200
    write_yaml(project, data)

    with pytest.raises(ConfigError, match="child_max"):
        load_config(project / "config.yaml")


def test_final_n_exceeds_rerank_top_n(project, monkeypatch):
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.delenv("CHAT_API_KEY", raising=False)
    data = config_data(project / "kb")
    data["retrieval"]["final_n"] = 50
    write_yaml(project, data)

    with pytest.raises(ConfigError, match="final_n"):
        load_config(project / "config.yaml")


def test_missing_section(project, monkeypatch):
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.delenv("CHAT_API_KEY", raising=False)
    data = config_data(project / "kb")
    del data["retrieval"]
    write_yaml(project, data)

    with pytest.raises(ConfigError, match="retrieval"):
        load_config(project / "config.yaml")


def test_redacted_hides_keys(project, monkeypatch):
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.delenv("CHAT_API_KEY", raising=False)
    write_yaml(project, config_data(project / "kb"))
    cfg = load_config(project / "config.yaml")

    dump = str(cfg.redacted())

    assert EMBED_KEY not in dump
    assert CHAT_KEY not in dump
    assert "sk-tes***" in dump


def test_redact_empty():
    assert redact("") == "(未设置)"


# ---------- multi_query ----------


def test_multi_query_section_loaded(project, monkeypatch):
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.delenv("CHAT_API_KEY", raising=False)
    data = config_data(project / "kb")
    data["multi_query"] = {"enabled": False, "num_queries": 3, "temperature": 0.0}
    write_yaml(project, data)

    cfg = load_config(project / "config.yaml")

    assert cfg.multi_query.enabled is False
    assert cfg.multi_query.num_queries == 3
    assert cfg.multi_query.temperature == 0.0


def test_multi_query_defaults_when_section_missing(project, monkeypatch):
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.delenv("CHAT_API_KEY", raising=False)
    write_yaml(project, config_data(project / "kb"))  # config_data 不含 multi_query 小节

    cfg = load_config(project / "config.yaml")

    assert cfg.multi_query.enabled is True
    assert cfg.multi_query.num_queries == 4
    assert cfg.multi_query.temperature == 0.2


def test_multi_query_num_queries_must_be_positive(project, monkeypatch):
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.delenv("CHAT_API_KEY", raising=False)
    data = config_data(project / "kb")
    data["multi_query"] = {"num_queries": 0}
    write_yaml(project, data)

    with pytest.raises(ConfigError, match="num_queries"):
        load_config(project / "config.yaml")


def test_multi_query_temperature_out_of_range(project, monkeypatch):
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.delenv("CHAT_API_KEY", raising=False)
    data = config_data(project / "kb")
    data["multi_query"] = {"temperature": 1.5}
    write_yaml(project, data)

    with pytest.raises(ConfigError, match="temperature"):
        load_config(project / "config.yaml")
