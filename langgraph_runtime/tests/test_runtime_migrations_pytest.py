from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory


def test_runtime_migrations_have_one_runtime_owned_head():
    root = Path(__file__).resolve().parents[1]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "migrations"))
    scripts = ScriptDirectory.from_config(config)
    assert set(scripts.get_heads()) == {"r1_runtime_schema"}
