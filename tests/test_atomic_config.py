import json
from concurrent.futures import ThreadPoolExecutor

from tac.config import Config, atomic_json


def test_concurrent_json_writers_use_unique_files(tmp_path):
    path = tmp_path / "progress.json"
    def write(index):
        for step in range(20):
            atomic_json(path, {"worker": index, "step": step, "text": "x" * 200})
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write, range(4)))
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value['step'] == 19 and value['worker'] in range(4)
    assert not list(tmp_path.glob('*.tmp'))


def test_reopening_existing_config_does_not_write(core, monkeypatch):
    def unexpected_write(*_):
        raise AssertionError("Consultar configuração existente não deve regravá-la")
    monkeypatch.setattr('tac.config.atomic_json', unexpected_write)
    reopened = Config(core.config.root)
    assert reopened.values == core.config.values
