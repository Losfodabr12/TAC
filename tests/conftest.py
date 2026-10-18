import os
import sys
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["TAC_TEST_MODE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
(Path(__file__).resolve().parents[1] / "reports").mkdir(exist_ok=True)

import pytest

from tac.core import Core


@pytest.fixture
def core(tmp_path):
    value = Core(tmp_path / "TAC")
    value.config.update(reserve_ram_mb=256, ram_mb=4096, cpu_threads=2)
    return value


@pytest.fixture(scope="session")
def qtapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app
