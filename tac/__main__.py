from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from .config import ROOT, atomic_json


def self_check(core):
    import sqlite3
    import torch
    from .neural.model import ModelConfig, TACTransformer
    from .neural.tokenizer import ByteTokenizer
    with sqlite3.connect(":memory:") as c:
        c.execute("CREATE VIRTUAL TABLE fts_test USING fts5(text)")
    tokenizer = ByteTokenizer()
    text = "TAC: inteligência local, ação e memória."
    assert tokenizer.decode(tokenizer.encode(text)) == text
    model = TACTransformer(ModelConfig(width=32, heads=2, layers=1, context=32))
    tokens = torch.tensor([tokenizer.encode("TAC local")])
    logits, loss = model(tokens[:, :-1], tokens[:, 1:])
    assert torch.isfinite(loss)
    result = {"python": sys.version, "torch": torch.__version__, "cuda": torch.cuda.is_available(),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "sqlite": sqlite3.sqlite_version, "fts5": True, "tokenizer": True,
              "neural_forward_loss": float(loss.detach()), "learning_enabled": core.gate.enabled,
              "auto_train": core.config.get("auto_train"), "resources": core.stats(),
              "root": str(core.config.root), "data": str(core.config.data), "status": "ok"}
    atomic_json(core.config.data / "diagnostics.json", result)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description="TAC — aplicação local")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--smoke-ui", action="store_true")
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--test-root", type=Path)
    args = parser.parse_args()
    root = args.test_root.resolve() if args.test_root else ROOT
    if not root.is_relative_to(ROOT):
        parser.error("O diretório de teste deve estar dentro de TAC.")
    from .core import Core
    core = Core(root)
    if args.self_check:
        return self_check(core)
    from PySide6.QtCore import QLockFile
    from .ui import launch
    lock = QLockFile(str(core.config.data / "tac.lock"))
    if not lock.tryLock(0):
        from PySide6.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance() or QApplication([])
        QMessageBox.information(None, "TAC", "A TAC já está aberta usando esta pasta de dados.")
        return 1
    core.recover_interrupted()
    def exception_hook(kind, value, tb):
        core.logger.error("Exceção não tratada", exc_info=(kind, value, tb))
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(None, "TAC — erro", f"{kind.__name__}: {value}\nDetalhes em data/logs/tac.log.")
    sys.excepthook = exception_hook
    try:
        return launch(core, args.smoke_ui, args.screenshot)
    finally:
        lock.unlock()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        (ROOT / "startup_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        if os.name == "nt":
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, "Falha ao iniciar a TAC. Consulte startup_error.log ou execute DIAGNOSTICO_TAC.bat.", "TAC", 16)
        raise
