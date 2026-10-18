from __future__ import annotations

import json
import os
import threading
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULTS = {
    "learning_enabled": False, "auto_train": False, "auto_on_start": False,
    "save_interactions": False, "data_dir": "data", "theme": "dark", "font_size": 13,
    "ram_mb": 3072, "reserve_ram_mb": 768, "vram_mb": 3072,
    "cpu_threads": 4, "device": "auto", "batch_size": 8, "accumulation": 2,
    "context": 128, "width": 96, "layers": 2, "heads": 4, "ff_width": 0,
    "learning_rate": 0.001, "retention_weight": 0.0, "epochs": 3, "steps_per_epoch": 40,
    "train_seconds": 120, "auto_interval_minutes": 60, "seed": 42,
    "conversation_profile": "10m", "conversation_steps": 6000, "conversation_seconds": 600,
    'conversation_curriculum': 'fluency',
    'conversation_learning_rate': .00002, 'conversation_stability': 'off',
    'conversation_sampler': 'replacement', 'conversation_generation': 'greedy',
    'conversation_document_dataset': '',
    'conversation_transfer_practice': False,
    "max_file_mb": 20, "max_text_chars": 1000000, "max_corpus_chars": 2000000,
    "web_depth": 1, "web_pages": 8, "web_page_mb": 2,
    "web_interval": 2.0, "backup_hours": 24, "backup_keep": 5,
    "research_auto": False, "research_pages": 3, "research_chars": 60000,
    "research_seconds": 90,
}


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name + ".", suffix=".tmp", delete=False) as f:
            temp = Path(f.name)
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        deadline = time.monotonic() + 1
        while True:
            try:
                os.replace(temp, path)
                break
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.01)  # Leitores breves podem bloquear rename no Windows.
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


class Config:
    def __init__(self, root: Path = ROOT):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "config.json"
        self.lock = threading.RLock()
        self.values = dict(DEFAULTS)
        saved = None
        if self.path.exists():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            self.values.update(saved)
        self.validate(self.values)
        if saved != self.values:
            self.save()
        for name in ("logs", "checkpoints", "inbox", "datasets", "backups"):
            (self.data / name).mkdir(parents=True, exist_ok=True)

    @property
    def data(self):
        path = (self.root / self.values["data_dir"]).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError("O diretório de dados deve ser uma subpasta da TAC.")
        return path

    def get(self, key):
        with self.lock:
            return self.values[key]

    def update(self, **values):
        with self.lock:
            candidate = {**self.values, **values}
            self.validate(candidate)
            old = self.values
            self.values = candidate
            try:
                self.save()
            except Exception:
                self.values = old
                raise

    def save(self):
        with self.lock:
            atomic_json(self.path, self.values)

    def validate(self, v):
        limits = {"font_size": (11, 18), "ram_mb": (512, 8192), "reserve_ram_mb": (256, 8192),
                  "vram_mb": (256, 4608), "cpu_threads": (1, 8),
                  "batch_size": (1, 32), "accumulation": (1, 16),
                  "context": (16, 256), "width": (32, 256), "layers": (1, 4),
                  "heads": (1, 8), "ff_width": (0, 2048), "epochs": (1, 100), "steps_per_epoch": (1, 1000),
                  "train_seconds": (1, 3600), "auto_interval_minutes": (1, 1440),
                  "conversation_steps": (100, 60000), "conversation_seconds": (1, 3600),
                  "max_file_mb": (1, 100), "max_text_chars": (1000, 2000000),
                  "max_corpus_chars": (1000, 8000000), "web_depth": (0, 4),
                  "web_pages": (1, 100), "web_page_mb": (1, 8),
                  "web_interval": (1, 60), "backup_hours": (1, 720),
                  "backup_keep": (1, 20), "learning_rate": (0.00001, 0.02), "retention_weight": (0, 50),
                  "research_pages": (1, 5), "research_chars": (2000, 100000), "research_seconds": (10, 180),
                  'conversation_learning_rate':(.00001,.003)}
        for key, (low, high) in limits.items():
            if not isinstance(v[key], (int, float)) or not low <= v[key] <= high:
                raise ValueError(f"{key}: valor permitido entre {low} e {high}.")
        for key in ("learning_enabled", "auto_train", "auto_on_start", "save_interactions", "research_auto",'conversation_transfer_practice'):
            if not isinstance(v[key], bool):
                raise ValueError(f"{key} deve ser booleano.")
        if v["width"] % v["heads"]:
            raise ValueError("A dimensão deve ser divisível pelo número de cabeças.")
        if v["conversation_profile"] not in ("10m", "current"):
            raise ValueError("Perfil de diálogo inválido")
        if v['conversation_curriculum'] not in ('fluency','legacy'):
            raise ValueError('Currículo de conversa inválido')
        if v['conversation_stability'] not in ('off','anchor_v1') or v['conversation_sampler'] not in ('replacement','deck_v1'):
            raise ValueError('Estratégia experimental de conversa inválida')
        if v['conversation_generation'] not in ('greedy','controlled'):
            raise ValueError('Geração de conversa inválida')
        if not isinstance(v['conversation_document_dataset'],str) or (v['conversation_document_dataset'] and
                (len(v['conversation_document_dataset'])!=64 or any(c not in '0123456789abcdef' for c in v['conversation_document_dataset']))):
            raise ValueError('Identificador dos textos opcionais inválido')
        if v["device"] not in ("auto", "cpu", "cuda") or v["theme"] not in ("dark", "light"):
            raise ValueError("Dispositivo ou tema inválido.")
        data = (self.root / v["data_dir"]).resolve()
        if not data.is_relative_to(self.root) or data == self.root:
            raise ValueError("Mantenha os dados em uma subpasta da TAC.")
