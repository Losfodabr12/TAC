from __future__ import annotations

import hashlib
import contextlib
import json
import shutil
import sqlite3
import tempfile
import uuid
import zipfile
from pathlib import Path

from .config import atomic_json
from .db import now


def relocate_data(core, relative):
    current = core.config.data.resolve()
    destination = (core.config.root / relative).resolve()
    if (not destination.is_relative_to(core.config.root) or destination == core.config.root
            or destination.is_relative_to(current) or current.is_relative_to(destination)):
        raise ValueError("Escolha outra subpasta da TAC, fora da pasta atual de dados.")
    if destination.exists():
        raise ValueError("A pasta de destino já existe. Escolha um nome novo ou importe um backup.")
    stage = core.config.root / ("migration_" + uuid.uuid4().hex)
    try:
        shutil.copytree(current, stage, ignore=shutil.ignore_patterns("tac.sqlite3", "tac.sqlite3-wal", "tac.sqlite3-shm", "tac.lock"))
        core.db.backup_to(stage / "tac.sqlite3")
        for name in ("tac.sqlite3-wal", "tac.sqlite3-shm", "tac.lock"):
            (stage / name).unlink(missing_ok=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(destination)
        core.config.update(data_dir=relative)
        return str(destination)
    except Exception:
        if stage.exists() and stage.resolve().is_relative_to(core.config.root) and stage.name.startswith("migration_"):
            shutil.rmtree(stage)
        raise


def export_backup(core, target):
    """Chamado com fila/treino inativos para incluir pesos e banco consistentes."""
    target = Path(target).resolve()
    if not target.is_relative_to(core.config.root):
        raise ValueError("Salve o backup dentro da pasta TAC.")
    if target.exists():
        raise ValueError("Já existe um backup com esse nome; escolha outro nome.")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=core.config.data / "backups") as temp:
        temp = Path(temp)
        core.db.backup_to(temp / "tac.sqlite3")
        members = [(temp / "tac.sqlite3", "tac.sqlite3")]
        manifest = {"format": "TAC-backup-v1", "created": now(), "files": {}}
        for folder in ("checkpoints", "datasets", "inbox", "books", "dialogue_imports"):
            directory = core.config.data / folder
            if directory.is_dir():
                members.extend((p, f"{folder}/{p.name}") for p in directory.iterdir() if p.is_file() and p.suffix != ".tmp")
        atomic_json(temp / "config.json", core.config.values)
        members.append((temp / "config.json", "config.json"))
        output = target.with_suffix(".tmp")
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for path, name in members:
                digest = hashlib.sha256()
                with path.open("rb") as f:
                    while block := f.read(1024 * 1024):
                        digest.update(block)
                manifest["files"][name] = digest.hexdigest()
                archive.write(path, name)
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False))
        output.replace(target)
    core.db.event("backup_exportado", str(target))
    return str(target)


def import_backup(core, source):
    """Valida e restaura em NOVA subpasta. Dados anteriores permanecem intactos."""
    source = Path(source)
    destination = core.config.root / ("data_restaurado_" + now().replace(":", "").replace("+", "_"))
    if destination.exists():
        raise ValueError("Destino de restauração já existe; tente novamente em um segundo.")
    destination.mkdir()
    try:
        with zipfile.ZipFile(source) as archive:
            infos = archive.infolist()
            if len(infos) > 20000 or sum(i.file_size for i in infos) > 4 * 2**30:
                raise ValueError("Backup excede limites de restauração (4 GB/20 mil arquivos).")
            names = [i.filename for i in infos]
            if len(names) != len(set(names)):
                raise ValueError("Backup contém nomes duplicados.")
            manifest = json.loads(archive.read("manifest.json"))
            if manifest.get("format") != "TAC-backup-v1":
                raise ValueError("Formato de backup incompatível.")
            expected = manifest["files"]
            if set(names) != set(expected) | {"manifest.json"}:
                raise ValueError("Manifesto de backup inconsistente.")
            for name, digest in expected.items():
                if "\\" in name or ":" in name or name.startswith("/") or ".." in Path(name).parts:
                    raise ValueError("Caminho inválido no backup.")
                target = (destination / name).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise ValueError("Caminho fora da pasta de restauração.")
                if name not in ("tac.sqlite3", "config.json") and not (name.count("/") == 1 and name.split("/")[0] in ("checkpoints", "datasets", "inbox", "books", "dialogue_imports")):
                    raise ValueError("Arquivo não autorizado no backup.")
                target.parent.mkdir(parents=True, exist_ok=True)
                check = hashlib.sha256()
                with archive.open(name) as inp, target.open("wb") as out:
                    while block := inp.read(1024 * 1024):
                        check.update(block)
                        out.write(block)
                if check.hexdigest() != digest:
                    raise ValueError("Arquivo de backup corrompido: " + name)
        # Não carrega objetos pickle durante importação. Checkpoints exigem weights_only=True.
        with contextlib.closing(sqlite3.connect(destination / "tac.sqlite3")) as c:
            if c.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Banco corrompido.")
            if c.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] not in ('1','2'):
                raise ValueError("Versão do banco incompatível.")
            # Caminhos da fila são relativos ao inbox; restauração não depende do local original.
            c.execute("UPDATE queue SET status='pendente' WHERE status='processando'")
            c.commit()
        # Segurança: importar preserva conhecimentos, mas não concede novas permissões.
        restored = json.loads((destination / "config.json").read_text(encoding="utf-8"))
        from .config import DEFAULTS
        candidate = {**DEFAULTS, **restored, "data_dir": destination.name,
                     "learning_enabled": False, "auto_train": False, "auto_on_start": False, "research_auto": False,
                     "save_interactions": False}
        core.config.validate(candidate)
        core.config.update(**candidate)
        return str(destination)
    except Exception:
        # Remove apenas a pasta recém-criada e verificada dentro da TAC.
        if destination.resolve().is_relative_to(core.config.root) and destination.name.startswith("data_restaurado_"):
            shutil.rmtree(destination)
        raise
