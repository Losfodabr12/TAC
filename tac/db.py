from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
INSERT OR IGNORE INTO meta VALUES('schema_version','1');
CREATE TABLE IF NOT EXISTS sources(
 id INTEGER PRIMARY KEY,title TEXT NOT NULL,origin TEXT NOT NULL UNIQUE,
 kind TEXT NOT NULL,content_hash TEXT NOT NULL,subject TEXT NOT NULL DEFAULT '',
 created TEXT NOT NULL,updated TEXT NOT NULL,metadata TEXT NOT NULL DEFAULT '{}');
CREATE TABLE IF NOT EXISTS chunks(
 id INTEGER PRIMARY KEY,source_id INTEGER NOT NULL REFERENCES sources ON DELETE CASCADE,
 ordinal INTEGER NOT NULL,section TEXT NOT NULL,text TEXT NOT NULL);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text,section,
 content='chunks',content_rowid='id',tokenize='unicode61 remove_diacritics 2');
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
 INSERT INTO chunks_fts(rowid,text,section) VALUES(new.id,new.text,new.section); END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
 INSERT INTO chunks_fts(chunks_fts,rowid,text,section) VALUES('delete',old.id,old.text,old.section); END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
 INSERT INTO chunks_fts(chunks_fts,rowid,text,section) VALUES('delete',old.id,old.text,old.section);
 INSERT INTO chunks_fts(rowid,text,section) VALUES(new.id,new.text,new.section); END;
CREATE INDEX IF NOT EXISTS chunks_source ON chunks(source_id);
CREATE INDEX IF NOT EXISTS sources_hash ON sources(content_hash);
CREATE TABLE IF NOT EXISTS queue(
 id INTEGER PRIMARY KEY,kind TEXT NOT NULL,target TEXT NOT NULL,options TEXT NOT NULL,
 status TEXT NOT NULL DEFAULT 'pendente',progress TEXT NOT NULL DEFAULT '',
 created TEXT NOT NULL,finished TEXT);
CREATE TABLE IF NOT EXISTS study_history(
 id INTEGER PRIMARY KEY,queue_id INTEGER REFERENCES queue ON DELETE SET NULL,
 origin TEXT NOT NULL,status TEXT NOT NULL,detail TEXT NOT NULL,created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS interactions(
 id INTEGER PRIMARY KEY,question TEXT NOT NULL,answer TEXT NOT NULL,mode TEXT NOT NULL,
 sources TEXT NOT NULL,created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS corrections(
 id INTEGER PRIMARY KEY,original TEXT NOT NULL,correction TEXT NOT NULL,
 subject TEXT NOT NULL,source TEXT NOT NULL,validated INTEGER NOT NULL DEFAULT 0,created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs(
 id TEXT PRIMARY KEY,kind TEXT NOT NULL,status TEXT NOT NULL,started TEXT NOT NULL,
 finished TEXT,config TEXT NOT NULL,dataset_hash TEXT NOT NULL,
 baseline REAL,final REAL,detail TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS metrics(
 id INTEGER PRIMARY KEY,run_id TEXT NOT NULL REFERENCES runs ON DELETE CASCADE,
 step INTEGER NOT NULL,epoch INTEGER NOT NULL,train_loss REAL,val_loss REAL,
 fixed_loss REAL,accuracy REAL,created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS checkpoints(
 id TEXT PRIMARY KEY,run_id TEXT NOT NULL REFERENCES runs ON DELETE CASCADE,
 filename TEXT NOT NULL,step INTEGER NOT NULL,val_loss REAL,status TEXT NOT NULL,
 created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS rl_attempts(
 id INTEGER PRIMARY KEY,run_id TEXT NOT NULL REFERENCES runs ON DELETE CASCADE,
 step INTEGER NOT NULL,a INTEGER NOT NULL,b INTEGER NOT NULL,action INTEGER NOT NULL,
 reward REAL NOT NULL,created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events(
 id INTEGER PRIMARY KEY,kind TEXT NOT NULL,detail TEXT NOT NULL,created TEXT NOT NULL);
"""

DIALOGUE_SCHEMA = """
CREATE TABLE IF NOT EXISTS dialogue_imports(
 id TEXT PRIMARY KEY,filename TEXT NOT NULL,origin TEXT NOT NULL,content_hash TEXT NOT NULL,
 tokenizer_hash TEXT NOT NULL,report TEXT NOT NULL,enabled INTEGER NOT NULL DEFAULT 1,created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS dialogue_groups(
 id TEXT PRIMARY KEY,split TEXT NOT NULL CHECK(split IN ('train','validation','test')),created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS dialogue_examples(
 id INTEGER PRIMARY KEY,import_id TEXT NOT NULL REFERENCES dialogue_imports ON DELETE CASCADE,
 group_id TEXT NOT NULL REFERENCES dialogue_groups,skill TEXT NOT NULL,messages TEXT NOT NULL,
 answer TEXT NOT NULL,prompt_hash TEXT NOT NULL,example_hash TEXT NOT NULL UNIQUE,
 episode TEXT NOT NULL,turn INTEGER NOT NULL,source_line INTEGER NOT NULL,created TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS dialogue_examples_group ON dialogue_examples(group_id);
CREATE INDEX IF NOT EXISTS dialogue_examples_prompt ON dialogue_examples(prompt_hash);
CREATE INDEX IF NOT EXISTS dialogue_examples_import ON dialogue_examples(import_id);
"""


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        existed = self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        with self.connect() as c:
            c.executescript("PRAGMA journal_mode=WAL;" + SCHEMA)
            version = c.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
        if version == '1':
            if existed:
                import uuid
                target = self.path.parent/'backups'/('antes_schema2_' + uuid.uuid4().hex + '.sqlite3')
                target.parent.mkdir(parents=True, exist_ok=True)
                self.backup_to(target)
            with self.lock, self.connect() as c:
                c.executescript("BEGIN IMMEDIATE;\n" + DIALOGUE_SCHEMA + "\nUPDATE meta SET value='2' WHERE key='schema_version';\nCOMMIT;")
        elif version != '2':
            raise RuntimeError("Versão do banco incompatível. Preserve o backup e atualize a TAC.")

    @contextlib.contextmanager
    def connect(self):
        c = sqlite3.connect(self.path, timeout=15)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        try:
            yield c
            c.commit()
        except BaseException:
            c.rollback()
            raise
        finally:
            c.close()

    def query(self, sql, args=()):
        with self.connect() as c:
            return [dict(r) for r in c.execute(sql, args).fetchall()]

    def iterate(self, sql, args=()):
        with self.connect() as c:
            cursor = c.execute(sql, args)
            while rows := cursor.fetchmany(32):
                for row in rows:
                    yield dict(row)

    def execute(self, sql, args=()):
        with self.lock, self.connect() as c:
            return c.execute(sql, args).lastrowid

    def event(self, kind, detail):
        self.execute("INSERT INTO events(kind,detail,created) VALUES(?,?,?)", (kind, detail, now()))

    def backup_to(self, target: Path):
        with self.lock, self.connect() as src, contextlib.closing(sqlite3.connect(target)) as dest:
            src.backup(dest)
            dest.commit()

    def dump_json(self, value):
        return json.dumps(value, ensure_ascii=False)
