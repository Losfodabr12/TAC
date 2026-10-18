import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path

import psutil


def setup_logging(data: Path):
    logger = logging.getLogger("tac." + str(data.resolve()))
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    handler = RotatingFileHandler(data / "logs" / "tac.log", maxBytes=2000000,
                                  backupCount=4, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def resources():
    mem = psutil.virtual_memory()
    process = psutil.Process(os.getpid())
    return {"cpu_percent": psutil.cpu_percent(), "ram_available_mb": mem.available / 2**20,
            "ram_percent": mem.percent, "process_mb": process.memory_info().rss / 2**20}


def check_resources(config):
    info = resources()
    if info["process_mb"] > config.get("ram_mb"):
        raise MemoryError("Limite de RAM da TAC atingido. Reduza lote/contexto ou feche tarefas.")
    if info["ram_available_mb"] < config.get("reserve_ram_mb"):
        raise MemoryError("Pouca RAM livre para o Windows. Operação interrompida com segurança.")
    return info
