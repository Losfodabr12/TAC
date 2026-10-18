"""Inventário dos pesos locais ativos, sem somar redes com funções diferentes."""
import torch


def active_models(config, db):
    models = []
    folder = (config.data / "checkpoints").resolve()
    rows = db.query("SELECT m.key,cp.id,cp.filename FROM meta m JOIN checkpoints cp ON cp.id=m.value WHERE m.key IN ('active_checkpoint','active_dialogue','active_conversation') ORDER BY m.key")
    for row in rows:
        path = (folder / row["filename"]).resolve()
        if not path.is_relative_to(folder) or path.suffix != ".pt" or path.stat().st_size > 300 * 2**20:
            raise ValueError("Arquivo do modelo ativo fora dos limites.")
        value = torch.load(path, map_location="cpu", weights_only=True)
        if value.get("format") not in ("TAC-v1", "TAC-dialogue-v1", "TAC-chat-v2"):
            raise ValueError("Formato de modelo desconhecido.")
        count = sum(t.numel() for t in value["model"].values())
        if value.get("format") == "TAC-chat-v2":
            from .model import ModelConfig, count_parameters
            count = count_parameters(ModelConfig(**value["model_config"]))
        models.append({"checkpoint": row["id"],
                       "name": {"active_checkpoint": "Transformer de texto", "active_dialogue": "Reconhecimento de intenções", "active_conversation": "Transformer de diálogo (variante)"}[row["key"]],
                       "parameters": count})
    return models
