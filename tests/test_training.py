import pytest
import torch

from tac.learning import LearningDisabled
from tac.neural.data import Corpus
from tac.neural.model import ModelConfig, TACTransformer
from tac.neural.tokenizer import ByteTokenizer
from tac.neural.training import safe_load, weight_hash


def setup_training(core, steps=20):
    core.config.update(device="cpu", width=32, heads=2, layers=1, context=32,
                       batch_size=4, accumulation=1, epochs=1, steps_per_epoch=steps,
                       learning_rate=0.005, train_seconds=60)
    core.gate.set_enabled(True)
    return Corpus(["abcde " * 60, "0123456789 " * 40], ["abcde " * 20, "0123456789 " * 20])


def test_tokenizer_and_causal_mask():
    tokenizer = ByteTokenizer()
    text = "Ação: café! 😺\nPython"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    torch.manual_seed(1)
    model = TACTransformer(ModelConfig(width=32, heads=2, layers=1, context=16)).eval()
    a = torch.tensor([[1, 55, 66, 77]])
    b = torch.tensor([[1, 55, 99, 88]])
    with torch.no_grad():
        first = model(a)[0][:, :2]
        second = model(b)[0][:, :2]
    assert torch.equal(first, second)


@pytest.mark.neural
def test_actual_learning_checkpoint_and_generation(core):
    corpus = setup_training(core, 30)
    result = core.perform("Treino real CPU", lambda: core.trainer.train(corpus=corpus), learning=True)
    assert result["initial_hash"] != result["final_hash"]
    assert result["validation"] < result["baseline"] * 0.9
    assert result["steps"] == 30
    assert result["status"] == "aprovado"
    loaded = safe_load(core.trainer.checkpoint_path(result["checkpoint"]))
    model = TACTransformer(ModelConfig(**loaded["model_config"]))
    model.load_state_dict(loaded["model"])
    assert weight_hash(model) == result["final_hash"]
    from tac.neural.inventory import active_models
    # Conta o checkpoint ativo mesmo que a configuração da próxima rede mude.
    core.config.update(width=64)
    assert active_models(core.config, core.db)[0]["parameters"] == result["parameters"]
    core.gate.set_enabled(False)
    before = weight_hash(model)
    answer = core.trainer.generate("abcde")
    assert answer["mode"] == "neural experimental"
    assert "Não é uma resposta factual" in answer["text"]
    assert before == weight_hash(model)
    with pytest.raises(LearningDisabled):
        core.trainer.activate(result["checkpoint"])


@pytest.mark.neural
def test_interrupt_resume_exact_cpu_weights(core, tmp_path):
    corpus = setup_training(core, 15)
    def interrupt(step, message):
        if step == 5:
            core.gate.set_enabled(False)
    interrupted = core.perform("Interromper", lambda: core.trainer.train(interrupt, corpus=corpus), learning=True)
    assert interrupted["status"] == "interrompido"
    assert interrupted["steps"] == 5
    cp = safe_load(core.trainer.checkpoint_path(interrupted["checkpoint"]))
    assert cp["step"] == 5
    core.gate.set_enabled(True)
    resumed = core.perform("Retomar", lambda: core.trainer.train(resume=interrupted["checkpoint"]), learning=True)
    assert resumed["steps"] == 15
    from tac.core import Core
    uninterrupted_core = Core(tmp_path / "uninterrupted")
    uninterrupted_core.config.update(reserve_ram_mb=256, ram_mb=4096, cpu_threads=2)
    other_corpus = setup_training(uninterrupted_core, 15)
    complete = uninterrupted_core.perform("Completo", lambda: uninterrupted_core.trainer.train(corpus=other_corpus), learning=True)
    assert resumed["final_hash"] == complete["final_hash"]


@pytest.mark.neural
def test_rollback_does_not_promote_regression(core, monkeypatch):
    corpus = setup_training(core, 2)
    real_evaluate = core.trainer.evaluate
    calls = {"validation": 0}
    def evaluate(model, data, split, device):
        if split == "validation":
            calls[split] += 1
            if calls[split] >= 3:
                return 50.0, 0.0  # Falha controlada da avaliação para testar a política de rejeição.
        return real_evaluate(model, data, split, device)
    monkeypatch.setattr(core.trainer, "evaluate", evaluate)
    result = core.perform("Regressão controlada", lambda: core.trainer.train(corpus=corpus), learning=True)
    assert result["status"] == "rejeitado"
    assert core.trainer.active_id() is None


@pytest.mark.neural
def test_resource_error_saves_checkpoint(core, monkeypatch):
    corpus = setup_training(core, 10)
    calls = [0]
    def guard(config):
        calls[0] += 1
        if calls[0] >= 4:
            raise MemoryError("limite de teste")
    monkeypatch.setattr("tac.neural.training.check_resources", guard)
    with pytest.raises(RuntimeError, match="limite de teste"):
        core.perform("Memória", lambda: core.trainer.train(corpus=corpus), learning=True)
    rows = core.db.query("SELECT * FROM checkpoints WHERE status='erro'")
    assert rows
    assert safe_load(core.trainer.checkpoint_path(rows[0]["id"]))["step"] > 0


@pytest.mark.neural
def test_reinforcement_actual_rewards_and_updates(core):
    from tac.neural.reinforcement import run_experiment
    setup_training(core)
    result = core.perform("Reforço real", lambda: run_experiment(core.config, core.db, core.gate, steps=200), learning=True)
    assert result["initial_hash"] != result["final_hash"]
    assert result["accuracy"] > result["baseline"]
    assert result["accuracy"] >= 0.75
    attempts = core.db.query("SELECT a,b,action,reward FROM rl_attempts")
    assert len(attempts) == 200 * 16
    assert all(row["reward"] == float(row["a"] + row["b"] == row["action"]) for row in attempts)


@pytest.mark.neural
def test_cuda_training_when_available(core):
    if not torch.cuda.is_available():
        pytest.skip("CUDA indisponível: caminho CPU coberto nos outros testes")
    corpus = setup_training(core, 3)
    core.config.update(device="cuda")
    result = core.perform("Treino CUDA", lambda: core.trainer.train(corpus=corpus), learning=True)
    assert result["device"] == "cuda"
    assert result["initial_hash"] != result["final_hash"]


def test_removed_source_invalidates_derived_dataset(core):
    core.gate.set_enabled(True)
    source_id, _ = core.memory.ingest("fonte", "local", "file", "Este documento possui conteúdo suficiente para separar treino e validação sem sobrepor os intervalos de texto.")
    corpus = Corpus.from_memory(core.db, core.config)
    path = core.config.data / "datasets" / f"{corpus.digest}.json"
    corpus.save(path)
    assert path.exists()
    core.delete_source(source_id)
    assert not path.exists()
    assert not core.memory.search("conteúdo")
