import pytest
import torch

from tac.learning import Cancelled, LearningDisabled
from tac.neural.model import ModelConfig, TACTransformer, expand_model
from tac.neural.training import safe_load, weight_hash
from test_training import setup_training


def test_expansion_preserves_old_predictions_and_stays_below_budget():
    torch.manual_seed(72)
    original = TACTransformer(ModelConfig()).eval()
    original_hash = weight_hash(original)
    expanded = expand_model(original, ModelConfig(context=256, ff_width=896)).eval()
    assert expanded.parameter_count == 495_808
    assert weight_hash(original) == original_hash
    tokens = torch.randint(3, 259, (2, 128))
    with torch.no_grad():
        before = original(tokens)[0]
        after = expanded(tokens)[0]
    torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)
    # A saída dos novos neurônios recebe gradientes reais já no primeiro passo.
    expanded.train()
    expanded(tokens[:, :-1], tokens[:, 1:])[1].backward()
    assert expanded.blocks[0].ff[2].weight.grad[:, 384:].abs().sum() > 0
    with pytest.raises(ValueError, match="limite"):
        expand_model(original, ModelConfig(context=256, ff_width=1024))
    with pytest.raises(ValueError, match="reduzir"):
        expand_model(original, ModelConfig(context=64))


@pytest.mark.neural
def test_expanded_checkpoint_resume_and_disabled_gate(core):
    corpus = setup_training(core, 8)
    core.perform("Base", lambda: core.trainer.train(corpus=corpus), learning=True)
    parent = core.trainer.active_id()
    cfg = ModelConfig(width=32, heads=2, layers=1, context=64, ff_width=192)
    core.gate.set_enabled(False)
    with pytest.raises((LearningDisabled, Cancelled)):
        core.trainer.train(corpus=corpus, model_config=cfg)
    assert core.trainer.active_id() == parent
    core.gate.set_enabled(True)
    core.config.update(steps_per_epoch=12)
    def stop(step, *_):
        if step == 5:
            core.gate.set_enabled(False)
    interrupted = core.perform("Ampliar", lambda: core.trainer.train(stop, corpus=corpus, model_config=cfg), learning=True)
    cp = safe_load(core.trainer.checkpoint_path(interrupted["checkpoint"]))
    assert cp["model_config"]["ff_width"] == 192
    assert cp["training_config"]["evaluation_context"] == 32
    assert cp["training_config"]["model_expansion"]["parent_checkpoint"] == parent
    assert interrupted["steps"] == 5 and interrupted["status"] == "interrompido"
    core.gate.set_enabled(True)
    resumed = core.perform("Retomar", lambda: core.trainer.train(resume=interrupted["checkpoint"]), learning=True)
    assert resumed["steps"] == 12 and resumed["initial_hash"] != resumed["final_hash"]
    final = safe_load(core.trainer.checkpoint_path(resumed["checkpoint"]))
    assert final["model_config"] == cp["model_config"]
    assert final["training_config"]["evaluation_context"] == 32


@pytest.mark.neural
def test_functional_retention_resume_is_exact_and_keeps_teacher(core, tmp_path):
    from tac.core import Core
    base = setup_training(core, 8)
    core.perform("Base", lambda: core.trainer.train(corpus=base), learning=True)
    teacher_id = core.trainer.active_id()
    teacher_hash = safe_load(core.trainer.checkpoint_path(teacher_id))["model"]["tokens.weight"].clone()
    core.config.update(steps_per_epoch=12, retention_weight=2)
    cfg = ModelConfig(width=32, heads=2, layers=1, context=64, ff_width=192)
    def stop(step, *_):
        if step == 5:
            core.gate.set_enabled(False)
    stopped = core.perform("Ampliação", lambda: core.trainer.train(stop, corpus=base, model_config=cfg), learning=True)
    core.gate.set_enabled(True)
    resumed = core.perform("Retomar", lambda: core.trainer.train(resume=stopped["checkpoint"]), learning=True)
    other = Core(tmp_path / "functional_uninterrupted")
    other.config.update(reserve_ram_mb=256, ram_mb=4096, cpu_threads=2)
    data = setup_training(other, 8)
    other.perform("Base", lambda: other.trainer.train(corpus=data), learning=True)
    other.config.update(steps_per_epoch=12, retention_weight=2)
    complete = other.perform("Ampliação", lambda: other.trainer.train(corpus=data, model_config=cfg), learning=True)
    assert resumed["final_hash"] == complete["final_hash"]
    assert resumed["teacher_checkpoint"] == teacher_id
    torch.testing.assert_close(safe_load(core.trainer.checkpoint_path(teacher_id))["model"]["tokens.weight"], teacher_hash)


@pytest.mark.neural
def test_cuda_overflow_discards_batch_without_ending_training(core, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA indisponível")
    data = setup_training(core, 3)
    core.config.update(device="cuda")
    real_init = TACTransformer.__init__
    injected = [False]
    def init_with_one_overflow(self, cfg):
        real_init(self, cfg)
        def inject(gradient):
            if not injected[0]:
                injected[0] = True
                return torch.full_like(gradient, float("inf"))
            return gradient
        self.tokens.weight.register_hook(inject)
    # Falha deliberada de teste; as métricas de produção não usam essa injeção.
    monkeypatch.setattr(TACTransformer, "__init__", init_with_one_overflow)
    result = core.perform("Recuperar overflow", lambda: core.trainer.train(corpus=data), learning=True)
    assert result["steps"] == 3 and result["precision_skips"] >= 1
    assert result["initial_hash"] != result["final_hash"]
    assert result["status"] == "aprovado"


def test_reference_prevents_accumulating_regression_across_rounds(core, monkeypatch):
    data = setup_training(core, 5)
    core.perform("Base", lambda: core.trainer.train(corpus=data), learning=True)
    original = core.trainer.active_id()
    cfg = ModelConfig(width=32, heads=2, layers=1, context=64, ff_width=192)
    expanded = core.perform("Ampliar", lambda: core.trainer.train(corpus=data, model_config=cfg), learning=True)
    assert expanded["status"] == "aprovado"
    current_id = core.trainer.active_id()
    calls = [0]
    def controlled(model, corpus, split, device, context=None):
        if split == "fixed":
            if model.cfg.ff_width == 0:
                return 1., .5  # Referência original (avaliação controlada de teste).
            calls[0] += 1
            return (1.015 if calls[0] == 1 else 1.035), .5
        return 10., .5
    monkeypatch.setattr(core.trainer, "evaluate", controlled)
    result = core.perform("Regressão acumulada", lambda: core.trainer.train(corpus=data), learning=True)
    assert result["fixed_baseline"] == 1.015
    assert result["fixed_reference_baseline"] == 1.
    assert result["fixed_reference_checkpoint"] == original
    assert result["status"] == "rejeitado" and core.trainer.active_id() == current_id
