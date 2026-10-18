import pytest
import torch

from tac.conversation_lessons import canonical, make_sets, prompt_text
from tac.learning import Cancelled, LearningDisabled
from tac.neural.conversation import ConversationData, ConversationEngine
from tac.neural.model import ModelConfig, TACTransformer
from tac.neural.training import safe_load
from test_training import setup_training


def small_sets():
    sets = {split: [{"messages": [["user", text]], "answer": "Oi.", "skill": "greeting"}]
            for split, text in (("train", "oi"), ("validation", "ei"), ("test", "ai"))}
    sets["train"] += [{"messages": [["user", text]], "answer": "Oi.", "skill": "greeting"} for text in ("a b", "olha", "oi ai")]
    return sets


def test_response_targets_and_disjoint_authorial_splits(tmp_path):
    sets = make_sets()
    keys = [{canonical(prompt_text(row["messages"])) for row in sets[split]} for split in ("train", "validation", "test")]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
    data = ConversationData(sets)
    path = tmp_path / "conversation.json"
    data.save(path)
    assert ConversationData.load(path).digest == data.digest
    for row in sets["train"]:
        x, y = data.encode(row, 256)
        assert len(x) <= 256 and (y == 0).any() and (y != 0).any()
        supervised = data.tokenizer.decode(y[y != 0].tolist())
        assert supervised == " " + row["answer"]
        assert y[-1] == data.tokenizer.eos


@pytest.mark.neural
def test_response_training_resume_preserves_reader_and_exact_cpu(core, tmp_path, monkeypatch):
    from tac.core import Core
    monkeypatch.setattr("tac.neural.conversation.make_sets", small_sets)
    data = setup_training(core, 5)
    core.perform("Leitor", lambda: core.trainer.train(corpus=data), learning=True)
    reader = core.trainer.active_id()
    def stop(step, *_):
        if step == 100:
            core.gate.set_enabled(False)
    stopped = core.perform("Diálogo", lambda: core.conversation.train(stop, steps=120, learning_rate=.003), learning=True)
    assert stopped["status"] == "interrompido" and stopped["steps"] == 100
    assert core.trainer.active_id() == reader
    core.gate.set_enabled(True)
    resumed = core.perform("Retomar", lambda: core.conversation.train(resume=stopped["checkpoint"]), learning=True)
    other = Core(tmp_path / "uninterrupted_sft")
    other.config.update(reserve_ram_mb=256, ram_mb=4096, cpu_threads=2)
    base = setup_training(other, 5)
    other.perform("Leitor", lambda: other.trainer.train(corpus=base), learning=True)
    full = other.perform("Diálogo", lambda: other.conversation.train(steps=120, learning_rate=.003), learning=True)
    assert resumed["final_hash"] == full["final_hash"]
    assert resumed["status"] == "aprovado" and resumed["validation"]["exact_accuracy"] == 1.
    assert resumed["initial_hash"] != resumed["final_hash"]
    assert core.trainer.active_id() == reader
    core.gate.set_enabled(False)
    monkeypatch.setattr("tac.neural.conversation.make_sets", lambda: pytest.fail("Inferência não pode consultar respostas dos exercícios"))
    monkeypatch.setattr(core.dialogue, "reply", lambda *_: pytest.fail("Diálogo gerativo não usa o classificador"))
    answer = core.chat("oi", conversation=True)
    assert answer["mode"] == "diálogo neural experimental" and answer["text"] == "Oi."
    assert not answer["sources"]
    assert core.neural_history[-1] == ["assistant", "Oi."]
    with pytest.raises((Cancelled, LearningDisabled)):
        core.conversation.train(steps=1)


def test_switching_off_during_final_evaluation_prevents_promotion(core, monkeypatch):
    monkeypatch.setattr("tac.neural.conversation.make_sets", small_sets)
    base = setup_training(core, 5)
    core.perform("Leitor", lambda: core.trainer.train(corpus=base), learning=True)
    parent = core.trainer.active_id()
    real_evaluate = core.conversation.evaluate
    def evaluate(*args, **kwargs):
        if kwargs.get("generation") and kwargs.get("interruptible"):
            core.gate.set_enabled(False)
        return real_evaluate(*args, **kwargs)
    monkeypatch.setattr(core.conversation, "evaluate", evaluate)
    result = core.perform("Parar avaliação", lambda: core.conversation.train(steps=2), learning=True)
    assert result["status"] == "interrompido"
    assert core.conversation.active_id() is None
    assert core.trainer.active_id() == parent


def test_greedy_decoder_has_no_sampling_dependency():
    model = TACTransformer(ModelConfig(width=32, heads=2, layers=1, context=32))
    prompt = torch.tensor([[1, 20, 30]])
    assert torch.equal(model.generate(prompt, 4, greedy=True, seed=1), model.generate(prompt, 4, greedy=True, seed=7))


def test_own_bpe_roundtrip_snapshot_and_parameter_budget(core):
    from tac.neural.bpe import LocalBPETokenizer
    from tac.neural.conversation import initialize_bpe_model
    tokenizer = LocalBPETokenizer.fit(["Olá! Vamos conversar sobre histórias? 😊" * 15, "A TAC aprende com exercícios locais." * 15], vocab_size=330)
    text = "Ação! Olá, João. 🐯\nUm texto novo: 日本語."
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert LocalBPETokenizer.from_state(tokenizer.state()).encode(text) == tokenizer.encode(text)
    base = TACTransformer(ModelConfig(context=256, ff_width=896))
    # O orçamento usa parâmetros únicos, inclusive com embeddings compartilhados.
    full = LocalBPETokenizer([(3, 4)] * 765)
    model = initialize_bpe_model({"model_config": vars(base.cfg), "model": base.state_dict()}, full)
    assert model.parameter_count == 494976
    assert model.tokens.weight is model.output.weight


def test_ten_million_architecture_and_context_curriculum():
    from tac.context_lessons import make_context_sets
    from tac.neural.conversation import conversation_10m_config
    from tac.neural.model import count_parameters
    cfg = conversation_10m_config()
    model = TACTransformer(cfg)
    assert count_parameters(cfg) == model.parameter_count == 10_004_544
    assert cfg.context == 512 and cfg.heads == 6 and cfg.layers == 5
    sets = make_context_sets()
    keys = [{canonical(prompt_text(r["messages"])) for r in sets[s]} for s in ("train", "validation", "test")]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
    assert sets == make_context_sets()
    assert {r["skill"] for r in sets["test"]} == {"context_color", "context_update", "context_fact", "context_location", "logic_condition", "arithmetic"}
    for split in sets:
        for row in sets[split]:
            if row["skill"] == "arithmetic":
                a, op, b, _, answer = row["answer"].rstrip('.').split()
                assert int(answer) == (int(a) + int(b) if op == '+' else int(a) - int(b))


def test_text_replay_respects_reserved_validation_and_integrity(core):
    from tac.neural.data import Corpus
    from tac.neural.conversation_corpus import TextReplay
    from tac.neural.bpe import LocalBPETokenizer
    corpus = Corpus(["abcde " * 60], ["RESERVED" * 10])
    path = core.config.data / "datasets" / (corpus.digest + ".json")
    corpus.save(path)
    tokenizer = LocalBPETokenizer()
    replay = TextReplay(path, tokenizer)
    assert all("RESERVED" not in tokenizer.decode(row.tolist()) for row in replay.rows)
    x, y = replay.batch(2, 32, torch.Generator().manual_seed(1), "cpu")
    assert x.shape == y.shape == (2, 32) and replay.tokens == 64
    import json
    content = json.loads(path.read_text(encoding="utf-8"))
    content["train"] = ["alterado"]
    path.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="alterado"):
        TextReplay(path, tokenizer)


def test_rotary_attention_preserves_relative_positions_and_causality():
    model = TACTransformer(ModelConfig(width=32, heads=2, layers=2, context=128, rope=True)).eval()
    x = torch.tensor([[1, 60, 33, 100, 23]])
    with torch.no_grad():
        original = model(x)[0]
        shifted = model(x, position_offset=73)[0]
        torch.testing.assert_close(original, shifted, atol=1e-6, rtol=1e-5)
        changed = x.clone()
        changed[0, -1] = 18
        torch.testing.assert_close(model(changed)[0][:, :-1], original[:, :-1], atol=1e-6, rtol=1e-5)
    assert 'position.weight' not in model.state_dict()
    from tac.neural.model import count_parameters
    assert count_parameters(model.cfg) == model.parameter_count


def test_context_acceptance_checks_every_memory_task_and_keeps_math_separate():
    from tac.neural.conversation import context_ready
    groups = {s: {"correct": 8, "total": 10} for s in ("context_color", "context_update", "context_fact", "context_location", "logic_condition")}
    groups['arithmetic'] = {"correct": 0, "total": 100}
    assert context_ready({"by_skill": groups})
    groups['context_update']['correct'] = 0
    assert not context_ready({"by_skill": groups})


def test_rotary_upgrade_preserves_own_embeddings_and_existing_mlp_weights():
    from tac.neural.conversation import initialize_rotary_model
    from tac.neural.bpe import LocalBPETokenizer
    old = TACTransformer(ModelConfig(vocab_size=1024, context=512, width=384, heads=6,
                                     layers=5, ff_width=1680, tie_embeddings=True))
    tokenizer = LocalBPETokenizer([(3, 4)] * 765)
    upgraded = initialize_rotary_model({"model_config": vars(old.cfg), "model": old.state_dict()}, tokenizer)
    assert upgraded.parameter_count == 10_004_544 and upgraded.cfg.rope
    assert torch.equal(upgraded.tokens.weight, old.tokens.weight)
    assert torch.equal(upgraded.blocks[0].ff[2].weight[:, :1680], old.blocks[0].ff[2].weight)
    assert not upgraded.blocks[0].ff[2].weight[:, 1680:].count_nonzero()


def test_gui_interrupted_evaluation_is_pending_and_source_switches_mode(core, qtapp):
    from tac.ui import Window
    core.gate.set_enabled(True)
    core.memory.ingest("Documento", "local", "file", "Um documento para consultar informações e acompanhar a fonte de origem.")
    window = Window(core)
    window.timer.stop()
    window.conversation_result({"status": "interrompido", "steps": 7, "device": "cpu",
                                "baseline": {"loss": 2.}, "validation": {"loss": 1.}})
    assert "pendente" in window.train_label.text() and "0.0%" not in window.train_label.text()
    window.chat_mode.model().item(2).setEnabled(True)
    window.chat_mode.setCurrentIndex(2)
    window.sources_table.selectRow(0)
    window.use_source()
    assert window.chat_mode.currentIndex() == 0 and window.current_source is not None
    window.close()


def test_small_fragment_vocabulary_cannot_stall_curriculum_generation():
    from tac.context_lessons import make_context_sets
    sets = make_context_sets(fragments=("x",))
    assert len(sets['train']) > 1000
    heldout = {canonical(prompt_text(r['messages'])) for s in ('validation', 'test') for r in sets[s]}
    assert not any(canonical(prompt_text(r['messages'])) in heldout for r in sets['train'])
