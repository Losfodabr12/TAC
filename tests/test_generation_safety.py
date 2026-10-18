"""Geração/parada com logits de teste; nenhum treinamento ou checkpoint real."""
import pytest
import torch

from tac.conversation_lessons import prompt_text
from tac.neural.decoding import ConversationStop, conversation_options, decode_conversation
from tac.neural.model import ModelConfig, TACTransformer
from tac.neural.tokenizer import ByteTokenizer


class ScriptedModel(TACTransformer):
    """Fixture determinística para testar o mecanismo, sem respostas na produção."""
    def __init__(self, output, context=256):
        super().__init__(ModelConfig(context=context, width=16, heads=2, layers=1))
        self.text_tokenizer = ByteTokenizer()
        self.script = self.text_tokenizer.encode(output, special=False) + [2]
        self.calls = []

    def forward(self, tokens, targets=None, position_offset=0):
        step = len(self.calls)
        self.calls.append(tokens.clone())
        logits = torch.full((*tokens.shape, self.cfg.vocab_size), -100., device=tokens.device)
        logits[:, -1, self.script[min(step, len(self.script) - 1)]] = 100.
        return logits, None


def test_cycle_stops_before_budget_and_preserves_neural_prefix():
    m = ScriptedModel(' O bom gosto, bom gosto, bom gosto, e mais texto.')
    result = decode_conversation(m, [['user', 'Oi']], 'cpu', max_tokens=100)
    assert result['stop_reason'] == 'repetition'
    assert result['text'] == 'O bom gosto'
    assert result['raw_text'].strip().startswith(result['text'])
    assert result['generated_tokens'] < 100 and not result['eos_generated']
    assert result['loop'] == {'phrase': 'bom gosto', 'repetitions': 3, 'word_count': 2}


def test_prompt_repetitions_do_not_block_reply_or_change_serialization():
    messages = [['user', 'bom bom bom bom'], ['assistant', 'Qual assunto?'],
                ['user', 'Meu nome é Carlos. Qual é meu nome?']]
    m = ScriptedModel(' Carlos.')
    result = decode_conversation(m, messages, 'cpu')
    expected = [1, *m.text_tokenizer.encode(prompt_text(messages), special=False)]
    assert m.calls[0][0].tolist() == expected
    assert result['text'] == 'Carlos.' and result['stop_reason'] == 'eos'
    assert result['eos_generated'] and result['truncated_tokens'] == 0


def test_repeated_names_twice_and_partial_words_are_not_cycles():
    stop = ConversationStop(ByteTokenizer())
    assert stop.inspect('Carlos saiu; Carlos voltou. Ana saiu; Ana voltou.') is None
    assert stop.inspect('bom bom bom bo') is None
    assert stop.inspect('bom bom bom bom') is None  # último subtoken ainda aberto
    assert stop.inspect('bom bom bom bom,') == 'repetition'


def test_interleaved_phrase_saturation_stops_but_two_mentions_do_not():
    sentence = ('Um lobo de areia aparece. Um lobo de areia se move. '
                'Um lobo de areia fica. Um lobo de areia descansa.')
    stop = ConversationStop(ByteTokenizer())
    assert stop.inspect('Um lobo de areia aparece. Um lobo de areia se move.') is None
    m = ScriptedModel(' ' + sentence)
    result = decode_conversation(m, [['user', 'O que é um lobo?']], 'cpu', max_tokens=200)
    assert result['stop_reason'] == 'repetition'
    assert result['loop']['pattern'] == 'interleaved'
    assert result['text'] == 'Um lobo de areia' and result['generated_tokens'] < 200


def test_eos_and_role_boundary_are_distinct_stops():
    m = ScriptedModel(' Oi.\nUsuário: Outra pergunta?')
    result = decode_conversation(m, [['user', 'Oi']], 'cpu')
    assert result['text'] == 'Oi.' and result['stop_reason'] == 'role_boundary'
    assert not result['eos_generated']
    m = ScriptedModel(' Olá!')
    result = decode_conversation(m, [['user', 'Oi']], 'cpu')
    assert result['stop_reason'] == 'eos' and result['text'] == 'Olá!'


def test_budget_and_context_truncation_are_reported_without_fake_eos():
    m = ScriptedModel(' abcdefghijklmnopqrstuvwxyz', context=32)
    messages = [['user', 'Histórico longo. ' * 20]]
    result = decode_conversation(m, messages, 'cpu', max_tokens=8)
    assert len(m.calls[0][0]) == 32 and m.calls[0][0][0] == 1
    assert result['truncated_tokens'] > 0 and result['input_tokens'] == 32
    assert result['stop_reason'] == 'length' and result['generated_tokens'] == 8
    assert not result['eos_generated']
    assert result['text'] == 'abcdefg'
    with pytest.raises(ValueError):
        m.generate(torch.tensor([[1]]), max_new_tokens=-1)


def test_real_engine_uses_guard_and_keeps_metadata_out_of_history(core, monkeypatch):
    engine = core.conversation
    model = ScriptedModel(' Opa, bom, bom, bom, bom, texto que não deve continuar.')
    monkeypatch.setattr(engine, 'active_id', lambda: 'fixture')
    engine._cached_id, engine._cached_model, engine._cached_generation = 'fixture', model, 'TAC-v2'
    core.config.update(conversation_generation='greedy')
    answer = core.chat('Opa, tudo bom?', conversation=True)
    assert answer['decoding']['stop_reason'] == 'repetition'
    assert answer['text'] == 'Opa, bom' and answer['sources'] == []
    assert core.neural_history == [['user', 'Opa, tudo bom?'], ['assistant', 'Opa, bom']]
    assert conversation_options('greedy', 42)['greedy']
    assert conversation_options('controlled', 42)['temperature'] == .55


def test_ui_discloses_loop_without_inventing_answer(core, qtapp, monkeypatch):
    from tac.ui import Window
    w = Window(core)
    w.timer.stop()
    w.input.setText('Oi')
    def job(name, operation, success, **kw):
        success({'text': 'Opa, bom', 'mode': 'diálogo neural experimental', 'sources': [],
                 'decoding': {'stop_reason': 'repetition'}})
    monkeypatch.setattr(w, 'start_job', job)
    w.send_chat()
    transcript = w.transcript.toPlainText()
    assert 'Opa, bom' in transcript and 'interrompida por repetição' in transcript
    assert 'Resposta parcial' in transcript
    w.close()
