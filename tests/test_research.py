import json
import pytest

from tac.core import Core
from tac.learning import Cancelled, LearningDisabled
from tac.neural.data import Corpus
from tac.research import compare_pages, page_answer, public_topic, title_urls
from tac.web import PublicFetcher


def fixture_fetch(monkeypatch, core, references=(), external=None):
    """Páginas fictícias explicitamente isoladas para testar a integração."""
    calls = []
    def fetch(client, url, scope=None):
        calls.append(url)
        if url.startswith('https://pt.wikipedia.org/wiki/'):
            client.reference_links = list(references)
            return url, 'Lunária – Wikipédia', 'Lunária é um país fictício criado neste teste. A capital de Lunária é Aurora. ' * 4, []
        client.reference_links = []
        return url, 'Portal de Lunária', external or ('A capital de Lunária é Aurora. Fonte editorial própria e contexto distinto. ' * 5), []
    monkeypatch.setattr(PublicFetcher, 'fetch', fetch)
    core.gate.set_enabled(True)
    return calls


def test_public_topic_does_not_export_private_context():
    assert public_topic('Qual é a capital do Chile?') == 'Chile'
    assert public_topic('O que é fotossíntese?') == 'fotossíntese'
    assert public_topic('oq e fotossintese?') == 'fotossintese'
    assert public_topic('qual e o nome do brasil?') == 'brasil'
    assert public_topic('Python (linguagem de programação)', manual=True)
    for question in ('eae', 'qual é meu nome?', 'Explique meu documento', 'Quem é fulano@email.com?',
                     'O que é 123.456.789-09?', 'O que é isso?', 'O que é https://site.test?'):
        assert public_topic(question) is None
    assert all(url.startswith('https://pt.wikipedia.org/wiki/') for url in title_urls('Chile'))


def test_fresh_install_has_no_automatic_network(core):
    assert not core.config.get('research_auto')


def test_automatic_research_prefers_local_memory_and_preserves_history(core, monkeypatch):
    calls = fixture_fetch(monkeypatch, core)
    core.config.update(research_auto=True)
    core.memory.ingest('Lunária', 'local', 'file', 'A capital de Lunária é Aurora. ' * 3)
    core.neural_history = [['user', 'meu nome é Ana'], ['assistant', 'Prazer, Ana.']]
    # Pesquisa automática pertence ao modo documental; conversa neural pura
    # não deve consultar a biblioteca nem receber respostas de referência.
    answer = core.chat('Qual é a capital de Lunária?')
    assert 'Aurora' in answer['display_text'] and answer['sources'] and not calls
    assert core.neural_history[0][1] == 'meu nome é Ana'


def test_automatic_lookup_saves_real_fixture_text_and_survives_restart(core, monkeypatch):
    calls = fixture_fetch(monkeypatch, core)
    core.config.update(research_auto=True, research_pages=1)
    result = core.perform('Pesquisa de teste', lambda: core.chat('Qual é a capital de Lunária?'), learning=True)
    assert len(calls) == 1 and result['mode'] == 'consulta web documental'
    assert result['research']['status'] == 'fonte_unica' and 'Aurora' in result['display_text']
    assert all(r['id'] > 0 and r['source_id'] > 0 for r in result['sources'])
    again = Core(core.config.root)
    again.gate.set_enabled(False)
    persisted = again.chat('Qual é a capital de Lunária?')
    assert 'Aurora' in persisted['display_text'] and persisted['web_verification']
    assert len(calls) == 1


def test_automatic_query_does_not_run_when_learning_is_off_or_private(core, monkeypatch):
    calls = fixture_fetch(monkeypatch, core)
    core.config.update(research_auto=True)
    core.chat('Qual é meu nome?')
    core.gate.set_enabled(False)
    core.chat('Qual é a capital de Lunária?')
    assert not calls and not core.memory.sources()


def test_switch_off_after_download_prevents_knowledge_commits(core, monkeypatch):
    def fetch(client, url, scope=None):
        core.gate.set_enabled(False)
        return url, 'Lunária', 'A capital de Lunária é Aurora. ' * 4, []
    monkeypatch.setattr(PublicFetcher, 'fetch', fetch)
    core.gate.set_enabled(True)
    with pytest.raises((Cancelled, LearningDisabled)):
        core.perform('Desligar durante consulta', lambda: core.chat('Qual é a capital de Lunária?', research=True), learning=True)
    assert not core.memory.sources()


def test_search_network_errors_are_reported_without_storing_fake_answers(core, monkeypatch):
    core.gate.set_enabled(True)
    monkeypatch.setattr(PublicFetcher, 'fetch', lambda *a: (_ for _ in ()).throw(OSError('rede indisponível de teste')))
    result = core.perform('Falha controlada', lambda: core.chat('O que é Lunária?', research=True), learning=True)
    assert not result['sources'] and result['research']['status'] == 'sem_resposta'
    assert result['research']['errors'] and not core.memory.sources()
    assert core.chat('oi')['text']


def test_cross_site_same_value_is_limited_agreement_not_proof(core, monkeypatch):
    fixture_fetch(monkeypatch, core, ['https://portal.example.org/pais'])
    result = core.perform('Comparar fontes', lambda: core.chat('Qual é a capital de Lunária?', research=True), learning=True)
    assert result['research']['status'] == 'concordancia_textual'
    assert 'veracidade não verificadas' in result['display_text']
    assert len(result['sources']) == 2


def test_conflicting_values_are_not_resolved_by_majority(core, monkeypatch):
    fixture_fetch(monkeypatch, core, ['https://portal.example.org/pais'], 'A capital de Lunária é Boreal. ' * 5)
    result = core.perform('Comparar divergência', lambda: core.chat('Qual é a capital de Lunária?', research=True), learning=True)
    assert result['research']['status'] == 'valores_divergentes'
    assert 'Não vou escolher' in result['display_text']
    assert 'Aurora' in result['display_text'] and 'Boreal' in result['display_text']


def test_same_site_and_exact_copies_are_not_extra_confirmations():
    def page(url, text):
        return {'origin':url, 'text':text, 'answer':{'values':['aurora']}}
    a = page('https://pt.wikipedia.org/wiki/A', 'texto original')
    assert compare_pages([a,page('https://en.wikipedia.org/wiki/B', 'outro texto')])[0] == 'fonte_unica'
    assert compare_pages([a,page('https://copy.example/abc', 'texto original')])[0] == 'fonte_unica'
    assert compare_pages([{**a,'answer':{'values':[]}},page('https://other.example/a','outro')])[0] == 'sem_comparacao_factual'


def test_research_pages_never_enter_neural_corpus_automatically(core, monkeypatch):
    fixture_fetch(monkeypatch, core)
    core.perform('Pesquisar', lambda: core.chat('O que é Lunária?', research=True), learning=True)
    with pytest.raises(ValueError, match='insuficientes'):
        Corpus.from_memory(core.db, core.config)
    local, _ = core.memory.ingest('Local', 'local', 'file', 'Somente o texto local autorizado deve entrar no conjunto neural. ' * 8)
    corpus = Corpus.from_memory(core.db, core.config)
    assert {p['source_id'] for p in corpus.provenance} == {local}


def test_refresh_of_researched_source_keeps_training_exclusion(core, monkeypatch):
    fixture_fetch(monkeypatch, core)
    core.perform('Pesquisar', lambda: core.chat('O que é Lunária?', research=True), learning=True)
    source = core.memory.sources()[0]
    core.memory.ingest(source['title'], source['origin'], 'web', 'Lunária é um país fictício com informações atualizadas. ' * 4)
    meta = json.loads(core.memory.sources()[0]['metadata'])
    assert meta['research_web'] and meta['verification'] == 'atualizada_sem_revisao'
    with pytest.raises(ValueError, match='insuficientes'):
        Corpus.from_memory(core.db, core.config)


def test_definition_prefers_explicit_relation_over_short_lateral_mention():
    page = {'title':'Lumina', 'origin':'https://public.example/Lumina',
            'text':'Lumina especial Não libera luz\nLumina é um processo fictício para converter energia e produzir luz.\nA água não participa de Lumina.'}
    answer = page_answer(page, 'O que é Lumina?')
    assert answer['evidence'][0].startswith('Lumina é um processo')
    assert page_answer(page, 'oq e Lumina?')['evidence'] == answer['evidence']


def test_html_reference_extraction_ignores_unrelated_links_and_scripts():
    from tac.documents import html_references
    links = html_references('<a href="https://ads.example">ads</a><script><a href="http://bad.example">x</a></script>'
                            '<li id="cite_note-1"><cite><a href="https://source.example/p">fonte</a></cite></li>')
    assert links == ['https://source.example/p']


def test_private_reference_is_blocked_by_actual_fetcher(core, monkeypatch):
    import tac.web as web
    core.gate.set_enabled(True)
    original = PublicFetcher.fetch
    def fetch(client, url, scope=None):
        if url.startswith('https://pt.wikipedia.org/'):
            client.reference_links = ['http://127.0.0.1/private']
            return url, 'Lunária', 'A capital de Lunária é Aurora. ' * 4, []
        return original(client,url,scope)
    monkeypatch.setattr(PublicFetcher, 'fetch', fetch)
    result = core.perform('Referência privada', lambda: core.chat('Qual é a capital de Lunária?', research=True), learning=True)
    assert result['research']['status'] == 'fonte_unica'
    assert any('bloqueados' in e for e in result['research']['errors'])


def test_ui_query_is_async_cancellable_and_default_off(core, qtapp, monkeypatch):
    from tac.ui import Window
    calls = fixture_fetch(monkeypatch, core)
    window = Window(core)
    window.timer.stop()
    window.show()
    assert not window.research_auto.isChecked()
    window.research_auto.setChecked(True)
    window.input.setText('Qual é a capital de Lunária?')
    window.send_button.click()
    assert window.job is not None and not window.research_stop.isHidden()
    window.job.wait(5000)
    qtapp.processEvents()
    assert calls
    window.close()
