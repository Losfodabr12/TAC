"""Consulta pública por título na Wikipédia e comparação textual conservadora.

Sem buscador geral, IA externa ou confiança probabilística inventada. Artigos
e referências são obtidos pelo mesmo cliente HTTP limitado e protegido da TAC.
"""
from __future__ import annotations

import json
import re
import time
from urllib.parse import quote, urljoin, urlsplit

from .answers import (FILLER, PROPERTY_WORDS, answer_question, explicit_facts,
                      fold, intent_of, primary_title, question_content, tokens)
from .db import now
from .documents import split_sections, text_hash
from .learning import Cancelled, LearningDisabled
from .memory import words
from .web import PublicFetcher

WIKI = 'https://pt.wikipedia.org/wiki/'


def public_topic(question, manual=False):
    """Só o assunto curto vai à rede; nunca histórico, textos ou arquivos."""
    value = question_content(question.strip())
    plain = fold(value)
    if (len(value) > 180 or re.search(r'https?://|@|[\\/]|\d[\d .()-]{6,}\d', value)
            or re.search(r'\b(meu|minha|meus|minhas|nosso|nossa|senha|cpf|email|e-mail|telefone|documento|arquivo)\b', plain)):
        return None
    prefixes = [
        r'^(?:tac[,! ]+)?(?:o que (?:é|e|são|sao)|quem (?:é|e|foi|foram)|onde fica(?:m)?|como funciona(?:m)?|explique(?: sobre)?|fale sobre|pesquise sobre)\s+',
        r'^(?:tac[,! ]+)?qual (?:é |e )?(?:(?:a|o) )?(?:capital|nome(?: oficial)?) (?:de|do|da)\s+',
    ]
    matched = False
    for pattern in prefixes:
        updated, n = re.subn(pattern, '', value, flags=re.I)
        if n:
            value, matched = updated, True
            break
    if not manual and not matched:
        return None
    value = value.strip(' ?!.:;"\'')
    if not value or len(value) > 100 or len(value.split()) > 10 or not re.fullmatch(r'[\wÀ-ÿ ()\-\'’,]+', value):
        return None
    if fold(value) in ('isso', 'isto', 'aquilo', 'ele', 'ela', 'voce', 'tac'):
        return None
    return value


def title_urls(topic):
    candidates = [topic[0].upper() + topic[1:], topic.title()]
    return list(dict.fromkeys(WIKI + quote(v.replace(' ', '_'), safe='()_-') for v in candidates))


def site_group(url):
    host = (urlsplit(url).hostname or '').lower().rstrip('.')
    parts = host.split('.')
    suffix = '.'.join(parts[-2:])
    return '.'.join(parts[-3:] if suffix in ('com.br', 'org.br', 'edu.br', 'gov.br', 'co.uk', 'org.uk') else parts[-2:])


class PageMemory:
    """Adaptador em RAM; IDs temporários nunca são apresentados ao usuário."""
    def __init__(self, page):
        self.source = {'id': -1, 'title': page['title'], 'origin': page['origin']}
        self.rows = []
        offset = 0
        for index, (section, text) in enumerate(split_sections(page['text'])):
            self.rows.append({**self.source, 'id': -index-1, 'source_id': -1,
                              'text': text, 'section': section, 'char_start': offset})
            offset += len(text) + 1

    def sources(self):
        return [self.source]

    def chunks(self, source_id):
        return self.rows

    def search(self, query, limit=24, source_id=None):
        wanted = set(words(query))
        return sorted((r for r in self.rows if wanted & set(words(r['text']))),
                      key=lambda r: -len(wanted & set(words(r['text']))))[:limit]


def page_answer(page, question):
    memory = PageMemory(page)
    answer = answer_question(memory, question)
    if not answer.get('sources'):
        return None
    kind = intent_of(question)
    values = set()
    if kind:
        topic = tokens(question) - FILLER - PROPERTY_WORDS[kind] - tokens('é sao foi ser esta sabe')
        for row in memory.rows:
            match = bool(topic and topic <= (tokens(primary_title(row)) - tokens('wikipedia enciclopedia livre')))
            for fact in explicit_facts(row, kind, topic, match):
                values.add(fold(re.sub(r'\[[^\]]*\]', '', fact.value)).strip(' .'))
    return {**answer, 'values': sorted(values)}


def compare_pages(pages):
    """Concordância de campos explícitos; sem inferência semântica geral."""
    unique, hashes, groups = [], set(), set()
    for page in pages:
        digest = text_hash(page['text'])
        group = site_group(page['origin'])
        if digest in hashes or group in groups:
            continue
        hashes.add(digest)
        groups.add(group)
        unique.append(page)
    values = [set(p['answer']['values']) for p in unique if p['answer']['values']]
    if len(unique) < 2:
        return 'fonte_unica', 'Fonte única; sem verificação em outro site.'
    if len(values) >= 2:
        if all(len(v) == 1 for v in values) and len(set.union(*values)) == 1:
            return 'concordancia_textual', 'Mesmo valor explícito em sites distintos; independência editorial e veracidade não verificadas.'
        return 'valores_divergentes', 'Valores textuais diferentes nas fontes; podem existir diferenças de nome, data ou contexto. Precisa de revisão.'
    return 'sem_comparacao_factual', 'Há trechos de sites distintos, mas a TAC não conseguiu verificar se fazem a mesma afirmação.'


class Researcher:
    def __init__(self, core):
        self.core = core

    def lookup(self, question, progress=lambda *_: None, topic=None):
        core = self.core
        core.gate.check()
        topic = public_topic(topic or question, manual=topic is not None)
        if not topic:
            return self.unavailable('Informe um assunto público curto. Histórico, documentos e dados pessoais não são enviados.')
        started = time.monotonic()
        errors, pages = [], []
        def network_progress(n, message):
            core.gate.check()
            if time.monotonic() - started >= core.config.get('research_seconds'):
                raise TimeoutError('Limite de duração da consulta atingido.')
            progress(n, message)
        fetcher = PublicFetcher(core.gate, core.config, network_progress)
        attempts = 0

        def get(url, article=False):
            nonlocal attempts
            core.gate.check()
            if attempts >= core.config.get('research_pages') + 2 or time.monotonic()-started >= core.config.get('research_seconds'):
                return None
            attempts += 1
            try:
                progress(len(pages), 'Consultando artigo na Wikipédia…' if article else 'Consultando uma referência pública do artigo…')
                origin, title, text, _ = fetcher.fetch(url, WIKI if article else None)
                cap = core.config.get('research_chars')
                page = {'origin': origin, 'title': title, 'text': text[:cap], 'partial': len(text) > cap,
                        'references': list(getattr(fetcher, 'reference_links', [])) if article else []}
                if article and re.search(r'pode (?:referir-se|se referir) a|página de desambiguação', text[:2000], re.I):
                    errors.append('Artigo ambíguo: informe um assunto mais específico.')
                    return None
                page['answer'] = page_answer(page, question)
                if page['answer']:
                    return page
                errors.append('A página não contém uma resposta explícita recuperável para a pergunta.')
            except (Cancelled, LearningDisabled):
                raise
            except Exception as exc:
                errors.append(f'{url}: {exc}')
                core.logger.info('Consulta web sem resultado: %s: %s', url, exc)
                progress(len(pages), f'Não foi possível consultar esta página: {exc}')
            return None

        for url in title_urls(topic):
            page = get(url, article=True)
            if page:
                pages.append(page)
                break
        if pages:
            urls = []
            for href in pages[0]['references']:
                try:
                    u = urljoin(pages[0]['origin'], href).split('#')[0]
                    parsed = urlsplit(u)
                    if (parsed.scheme in ('http', 'https') and parsed.hostname and not parsed.username
                            and not parsed.password and site_group(u) not in ('wikipedia.org', 'wikimedia.org')):
                        urls.append(u)
                except ValueError:
                    continue
            urls = list(dict.fromkeys(urls))
            # Não segue links arbitrários do corpo nem solicita páginas ilimitadas.
            seen = {site_group(pages[0]['origin'])}
            for url in urls:
                if len(pages) >= core.config.get('research_pages') or attempts >= core.config.get('research_pages') + 2:
                    break
                group = site_group(url)
                if group in seen:
                    continue
                seen.add(group)
                page = get(url)
                if page:
                    pages.append(page)
        if not pages:
            result = self.unavailable('Não encontrei um artigo com resposta fundamentada. Tente o título exato do assunto ou adicione uma URL em Ensinar.')
            result['research'] = {'topic': topic, 'status': 'sem_resposta', 'errors': errors, 'checked_at': now(), 'attempts': attempts}
            core.db.event('pesquisa_web', json.dumps(result['research'], ensure_ascii=False))
            return result

        status, note = compare_pages(pages)
        checked = now()
        evidence, refs, ids = [], [], []
        progress(len(pages), 'Salvando textos e referências na biblioteca; sem alterar pesos neurais…')
        # Desligar o aprendizado bloqueia o lote inteiro antes do primeiro commit.
        with core.gate.writing():
            core.gate.check()
            for page in pages:
                core.gate.check()
                # Consulta automática não substitui uma fonte já administrada pelo usuário.
                existing = core.db.query('SELECT id FROM sources WHERE origin=?', (page['origin'],))
                if existing:
                    source_id = existing[0]['id']
                else:
                    metadata = {'research_web': True, 'verification': status, 'verification_note': note,
                                'checked_at': checked, 'partial': page['partial'], 'training_approved': False,
                                'verification_topic': topic, 'verification_field': intent_of(question) or 'trecho textual',
                                'retrieval': 'artigo por título e referências; sem buscador geral'}
                    source_id, _ = core.memory.ingest(page['title'], page['origin'], 'web', page['text'],
                                                     'Consulta web / revisão pendente', metadata)
                # Responde com o conteúdo efetivamente armazenado, inclusive em duplicatas.
                stored = core.memory.answer(question, source_id)
                if not stored.get('sources'):
                    continue
                if source_id in ids:
                    continue
                actual_source = core.db.query('SELECT title,origin FROM sources WHERE id=?', (source_id,))[0]
                page.update(actual_source)
                page['text'] = '\n'.join(r['text'] for r in core.memory.chunks(source_id))
                page['answer'] = page_answer(page, question)
                ids.append(source_id)
                refs.extend(stored['sources'])
                evidence.extend(stored.get('evidence', []))
                if len(ids) == 1:
                    primary = stored
            represented = [p for p in pages if p.get('answer') and p['origin'] in {r['origin'] for r in refs}]
            status, note = compare_pages(represented)
            for source_id in ids:
                row = core.db.query('SELECT metadata FROM sources WHERE id=?', (source_id,))[0]
                meta = json.loads(row['metadata'])
                if meta.get('research_web'):
                    meta.update(verification=status, verification_note=note)
                    core.db.execute('UPDATE sources SET metadata=? WHERE id=?', (json.dumps(meta, ensure_ascii=False), source_id))
        if not ids:
            return self.unavailable('A consulta não produziu evidência recuperável na biblioteca.')
        research = {'topic': topic, 'status': status, 'note': note, 'checked_at': checked,
                    'source_ids': list(dict.fromkeys(ids)), 'errors': errors, 'attempts': attempts,
                    'provider': 'Wikipédia: consulta por título', 'neural_training': False}
        core.db.event('pesquisa_web', json.dumps(research, ensure_ascii=False))
        core.study_history(None, pages[0]['origin'], 'pesquisa_web', json.dumps(research, ensure_ascii=False))
        visible = primary.get('display_text', primary['text'])
        if status == 'valores_divergentes':
            visible = 'As fontes apresentam valores diferentes. Não vou escolher um como correto.\n\n' + '\n\n'.join(evidence[:3])
        return {**primary, 'mode': 'consulta web documental', 'display_text': visible + '\n\n' + note,
                'text': visible + '\n\n' + note, 'sources': refs, 'evidence': evidence,
                'method': 'extração documental local; comparação limitada de campos explícitos', 'research': research}

    @staticmethod
    def unavailable(message):
        return {'mode': 'consulta web documental', 'text': message, 'sources': []}


def qualify_saved_answer(core, answer):
    ids = list(dict.fromkeys(r['source_id'] for r in answer.get('sources', [])))
    researched = []
    for source_id in ids:
        rows = core.db.query('SELECT metadata FROM sources WHERE id=?', (source_id,))
        if rows:
            meta = json.loads(rows[0]['metadata'])
            if meta.get('research_web'):
                researched.append({'source_id': source_id, **meta})
    if researched and not answer.get('research'):
        answer = dict(answer)
        note = 'Fonte de consulta web; verificação limitada. Confira a data e as evidências em Ver fontes.'
        answer['display_text'] = answer.get('display_text', answer['text']) + '\n\n' + note
        answer['web_verification'] = researched
    return answer
