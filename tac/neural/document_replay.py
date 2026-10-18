"""Integração opcional de escrita, preparada somente com fontes escolhidas."""
from __future__ import annotations

import json
from pathlib import Path

from ..config import atomic_json
from .data import Corpus


def prepare_document_replay(core,source_ids,max_chars=20000,max_fraction=.1):
    if not source_ids or not 1000<=max_chars<=100000 or not 0<max_fraction<=.1:
        raise ValueError('Escolha fontes e limite de até 10% de trechos adicionais.')
    core.gate.check()
    parent=core.conversation.active_id()
    if not parent:raise ValueError('Prepare primeiro um modelo de conversa.')
    # Não reintroduz livros antigos que não foram selecionados nesta operação.
    from ..fluency_lessons import make_language_corpus
    base=make_language_corpus()
    class Limits:
        def get(self,key):return max_chars if key=='max_corpus_chars' else core.config.get(key)
    document=Corpus.from_memory(core.db,Limits(),source_ids=source_ids)
    # Novos trechos não são repetidos artificialmente para atingir uma quota.
    limit=int(len(base.train_text)*max_fraction/(1-max_fraction))
    selected=document.train_text[:limit]
    if not selected:raise ValueError('Não há texto elegível suficiente nas fontes escolhidas.')
    combined=Corpus([*base.train_text,*selected],[*base.val_text,*document.val_text],
                    [*base.provenance,*document.provenance[:len(selected)]])
    result={'format':'TAC-document-replay-v1','dataset_hash':combined.digest,'parent':parent,
        'source_ids':list(source_ids),'base_samples':len(base.train_text),'document_samples':len(selected),
        'document_sample_fraction':len(selected)/len(combined.train_text),'max_chars':max_chars,
        'scope':'Previsão de texto em lotes de linguagem; não cria exemplos de diálogo nem substitui a Biblioteca/RAG.'}
    with core.gate.writing():
        core.gate.check();combined.save(core.config.data/'datasets'/(combined.digest+'.json'))
        atomic_json(core.config.data/'datasets'/('document_replay_'+combined.digest+'.manifest.json'),result)
    return result
