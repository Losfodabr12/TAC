"""Pré-requisito de reconstrução. Somente leitura de originais; não treina/exclui."""
from __future__ import annotations
import json
import os
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/reconstrucao_zero/preparacao_20261017_01'
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8',errors='replace')


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    con=sqlite3.connect((ROOT/'data/tac.sqlite3').as_uri()+'?mode=ro&immutable=1',uri=True)
    con.row_factory=sqlite3.Row
    sources=[dict(r) for r in con.execute('SELECT s.id,s.title,s.kind,s.origin,s.metadata,COUNT(ch.id) chunks,SUM(LENGTH(ch.text)) chars FROM sources s LEFT JOIN chunks ch ON ch.source_id=s.id GROUP BY s.id')]
    kinds=[dict(r) for r in con.execute('SELECT r.kind,COUNT(cp.id) checkpoints FROM runs r LEFT JOIN checkpoints cp ON cp.run_id=r.id GROUP BY r.kind')]
    catalog=[dict(r) for r in con.execute('SELECT cp.*,r.kind,r.dataset_hash FROM checkpoints cp JOIN runs r ON r.id=cp.run_id')]
    runs=[dict(r) for r in con.execute('SELECT id,kind,dataset_hash FROM runs')]
    meta=[dict(r) for r in con.execute('SELECT key,value FROM meta')]
    imported=[dict(r) for r in con.execute("SELECT e.skill,COUNT(*) n,COUNT(DISTINCT e.answer) respostas,MIN(LENGTH(e.answer)) min_chars,MAX(LENGTH(e.answer)) max_chars FROM dialogue_examples e JOIN dialogue_groups g ON g.id=e.group_id WHERE g.split='train' GROUP BY e.skill")]
    answers=Counter();last=Counter();turns=Counter();maxwords=0
    for r in con.execute("SELECT e.messages,e.answer FROM dialogue_examples e JOIN dialogue_groups g ON g.id=e.group_id WHERE g.split='train'"):
        msg=json.loads(r['messages']);turns[len(msg)]+=1
        answers[' '.join(r['answer'].casefold().split())]+=1
        if msg:last[' '.join(msg[-1][1].casefold().split())]+=1
        maxwords=max(maxwords,len(r['answer'].split()))
    archive=ROOT/'reports/auditoria_pesos_e_dados/rodada_20261017_01'
    old=json.loads((archive/'datasets.json').read_text(encoding='utf-8'))
    corpus=next(iter(old['text_corpora'].values()))
    raw={}
    for name in ('books','inbox','dialogue_imports'):
        raw[name]=[{'file':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size} for p in sorted((ROOT/'data'/name).rglob('*')) if p.is_file()]
    pts=[]
    for parent in (ROOT/'data/checkpoints',ROOT/'reports'):
        for p in parent.rglob('*.pt'):
            if p.is_file() and not p.is_symlink() and not re.search(r'reservad|reserved|sealed|selado',p.name,re.I):
                pts.append({'file':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size})
    archives=[{'file':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size} for p in sorted((ROOT/'data/backups').glob('*')) if p.is_file()]
    result={'purpose':'checar material e identificar estritamente artefatos antes de reconstruir; não é nova auditoria de pesos',
        'source_library':sources,'source_characters_in_chunks':sum(s['chars'] or 0 for s in sources),
        'preserved_raw_files':raw,'old_training_corpus_reference':{k:corpus[k] for k in ('training_texts','training_characters','training_bpe_tokens_one_pass')},
        'imported_train_only':{'by_skill':imported,'rows':sum(x['n'] for x in imported),'unique_answers':len(answers),
            'unique_last_user_turns':len(last),'prompt_message_counts':dict(turns),'largest_answer_word_count':maxwords,
            'top_answer_counts':[{'answer':a,'count':n} for a,n in answers.most_common(12)],
            'not_approved_or_reused_for_new_training':True},
        'checkpoint_catalog_by_run_kind':kinds,'checkpoint_catalog':catalog,'runs':runs,'meta':meta,
        'candidate_pt_files_not_yet_deleted_or_classified':pts,'mixed_archives_preserved':archives,
        'training_started':False,'optimizer_steps':0,'deletions':0,'new_backups':0,
        'reserved_final_tests_read':False}
    with (OUT/'preparacao.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k in ('source_characters_in_chunks','old_training_corpus_reference','imported_train_only','checkpoint_catalog_by_run_kind')},ensure_ascii=False,indent=2))
    print('Candidatos .pt',len(pts),'bytes',sum(x['bytes'] for x in pts))
    print('Originais', {k:len(v) for k,v in raw.items()})
    print('Fontes',[(s['id'],s['title'],s['chars']) for s in sources])


if __name__=='__main__':main()
