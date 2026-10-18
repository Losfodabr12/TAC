"""Freeze development cases and validate newly prepared data before optimizer use."""
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
import numpy as np
from tac.config import atomic_json
from tac.neural.v2_data import load_jsonl,prompt_text,fingerprint
from tac.neural.v3 import V3Tokenizer
from evaluate_tac_v3 import cases
from prepare_tac_v3 import family,sha

base=ROOT/'data/tac_v3';out=ROOT/'reports/tac_v3/rodada_20261018_01'
if (base/'progress.json').exists():raise RuntimeError('Não redefinir avaliação de desenvolvimento depois de iniciar o treinamento')
ready=json.loads((base/'training_ready.json').read_text(encoding='utf-8'))
language=json.loads((base/'language_ready.json').read_text(encoding='utf-8'))
tok=V3Tokenizer.from_state(json.loads((base/'tokenizer.json').read_text(encoding='utf-8')))
rows={p:list(load_jsonl(base/f'instruction_curated_{p}.jsonl')) for p in ('train','validation')}
for part,values in rows.items():
    for row in values:
        prefix=[1]+tok.encode(prompt_text(row['messages']),special=False)
        answer=tok.encode(' '+row['answer'],special=False)+[2]
        assert row['sequence']==prefix+answer and row['prefix_length']==len(prefix)
        assert len(row['sequence'])<=1025 and row['sequence'][-1]==2
train_questions={r['prompt_hash'] for r in rows['train']}
train_families={r['group'] for r in rows['train']}
dev=[]
for row in cases():
    row.update(exact_prompt_seen_in_train=fingerprint(prompt_text(row['messages'])) in train_questions,
               lexical_initial_question_family_seen=family(row['messages'][0][1]) in train_families)
    dev.append(row)
atomic_json(out/'avaliacao_dev.json',{'created_before_training':True,'cases':dev,'semantic_family_independence_certified':False})
source_offsets={}
for part in ('train','validation'):
    values=np.memmap(base/f'language_{part}.u16',dtype='<u2',mode='r')
    assert values[0]==1 and values[-1]==2 and values.max()<8192 and values.min()>0
    eos=np.flatnonzero(values==2)
    fw_docs=language['stats']['HuggingFaceFW/fineweb-2:'+part+':docs']
    wiki_docs=language['stats']['wikimedia/wikipedia:'+part+':docs']
    assert len(eos)==fw_docs+wiki_docs
    boundary=int(eos[fw_docs-1])+1
    source_offsets[part]={'tokens':len(values),'documents':len(eos),'fineweb_tokens':boundary,
                          'wikipedia_tokens':len(values)-boundary,'wikipedia_start_offset':boundary}
atomic_json(base/'language_source_offsets.json',source_offsets)
acquired={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in base.iterdir()
          if p.is_file() and (p.name.endswith('.jsonl.gz') or p.name.startswith(('sft_','curated_sft_')) and p.suffix=='.jsonl')}
atomic_json(out/'arquivos_publicos_adquiridos.json',acquired)
result={'ok':True,'parameters':29414400,'dataset_hash':ready['dataset_hash'],'sft_train':len(rows['train']),
        'sft_validation':len(rows['validation']),'all_sft_sequences_reencoded_and_verified':True,
        'train_multiturn':sum(len(r['messages'])>1 for r in rows['train']),
        'new_development_cases':sum(not r['known_probe'] for r in dev),
        'new_exact_prompts_seen_in_train':[r['id'] for r in dev if not r['known_probe'] and r['exact_prompt_seen_in_train']],
        'language_sources':source_offsets,'reserved_tests_opened':False,
        'development_file_sha256':hashlib.sha256((out/'avaliacao_dev.json').read_bytes()).hexdigest()}
atomic_json(out/'preparacao_verificada.json',result)
print(json.dumps(result,ensure_ascii=False,indent=2))
