"""Avaliação reservada e rubricas limitadas explícitas, sem modelo julgador externo."""
from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter,defaultdict
from pathlib import Path

from ..config import atomic_json
from ..conversation_lessons import prompt_text
from .conversation import decode_reply


def plain(text):
    return ' '.join(''.join(c for c in unicodedata.normalize('NFKD',text.casefold()) if not unicodedata.combining(c)).split())


def make_suite(data,seed,kind):
    rng=random.Random(seed)
    occupied=set(re.findall(r'\w+',' '.join(t for rows in data.sets.values() for row in rows for r,t in row['messages']).casefold()))
    names=[]
    while len(names)<32:
        name=''.join(rng.choice(('za','rel','qui','no','vra','sil','do','me','lu','fa','en','tor')) for _ in range(4)).capitalize()
        if name.casefold() not in occupied and name not in names:names.append(name)
    episodes=[]
    for i in range(4):
        name,other=names[2*i:2*i+2]
        cases=[
            ('nome_inedito',[f'Pode me chamar de {name}.','Com qual nome você vai se dirigir a mim?'],name,[],f'Seu nome é {name}.'),
            ('correcao_negada',[f'Meu nome é {name}.',f'Eu me confundi: não me chamo {name}. Meu nome é {other}.','Qual nome vale agora?'],other,[name],f'Seu nome é {other}.'),
            ('pergunta_indireta',[f'Guarde só nesta conversa: o código da caixa é {name.lower()}.','Preciso abrir a caixa. Você lembra o código que informei?'],name.lower(),[],name.lower()+'.'),
            ('negacao',[f'Não tenho gato. Tenho um cachorro chamado {name}.','Eu disse que tenho gato?'],'não',['sim'],'Não.'),
            ('local_atual',[f'O estojo estava em {name.lower()}.',f'Ele saiu de {name.lower()} e foi para {other.lower()}.','Se eu procurar o estojo agora, onde devo ir?'],other.lower(),[name.lower()],f'Em {other.lower()}.'),
            ('memoria_curta',[f'Meu nome é {name}.','Vamos conversar sobre uma caminhada.','Retomando: qual é meu nome?'],name,[],f'Seu nome é {name}.'),
            ('memoria_apos_janela',[f'Meu nome é {name}.','Hoje observei uma praça.','Depois vi uma árvore.','Também vi uma fonte.','Agora voltemos à identificação: qual era meu nome?'],name,[],f'Seu nome é {name}.'),
            ('situacao_aberta',['Tenho dois amigos que querem horários diferentes para estudar. Como posso organizar isso sem escolher por eles?'],None,[],None),
        ]
        for skill,users,expected,forbidden,exact in cases:
            episodes.append({'id':f'{kind}_{skill}_{i}','skill':skill,'users':users,'expected_fact':expected,
                'forbidden_facts':forbidden,'exact_reference':exact,'needs_semantic_human_review':skill=='situacao_aberta'})
    prompts={' '.join(prompt_text(row['messages']).casefold().split()) for rows in data.sets.values() for row in rows}
    for episode in episodes:
        assert all(' '.join(prompt_text([['user',u]]).casefold().split()) not in prompts for u in episode['users'])
    return {'format':'TAC-transfer-evaluation-v1','kind':kind,'seed':seed,'dataset_hash':data.digest,
            'episodes':episodes,'policy':'Somente avaliação. Nenhum item entra no treinamento. Teste reservado é executado em um único evento final.'}


def freeze_suite(path,data,seed,kind):
    path=Path(path)
    if path.exists():raise FileExistsError('O conjunto já foi congelado; não será substituído.')
    suite=make_suite(data,seed,kind)
    atomic_json(path,suite)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(path.with_suffix('.manifest.json'),{'sha256':digest,'format':suite['format'],'dataset_hash':data.digest,
        'rubric':'exact text + positive literal fact + context conflict rules; general semantics pending human review'})
    return suite


def score(text,episode):
    expected=episode['expected_fact']; reference=episode['exact_reference']
    words=plain(text)
    if expected is None:
        return {'exact':None,'factual_correct':None,'context_consistent':None,'semantic_equivalence':None,'semantic_review':'pendente_humano'}
    target=plain(expected)
    def positive(value):
        for match in re.finditer(r'(?<!\w)'+re.escape(plain(value))+r'(?!\w)',words):
            before=words[:match.start()].split()[-4:]
            if not any(w in ('nao','talvez','possivelmente') for w in before):return True
        return False
    if target in ('sim','nao'):
        factual=bool(re.match(r'^'+target+r'\b',words))
        consistent=factual and not bool(re.match(r'^(?:sim.*\bnao|nao.*\bsim)\b',words))
    else:
        factual=positive(expected)
        consistent=factual and not any(positive(value) for value in episode['forbidden_facts'])
    return {'exact':text==reference,'factual_correct':factual,'context_consistent':consistent,
            'semantic_equivalence':None,'semantic_equivalence_closed_proxy':factual and consistent,
            'semantic_review':'Rubrica literal limitada. Equivalência semântica geral exige revisão humana.'}


def evaluate_suite(model,suite,device,gate,generation_options=None):
    results=[]
    for episode in suite['episodes']:
        history=[]; visible=True
        for user in episode['users']:
            gate.check();messages=[*history[-6:],['user',user]]
            if episode['expected_fact'] and user==episode['users'][-1]:
                tokenizer=model.text_tokenizer
                visible=plain(episode['expected_fact']) in plain(tokenizer.decode(tokenizer.encode(prompt_text(messages),special=False)[-model.cfg.context+1:]))
            answer=decode_reply(model,messages,device,generation_options=generation_options)
            history.extend([['user',user],['assistant',answer]])
        results.append({'id':episode['id'],'skill':episode['skill'],'actual':answer,'expected':episode['exact_reference'],
            'evidence_literal_visible':visible,'history':history,**score(answer,episode)})
    by_skill={}
    for skill in sorted({r['skill'] for r in results}):
        rows=[r for r in results if r['skill']==skill]; stats={'episodes':len(rows)}
        for metric in ('exact','factual_correct','context_consistent','semantic_equivalence'):
            measured=[r[metric] for r in rows if r[metric] is not None]
            stats[metric]={'correct':sum(measured),'total':len(measured),'accuracy':sum(measured)/len(measured) if measured else None}
        stats['evidence_not_visible']=sum(not r['evidence_literal_visible'] for r in rows)
        by_skill[skill]=stats
    return {'by_skill':by_skill,'results':results,'scope':'32 episódios sintéticos novos. Fatos/consistência têm rubrica literal; semântica geral não avaliada automaticamente.'}


def evaluate_once(path,models,device,gate):
    path=Path(path); manifest=json.loads(path.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['sha256']:raise ValueError('Teste reservado alterado.')
    ledger=path.with_suffix('.consumed.json')
    # Criação exclusiva antecede inferência: não se pode usar repetidamente para ajustar.
    with ledger.open('x',encoding='utf-8') as file:
        json.dump({'suite_sha256':manifest['sha256'],'models':[label for label,model in models],'state':'iniciado'},file)
    suite=json.loads(path.read_text(encoding='utf-8')); report={'suite_sha256':manifest['sha256'],'models':{}}
    configs={'greedy':{'greedy':True},'controlled':{'greedy':False,'temperature':.65,'top_k':30,'top_p':.9,'repetition_penalty':1.08,'seed':72841}}
    for label,model in models:
        report['models'][label]={name:evaluate_suite(model,suite,device,gate,options) for name,options in configs.items()}
    atomic_json(path.with_suffix('.results.json'),report)
    atomic_json(ledger,{'suite_sha256':manifest['sha256'],'models':list(report['models']),'state':'concluido'})
    return report


def review_template(results):
    """Ficha vazia: nunca inventa notas semânticas para a rede."""
    rows=[]
    for model,modes in results['models'].items():
        for mode,data in modes.items():
            for result in data['results']:
                rows.append({'model':model,'mode':mode,'id':result['id'],'skill':result['skill'],
                    'answer_sha256':hashlib.sha256(result['actual'].encode('utf-8')).hexdigest(),
                    'history':result['history'],'answer':result['actual'],'expected':result['expected'],
                    'semantic_equivalence':None})
    return {'format':'TAC-human-semantic-review-v1','suite_sha256':results['suite_sha256'],'rows':rows,
        'instructions':'Marque true, false ou null após ler o contexto. null significa não avaliado. A nota não modifica pesos nem aprovação.'}


def validate_review(results,review):
    if review.get('format')!='TAC-human-semantic-review-v1' or review.get('suite_sha256')!=results['suite_sha256']:
        raise ValueError('Ficha não corresponde à avaliação.')
    allowed={(r['model'],r['mode'],r['id']):r for r in review_template(results)['rows']}
    seen=set(); groups=defaultdict(list)
    for row in review['rows']:
        key=row['model'],row['mode'],row['id']
        if key in seen or key not in allowed:raise ValueError('Resposta ausente ou duplicada na ficha.')
        seen.add(key)
        if row['answer_sha256']!=allowed[key]['answer_sha256'] or row['answer']!=allowed[key]['answer']:
            raise ValueError('A resposta foi alterada depois da avaliação.')
        if row['skill']!=allowed[key]['skill'] or row['history']!=allowed[key]['history'] or row['expected']!=allowed[key]['expected']:
            raise ValueError('O contexto ou a habilidade foi alterado na ficha.')
        value=row['semantic_equivalence']
        if value is not None and type(value) is not bool:raise ValueError('Use true, false ou null na revisão.')
        groups[(key[0],key[1],row['skill'])].append(value)
    return {'format':'TAC-human-semantic-results-v1','suite_sha256':results['suite_sha256'],
        'source':'avaliação fornecida pelo usuário, sem julgador neural externo','rows_missing':len(allowed)-len(seen),
        'by_model_mode_skill':[{'model':m,'mode':d,'skill':s,'correct':sum(v is True for v in values),
            'total':sum(v is not None for v in values),'pending':sum(v is None for v in values),
            'accuracy':sum(v is True for v in values)/sum(v is not None for v in values) if any(v is not None for v in values) else None}
            for (m,d,s),values in groups.items()], 'changes_weights':False,'changes_approval':False}
