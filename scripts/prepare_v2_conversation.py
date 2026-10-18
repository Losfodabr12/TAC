"""Novo corpus SFT: dados públicos e alvos determinísticos, sem inferência externa."""
from __future__ import annotations
import hashlib
import json
import os
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
BASE=ROOT/'data/tac_v2_conversation_20261018';OUT=ROOT/'reports/conversacao_v2/rodada_20261018_01'
for k,v in {'HF_HOME':str(ROOT/'data/tac_v2/cache'),'HF_HUB_DISABLE_TELEMETRY':'1',
            'HF_HUB_DISABLE_XET':'1','DO_NOT_TRACK':'1'}.items():os.environ[k]=v
from tac.neural.v2_data import clean,canonical,fingerprint,problem,label_values,load_jsonl,prompt_text
from tac.neural.bpe import LocalBPETokenizer
from tac.config import atomic_json


def fetch(url,path,limit=60*2**20):
    if path.exists():return path.read_bytes()
    req=urllib.request.Request(url,headers={'User-Agent':'TAC-local-corpus/2.0'})
    with urllib.request.urlopen(req,timeout=90) as r:content=r.read(limit+1)
    if len(content)>limit:raise ValueError('Fonte excedeu o limite explícito')
    path.write_bytes(content);return content


def safe(text):
    issue=problem(text)
    if issue=='too_short' and re.search(r'\w',text):return None
    return issue


def skill(question):
    q=canonical(question)
    if re.search(r'\b(história|historia|conto|poema|narrativa|crie|escreva|redija)\b',q):return 'writing'
    if re.search(r'\b(olá|ola|oi|bom dia|boa tarde|boa noite|tudo bem|tudo bom|obrigad)\b',q):return 'conversation'
    if re.search(r'\d\s*[+*/×÷=-]\s*\d|calcule|aritmética|multipli|divis[ãa]o|som[ae]',q):return 'math_public'
    return 'general'


def public_rows():
    from datasets import load_dataset
    from huggingface_hub import HfApi
    sources={}
    for repo,rev in [('CohereLabs/aya_dataset','f9ea04583f02a8f86404ff6c58bf75fe637df8a2'),
                     ('OpenAssistant/oasst2','179dd21fc55192153d94adb0e0ce8f69e222bf75'),
                     ('OpenAssistant/oasst1',None)]:
        rev=rev or HfApi().dataset_info(repo).sha
        name=repo.rsplit('/',1)[-1];raw=BASE/(name+'_pt_train.jsonl')
        sources[repo]={'revision':rev,'license':'Apache-2.0','only_split':'train','url':'https://huggingface.co/datasets/'+repo}
        fetch(f'https://huggingface.co/datasets/{repo}/raw/{rev}/README.md',BASE/'licenses'/(name+'_README.md'),2**20)
        if not raw.exists():
            filters=[('language_code','==','por')] if name=='aya_dataset' else [('lang','in',['pt','pt-BR','pt-PT'])]
            kw={'name':'default'} if name=='aya_dataset' else {}
            with raw.open('x',encoding='utf-8') as f:
                for i,row in enumerate(load_dataset(repo,revision=rev,split='train',streaming=True,filters=filters,**kw)):
                    if name=='aya_dataset':
                        item={'messages':[['user',clean(row['inputs'])]],'answer':clean(row['targets']),
                              'source':repo,'source_id':str(i),'group':'aya:'+fingerprint(row['inputs']),
                              'annotation_type':row.get('annotation_type')}
                    else:item={k:v for k,v in row.items() if k not in ('user_id','emojis','detoxify','created_date')}
                    f.write(json.dumps(item,ensure_ascii=False)+'\n')
        print('Fonte disponível:',repo,raw.stat().st_size,flush=True)
    atomic_json(BASE/'sources.json',sources)
    return sources


def conversations(repo,counts):
    rows=list(load_jsonl(BASE/(repo.rsplit('/',1)[-1]+'_pt_train.jsonl')))
    lookup={r['message_id']:r for r in rows};best={}
    for r in rows:
        if r['role']=='assistant':
            rank=r['rank'] if r.get('rank') is not None else 10**6
            key=r.get('parent_id')
            if key not in best or (rank,r['message_id'])<best[key][0]:best[key]=((rank,r['message_id']),r)
    for _,r in best.values():
        path=[];seen=set();cursor=r;reason=None
        while cursor:
            if cursor['message_id'] in seen:reason='cycle';break
            seen.add(cursor['message_id']);path.append(cursor)
            labels=label_values(cursor)
            if cursor.get('deleted') or cursor.get('synthetic') or cursor.get('review_result') is False:reason='rejected';break
            if any(float(labels.get(k,0) or 0)>.1 for k in ('pii','spam','lang_mismatch','not_appropriate')):reason='annotation';break
            if cursor['role']=='assistant' and float(labels.get('quality',.6))<.5:reason='quality';break
            reason=safe(clean(cursor.get('text')))
            if reason:break
            parent=cursor.get('parent_id')
            if parent and parent not in lookup:reason='missing_parent';break
            cursor=lookup.get(parent)
        if reason:counts['oasst_'+reason]+=1;continue
        path.reverse()
        if not path or len(path)%2 or any(t['role']!=('prompter' if i%2==0 else 'assistant') for i,t in enumerate(path)):continue
        yield {'messages':[['user' if t['role']=='prompter' else 'assistant',clean(t['text'])] for t in path[:-1]],
               'answer':clean(r['text']),'source':repo,'source_id':r['message_id'],
               'group':'oasst:'+str(r.get('message_tree_id') or path[0]['message_id']),
               'ancestor_ids':[t['message_id'] for t in path[:-1]]}


def computed():
    """Referências calculadas/extraídas. Não gera prosa com uma IA."""
    rng=random.Random(202610181)
    math_templates=[('m_symbol','{a} {op} {b} = ?'),('m_compute','Calcule {a} {op} {b}.'),
        ('m_value','Qual é o resultado de {a} {op} {b}?'),('m_operation','Resolva esta operação: {a} {op} {b}.'),
        ('m_request','Preciso do valor de {a} {op} {b}.'),('m_held_value','Quanto resulta a conta {a} {op} {b}?'),
        ('m_held_instruction','Dê apenas o número obtido ao calcular {a} {op} {b}.')]
    seen=set()
    for i in range(3200):
        a,b=rng.randrange(0,40),rng.randrange(0,25);op=rng.choice(['+','-','×'])
        if op=='-':a,b=max(a,b),min(a,b)
        key=(a,b,op)
        if key in seen:continue
        seen.add(key);family,template=math_templates[i%len(math_templates)]
        answer=a+b if op=='+' else a-b if op=='-' else a*b
        yield {'messages':[['user',template.format(a=a,b=b,op=op)]],'answer':str(answer),'source':'TAC-computed-v2',
               'source_id':'math:'+str(i),'group':family,'family':family,'skill':'arithmetic',
               'split':'validation' if 'held' in family else 'train','verification':{'a':a,'b':b,'op':op,'result':answer}}
    # Famílias e vocabulários distintos, sem reutilizar exercícios antigos.
    train_names='Aline Bianca Célia Denise Elisa Fábio Giovana Heitor Ingrid Jonas Laís Mauro Nádia Otávio Paula Renata Sérgio Tânia Ulisses Vitória'.split()
    val_names='Zilda Iara Murilo Cíntia Dália Bento'.split()
    colors=['azul','verde','vermelho','amarelo','roxo','laranja','branco','preto']
    for held,names in [(False,train_names),(True,val_names)]:
        for i in range(700 if not held else 100):
            a,b,c=rng.sample(names,3);x,y=rng.sample(colors,2)
            kind=i%7
            if kind<3:
                messages=[['user',f'Meu nome é {a}.'],['assistant',a+'.'],['user',f'Passei a usar o nome {b}.'],['assistant',b+'.']]
                if held:q=['Como devo ser chamado neste momento?','Qual nome foi substituído?','Recupere o nome informado no início.'][kind]
                else:q=rng.choice([['Qual é meu nome agora?','Diga meu nome atual.','Que nome estou usando?'],
                                    ['Qual era meu nome antes da mudança?','Diga meu nome anterior.','Que nome deixei de usar?'],
                                    ['Qual foi o primeiro nome que informei?','Diga o nome inicial.','Como me apresentei primeiro?']][kind])
                messages.append(['user',q]);answer=b if kind==0 else a;label=['current','previous','first'][kind]
            elif kind==3:
                q=f'Meu nome é {a}. Tenho uma mochila {x}. Qual é meu nome?' if not held else f'Pode me chamar de {a}. Qual nome acabei de informar?'
                messages=[['user',q]];answer=a;label='name_copy'
            elif kind==4:
                messages=[['user',f'A cor da minha mochila é {x}.'],['assistant',x+'.'],['user',f'Troquei a mochila: a cor agora é {y}.'],
                          ['assistant',y+'.'],['user','Qual é a cor atual da mochila?' if not held else 'Depois da troca, qual cor vale para a mochila?']]
                answer=y;label='context_update'
            elif kind==5:
                messages=[['user',f'{a} escolheu {x} e {b} escolheu {y}. '+(f'Qual cor {b} escolheu?' if not held else f'Identifique a cor escolhida por {b}.')]]
                answer=y;label='binding'
            else:
                messages=[['user',f'Listei os nomes {a}, {b} e {c}. '+('Repita os nomes na mesma ordem.' if not held else 'Recupere a sequência de nomes sem mudar a ordem.')]]
                answer=f'{a}, {b}, {c}';label='ordering'
            family=('held_' if held else '')+label
            yield {'messages':messages,'answer':answer,'source':'TAC-extraction-v2','source_id':family+':'+str(i),
                   'family':family,'group':family,'skill':label,'split':'validation' if held else 'train',
                   'verification':'literal span or deterministic chronological extraction'}


def main():
    BASE.mkdir(exist_ok=True);OUT.mkdir(parents=True,exist_ok=True);(BASE/'licenses').mkdir(exist_ok=True)
    if (BASE/'ready.json').exists():print('Corpus adicional pronto; preservado.');return
    sources=public_rows();counts=Counter();rows=list(load_jsonl(BASE/'aya_dataset_pt_train.jsonl'))
    rows+=list(conversations('OpenAssistant/oasst2',counts))+list(conversations('OpenAssistant/oasst1',counts))
    # Fonte nova de compreensão, somente train. Nunca baixa dev/test.
    rev=json.loads(fetch('https://api.github.com/repos/nunorc/squad-v1.1-pt/commits/master',BASE/'squad_revision.json',2**20))['sha']
    for name in ('LICENSE','README.md'):
        fetch(f'https://raw.githubusercontent.com/nunorc/squad-v1.1-pt/{rev}/{name}',BASE/'licenses'/('squad_'+name),2**20)
    data=json.loads(fetch(f'https://raw.githubusercontent.com/nunorc/squad-v1.1-pt/{rev}/train-v1.1-pt.json',BASE/'squad_train.json'))
    sources['nunorc/squad-v1.1-pt']={'revision':rev,'license':'MIT translation repository; underlying SQuAD CC-BY-SA-4.0',
        'url':'https://github.com/nunorc/squad-v1.1-pt','modification':'train-only sample; answer must occur literally in passage; split by article'}
    qa=[]
    for article in data['data']:
        candidates=[]
        for paragraph in article['paragraphs']:
            context=clean(paragraph['context'])
            if len(context)>1700:continue
            for q in paragraph['qas']:
                answer=clean(q['answers'][0]['text']) if q.get('answers') else ''
                if not answer or answer not in context:counts['qa_answer_not_in_context']+=1;continue
                candidates.append({'messages':[['user','Leia o texto e responda à pergunta.\nTexto: '+context+'\nPergunta: '+clean(q['question'])]],
                    'answer':answer,'source':'nunorc/squad-v1.1-pt','source_id':q['id'],'group':'article:'+article['title'],
                    'family':'article:'+article['title'],'skill':'reading','verification':'answer is literal substring of context'})
        random.Random(fingerprint(article['title'])).shuffle(candidates);qa.extend(candidates[:20])
    random.Random(202610182).shuffle(qa);rows+=qa[:6500]
    counts['qa_sampled']=min(6500,len(qa))
    oldtrain=list(load_jsonl(ROOT/'data/tac_v2/instruction_train.jsonl'))
    oldval=list(load_jsonl(ROOT/'data/tac_v2/instruction_validation.jsonl'))
    oldtrain_prompts={fingerprint(prompt_text(r['messages'])) for r in oldtrain}
    blocked_groups={r['group'] for r in oldval}
    blocked_texts={fingerprint(text) for r in oldval for role,text in r['messages'] if role=='user'}
    # Famílias textuais: variação de assunto/nomes não cria uma família nova.
    def family(r):
        if r.get('family'):return r['family']
        q=canonical(r['messages'][-1][1]);q=re.sub(r'\d+','N',q)
        return 'public:'+' '.join(q.split()[:4])
    tok=LocalBPETokenizer.from_state(json.loads((ROOT/'data/tac_v2/tokenizer.json').read_text(encoding='utf-8')))
    accepted=[]
    for r in rows:
        r['family']=family(r);r['skill']=r.get('skill') or skill(r['messages'][-1][1])
        isqa=r['source']=='nunorc/squad-v1.1-pt'
        f=r['family'];r['split']='validation' if int(fingerprint(f)[:8],16)%12==0 else 'train'
        if r['group'] in blocked_groups or any(fingerprint(t) in blocked_texts for role,t in r['messages'] if role=='user'):
            counts['old_validation_protected']+=1;continue
        key=fingerprint(prompt_text(r['messages']))
        if r['split']=='validation' and key in oldtrain_prompts:
            # Já estudado: retenção, nunca anunciado como generalização inédita.
            r['split']='retention'
        if len(r['answer'])>1500 or any(len(t)>3500 for _,t in r['messages']):counts['too_long_for_basic_curriculum']+=1;continue
        reason=next((safe(t) for t in [x for _,x in r['messages']]+[r['answer']] if safe(t)),None)
        if reason:counts[reason]+=1;continue
        # Exclui blocos formulaicos, múltipla escolha e questões especializadas de bandeiras.
        if re.search(r'\b(?:bandeira|alternativa correta|categoria desta pergunta|opção correta)\b',canonical(r['messages'][-1][1])):
            counts['formulaic_quiz_excluded']+=1;continue
        accepted.append(r)
    accepted+=list(computed())
    # Deduplicação de prompts com resposta conflitante e de famílias entre splits.
    groups=defaultdict(list)
    for r in accepted:groups[fingerprint(prompt_text(r['messages']))].append(r)
    unique=[]
    for group in groups.values():
        if len({canonical(r['answer']) for r in group})>1:counts['contradictory_prompt_removed']+=len(group);continue
        counts['duplicate_removed']+=len(group)-1;unique.append(group[0])
    # Árvores inteiras no mesmo split; val somente famílias que nunca entram no treino.
    valfamilies={r['family'] for r in unique if r['split']=='validation'}
    valtrees={r['group'] for r in unique if r['split']=='validation' and r['source'].startswith('OpenAssistant/')}
    for r in unique:
        if r['split']=='train' and (r['family'] in valfamilies or r['group'] in valtrees):r['split']='validation'
        if r['split']=='validation' and fingerprint(prompt_text(r['messages'])) in oldtrain_prompts:r['split']='retention'
    bysplit=defaultdict(list)
    for r in unique:
        prefix=[1,*tok.encode(prompt_text(r['messages']),special=False)];answer=tok.encode(' '+r['answer'],special=False)+[2]
        if len(prefix)+len(answer)-1>1024:counts['over_context_excluded_without_cropping']+=1;continue
        r.update(sequence=prefix+answer,prefix_length=len(prefix),supervised_tokens=len(answer))
        bysplit[r['split']].append(r)
    trainfamilies={r['family'] for r in bysplit['train']};valfamilies={r['family'] for r in bysplit['validation']}
    assert not trainfamilies&valfamilies
    assert not {fingerprint(prompt_text(r['messages'])) for r in bysplit['train']}&{fingerprint(prompt_text(r['messages'])) for r in bysplit['validation']}
    for split,values in bysplit.items():
        with (BASE/(split+'.jsonl')).open('x',encoding='utf-8') as f:
            for r in values:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    # Amostra estratificada para revisão humana do agente (sem gerar/rescrever alvos).
    reviews=[]
    for source in sorted({r['source'] for r in bysplit['train']}):
        for s in sorted({r['skill'] for r in bysplit['train'] if r['source']==source}):
            candidates=[r for r in bysplit['train'] if r['source']==source and r['skill']==s]
            random.Random(731).shuffle(candidates)
            reviews.extend({k:r[k] for k in ('source','source_id','skill','messages','answer')} for r in candidates[:3])
    atomic_json(OUT/'amostra_revisao.json',reviews)
    stats={'generation':'TAC-v2','sources':sources,'counts':{s:len(v) for s,v in bysplit.items()},
           'by_source':{s:dict(Counter(r['source'] for r in v)) for s,v in bysplit.items()},
           'by_skill':{s:dict(Counter(r['skill'] for r in v)) for s,v in bysplit.items()},
           'supervised_tokens':{s:sum(r['supervised_tokens'] for r in v) for s,v in bysplit.items()},
           'families':{s:len({r['family'] for r in v}) for s,v in bysplit.items()},'filters':dict(counts),
           'files':{s:hashlib.sha256((BASE/(s+'.jsonl')).read_bytes()).hexdigest() for s in bysplit},
           'family_overlap':0,'prompt_overlap':0,'short_valid_answers_allowed':True,'old_v1_exercises_used':False,
           'external_model_calls':0,'synthetic_component':'arithmetic calculated in Python; context answers copied literally, no LLM-generated targets',
           'manual_review_scope':'stratified sample only; not every public factual claim certified',
           'reserved_tests_opened':False,'context':1024,'tokenizer_unchanged':True}
    stats['dataset_hash']=fingerprint(json.dumps(stats['files'],sort_keys=True))
    atomic_json(BASE/'sources.json',sources);atomic_json(BASE/'ready.json',stats);atomic_json(OUT/'DADOS.json',stats)
    print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
