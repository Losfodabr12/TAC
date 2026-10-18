"""Revisão anterior ao treino; preserva a primeira versão preparada do corpus."""
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
import yaml
from prepare_v2_conversation import BASE,OUT,fetch,safe
from tac.config import atomic_json
from tac.neural.bpe import LocalBPETokenizer
from tac.neural.v2_data import canonical,fingerprint,load_jsonl,prompt_text,PT


def main():
    if (BASE/'curation.json').exists():print('Revisão já aplicada.');return
    ready=json.loads((BASE/'ready.json').read_text(encoding='utf-8'))
    atomic_json(BASE/'pre_curation_manifest.json',ready)
    tok=LocalBPETokenizer.from_state(json.loads((ROOT/'data/tac_v2/tokenizer.json').read_text(encoding='utf-8')))
    rev=json.loads(fetch('https://api.github.com/repos/gunthercox/chatterbot-corpus/commits/master',BASE/'dialogue_revision.json',2**20))['sha']
    fetch(f'https://raw.githubusercontent.com/gunthercox/chatterbot-corpus/{rev}/license.md',BASE/'licenses/chatterbot_license.md',2**20)
    dialogues=[]
    for file in ('greetings','conversations'):
        raw=fetch(f'https://raw.githubusercontent.com/gunthercox/chatterbot-corpus/{rev}/chatterbot_corpus/data/portuguese/{file}.yml',BASE/(file+'.yml'),2**20)
        parsed=yaml.safe_load(raw)['conversations']
        # conversations: somente os três diálogos iniciais revisados; exclui
        # traduções quebradas, identidade fictícia e respostas sem nexo posteriores.
        if file=='conversations':parsed=parsed[:3]
        seen=set()
        for n,chain in enumerate(parsed):
            for turn in range(1,len(chain),2):
                messages=[['user' if i%2==0 else 'assistant',t] for i,t in enumerate(chain[:turn])]
                key=fingerprint(prompt_text(messages))
                if key in seen:continue  # Alternativas de humor não são fatos contraditórios.
                seen.add(key)
                if any(safe(t) for t in chain[:turn+1]):continue
                if any(s in canonical(chain[turn]) for s in ('não muito bem','não estou tão bem','céu esta acima','poderia estar melhor')):continue
                dialogues.append({'messages':messages,'answer':chain[turn],'source':'gunthercox/chatterbot-corpus',
                    'source_id':f'{file}:{n}:{turn}','group':'dialogue:'+file+':'+str(n),'family':'dialogue:'+file+':'+str(n),
                    'skill':'conversation','split':'train','license':'BSD-3-Clause','revision':rev})
    counts=Counter();curated={}
    for split in ('train','validation','retention'):
        values=[];reading=0
        for r in load_jsonl(BASE/(split+'.jsonl')):
            q=canonical(r['messages'][-1][1]);a=canonical(r['answer'])
            if r['source'].startswith(('CohereLabs/','OpenAssistant/')):
                if re.search(r'pergunta da categoria|qual.*categoria|finja que você é|em latim|professor de inglês|speak|thank you',q):
                    counts['reviewed_inadequate_instruction']+=1;continue
                words=re.findall(r'\w+',a)
                if len(words)>12 and not any(w in PT for w in words):counts['non_portuguese_answer']+=1;continue
                if re.search(r'\b(?:escreva|crie|redija|componha|invente)\b',q):r['skill']='writing'
                elif r['skill']=='writing':r['skill']='general'
            if r['source']=='TAC-extraction-v2':
                # Corrige a concordância do atributo das cores, sem mudar o alvo.
                replacements=[('uma mochila','um caderno'),('da minha mochila','do meu caderno'),
                              ('a mochila','o caderno'),('da mochila','do caderno'),('para a mochila','para o caderno')]
                for pair in r['messages']:
                    for old,new in replacements:pair[1]=pair[1].replace(old,new)
            if split=='train' and r['skill']=='reading':
                reading+=1
                if reading>2500:counts['reading_share_capped']+=1;continue
            values.append(r)
        if split=='train':values+=dialogues
        seen=set();result=[]
        for r in values:
            key=fingerprint(prompt_text(r['messages']))
            if key in seen:continue
            seen.add(key)
            prefix=[1,*tok.encode(prompt_text(r['messages']),special=False)];answer=tok.encode(' '+r['answer'],special=False)+[2]
            r.update(sequence=prefix+answer,prefix_length=len(prefix),supervised_tokens=len(answer))
            result.append(r)
        curated[split]=result
        with (BASE/('curated_'+split+'.jsonl')).open('x',encoding='utf-8') as f:
            for r in result:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    assert not {r['family'] for r in curated['train']}&{r['family'] for r in curated['validation']}
    ready['sources']['gunthercox/chatterbot-corpus']={'revision':rev,'license':'BSD-3-Clause',
        'url':'https://github.com/gunthercox/chatterbot-corpus','modification':'selected greetings and first three natural conversation chains; neural training only, no response lookup'}
    for key in ('counts','by_source','by_skill','supervised_tokens','files'):
        ready[key]={}
    for s,values in curated.items():
        ready['counts'][s]=len(values);ready['by_source'][s]=dict(Counter(r['source'] for r in values));ready['by_skill'][s]=dict(Counter(r['skill'] for r in values))
        ready['supervised_tokens'][s]=sum(r['supervised_tokens'] for r in values)
        ready['files'][s]=hashlib.sha256((BASE/('curated_'+s+'.jsonl')).read_bytes()).hexdigest()
    ready['dataset_hash']=fingerprint(json.dumps(ready['files'],sort_keys=True));ready['curated']=True
    review={'manual_sample_file':'amostra_revisao.json','scope':'stratified sample reviewed; no universal factual certification',
            'issues_found':['misclassified creative tasks','category quizzes','non-Portuguese answers','color noun agreement'],
            'changes':dict(counts),'new_public_dialogue_rows':len(dialogues),
            'no_targets_generated_by_external_models':True,'original_corpus_files_preserved':True}
    atomic_json(BASE/'curation.json',review);atomic_json(OUT/'REVISAO.json',review)
    atomic_json(BASE/'ready.json',ready);atomic_json(OUT/'DADOS.json',ready)
    print(json.dumps({'counts':ready['counts'],'skills':ready['by_skill'],'review':review},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
