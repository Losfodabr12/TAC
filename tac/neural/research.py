"""Ferramentas experimentais locais. Nunca promovem um modelo para o chat."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter,defaultdict

import torch

from ..conversation_lessons import canonical,prompt_text
from .conversation import ConversationData


def precise_prompt(messages):
    """Identidade conservadora: preserva operadores, acentos e histórico completo."""
    return ' '.join(unicodedata.normalize('NFKC',prompt_text(messages)).casefold().split())


def row_identity(row):
    return precise_prompt(row['messages']),unicodedata.normalize('NFKC',row['answer'])


def protected_selection_key(validation,baseline,internal_loss,imported_loss):
    """Validação importada só desempata versões que passam na proteção básica."""
    protected=validation['loss']<baseline['loss']*1.02
    return (0,imported_loss,internal_loss) if protected else (1,internal_loss,imported_loss)


def audit_data(data,context):
    result={'dataset_hash':data.digest,'context':context,'splits':{},
        'politica':'Somente estatísticas; nenhum exemplo foi removido ou alterado.'}
    identities={}; visible_identities={}; groups={}
    for split,rows in data.sets.items():
        lossy=defaultdict(list); precise=defaultdict(list); visible=defaultdict(list); length=Counter(); skills=Counter(); pools=Counter()
        groups[split]={r['group'] for r in rows if r.get('imported') and r.get('group')}
        cropped=turn_crop=answer_large=0; orphan=0; max_answer=0
        for row in rows:
            all_messages=row['messages']; messages=all_messages[-7:] if row.get('imported') else all_messages
            lossy[canonical(prompt_text(messages))].append(row)
            precise[precise_prompt(all_messages)].append(row)
            length[len(all_messages)]+=1; skills[row['skill']]+=1; pools[row.get('mixture','legacy')]+=1
            answer=len(data.tokenizer.encode(' '+row['answer'],special=False))+1
            prefix=len(data.tokenizer.encode(prompt_text(messages),special=False))
            # Igual ao prefixo que encode entrega no SFT: reserva espaço da resposta.
            ids=data.tokenizer.encode(prompt_text(messages),special=False)[-max(1,context-answer):]
            visible[hashlib.sha256(json.dumps(ids).encode()).hexdigest()].append(row)
            max_answer=max(max_answer,answer)
            cropped+=prefix>context-answer; turn_crop+=len(all_messages)>len(messages)
            answer_large+=answer>=context-4
            question=canonical(all_messages[-1][1])
            if re.search(r'\b(?:qual|lembra|recorda).*\bmeu nome\b',question):
                preceding=' '.join(canonical(t) for role,t in all_messages[:-1] if role=='user')
                orphan+=not bool(re.search(r'\b(?:meu nome e|me chamo|me chama de|pode me chamar de)\b',preceding))
        ambiguous=[xs for xs in lossy.values() if len({canonical(x['answer']) for x in xs})>1]
        result['splits'][split]={
            'examples':len(rows),'legacy_normalized_duplicates':sum(len(xs)-1 for xs in lossy.values()),
            'legacy_ambiguous_keys':len(ambiguous),
            'normalized_collisions_with_distinct_precise_inputs':sum(len({precise_prompt(x['messages']) for x in xs})>1 for xs in ambiguous),
            'precise_inputs_with_multiple_answer_texts':sum(len({x['answer'] for x in xs})>1 for xs in precise.values()),
            'model_visible_inputs_with_multiple_answer_texts':sum(len({x['answer'] for x in xs})>1 for xs in visible.values()),
            'visible_input_collisions_introduced_by_truncation':sum(len({precise_prompt(x['messages']) for x in xs})>1 for xs in visible.values()),
            'identical_prompt_and_answer_repeats':len(rows)-len({row_identity(r) for r in rows}),
            'ambiguous_keys_by_skill':dict(Counter(s for xs in ambiguous for s in {x['skill'] for x in xs})),
            'histories_messages':dict(length),'by_skill':dict(skills),'by_pool':dict(pools),
            'prefixes_token_cropped':cropped,'histories_turn_cropped':turn_crop,'answers_out_of_context':answer_large,
            'max_answer_tokens':max_answer,'name_recall_without_recognized_user_name_in_history':orphan,
            'interpretation':'Respostas diferentes podem ser paráfrases válidas. Ausência de um padrão de nome é sinal para revisão, não prova de erro.'}
        identities[split]=set(precise);visible_identities[split]=set(visible)
    train=identities.get('train',set());train_visible=visible_identities.get('train',set());train_groups=groups.get('train',set())
    result['overlap_with_train']={split:{
        'identical_full_prompt_keys':len(train&keys),
        'identical_model_visible_prefix_keys':len(train_visible&visible_identities[split]),
        'shared_imported_groups':len(train_groups&groups[split])}
        for split,keys in identities.items() if split!='train'}
    result['overlap_scope']='Interseção de chaves distintas, não de linhas. Igualdade não prova vazamento semântico; grupos só medidos para importações.'
    return result


def unique_training_view(data):
    """Visão derivada opcional: só colapsa prompt COMPLETO + resposta idênticos."""
    seen=set(); train=[]
    for row in data.sets['train']:
        key=(row.get('mixture','legacy'),*row_identity(row))
        if key not in seen:train.append(row); seen.add(key)
    return ConversationData({**data.sets,'train':train},data.tokenizer)


class DeckSampler:
    """Embaralha grupos e exemplos sem reposição dentro de cada ciclo."""
    def __init__(self,data,state=None):
        self.data=data
        self.state=state if state is not None else {'groups':{},'rows':{}}

    def sample(self,pool,size,rng):
        groups=self.data.pools[pool]; names=sorted(groups); out=[]
        deck=self.state['groups'].setdefault(pool,[])
        for _ in range(size):
            if not deck:deck.extend(torch.randperm(len(names),generator=rng).tolist())
            name=names[deck.pop()]
            key=pool+'|'+name
            rows=self.state['rows'].setdefault(key,[])
            choices=groups[name]
            if not rows:rows.extend(torch.randperm(len(choices),generator=rng).tolist())
            out.append(self.data.sets['train'][choices[rows.pop()]])
        return out


def stratified(rows,limit):
    groups=defaultdict(list)
    for row in rows:groups[row['skill']].append(row)
    result=[]; index=0
    while len(result)<limit:
        added=False
        for skill in sorted(groups):
            if index<len(groups[skill]):result.append(groups[skill][index]); added=True
            if len(result)==limit:break
        if not added:break
        index+=1
    return result


def development_view(data):
    """Somente validações antigas; nenhum exemplo do teste é selecionado."""
    sets={'train':data.sets['train'],'validation':data.sets['validation']}
    for key,limit in (('validation_context',30),('validation_natural',24),('validation_imported',60)):
        sets[key]=stratified(data.sets.get(key,[]),limit)
    return ConversationData(sets,data.tokenizer)


def anchor_loss(model,teacher,data,rng,device,batch_size,retention_weight=2.):
    """Ancora cada passo em treino antigo. Não usa validação nem teste."""
    rows=data.sample_pool(batch_size,rng,'legacy')
    x,y=data.batch(rows,model.cfg.context,device)
    logits,ce=model(x,y)
    with torch.no_grad():target,_=teacher(x)
    mask=y!=0
    kl=torch.nn.functional.kl_div(logits[mask].float().log_softmax(-1),target[mask].float().softmax(-1),reduction='batchmean')
    return ce+retention_weight*kl,ce,kl


def weight_delta(model,teacher):
    with torch.no_grad():
        delta=sum(float((a.float()-b.float()).square().sum()) for a,b in zip(model.parameters(),teacher.parameters()))
        base=sum(float(b.float().square().sum()) for b in teacher.parameters())
    return {'l2':delta**.5,'relative_l2':(delta/max(base,1e-30))**.5}


def gradient_interference(model,teacher,data,replay,device,seed=918):
    """Cossenos em embeddings compartilhados; não mede todos os parâmetros."""
    rng=torch.Generator().manual_seed(seed); vectors={}; losses={}
    model.eval()
    for kind in ('imported','reviewed','language','legacy','kl'):
        if kind=='language':x,y=replay.batch(2,min(256,model.cfg.context),rng,device)
        else:
            rows=data.sample_pool(2,rng,'legacy' if kind=='kl' else kind)
            x,y=data.batch(rows,model.cfg.context,device)
        logits,loss=model(x,y)
        if kind=='kl':
            with torch.no_grad():ref,_=teacher(x)
            mask=y!=0
            loss=torch.nn.functional.kl_div(logits[mask].float().log_softmax(-1),ref[mask].float().softmax(-1),reduction='batchmean')
        gradient=torch.autograd.grad(loss,model.tokens.weight)[0].detach().float().flatten()
        vectors[kind]=gradient; losses[kind]=float(loss.detach())
    result={'losses':losses,'embedding_gradient_norms':{k:float(v.norm()) for k,v in vectors.items()},'cosines':{}}
    for a,x in vectors.items():
        for b,y in vectors.items():
            if a<b:
                denom=float(x.norm()*y.norm())
                result['cosines'][a+'__'+b]=float(torch.dot(x,y))/denom if denom>1e-12 else None
    result['scope']='Um lote fixo de 2 exemplos por objetivo; somente gradiente dos embeddings. Cosseno negativo é interferência local, não prova global.'
    return result
