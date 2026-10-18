"""Componentes experimentais isolados. Não alteram as políticas do chat/treinador."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter,defaultdict

import torch

from ..conversation_lessons import prompt_text
from .research import precise_prompt


def plain(text):
    return ' '.join(''.join(c for c in unicodedata.normalize('NFKD',text.casefold()) if not unicodedata.combining(c)).split())


def category(row):
    """Categorias auditáveis por regras; negação TEXTUAL não equivale a lógica."""
    question=plain(row['messages'][-1][1]);skill=row['skill']
    if skill=='correcao' or re.search(r'\b(errei|corrig\w*|troca\w*|mudei|enganei|na verdade)\b',question):return 'correcoes'
    if re.search(r'\b(meu nome|me chamo|me chamar|lembra.*nome|sou [oa] \w+)\b',question):return 'memoria_nomes'
    if re.search(r'\bnao\b',question):return 'negacao_textual'
    if skill in ('conhecimento','contexto_capitais','contexto_contas') or re.search(r'\b(capital|quanto e)\b',question):return 'conhecimento_factual'
    if len(row['messages'])>1 and (skill in ('memoria_dados','memoria_conversa') or re.search(r'\b(onde|lembra|antes|assunto|moro|cachorro|favorita)\b',question)):return 'contexto'
    return 'conversa_geral'


def row_key(row):
    raw=json.dumps([row['messages'],row['answer']],ensure_ascii=False,separators=(',',':'))
    return hashlib.sha256(raw.encode()).hexdigest()


class Exposure:
    def __init__(self):
        self.draws=Counter();self.tokens=Counter();self.by_skill=Counter();self.tokens_skill=Counter()
        self.seen=defaultdict(Counter);self.groups=defaultdict(set);self.steps=0

    def add(self,pool,rows,y):
        for i,row in enumerate(rows):
            cat=category(row) if pool=='imported' else row.get('category',row['skill'])
            key=pool+'/'+cat;count=int((y[i]!=0).sum());self.draws[key]+=1;self.tokens[key]+=count
            self.by_skill[pool+'/'+row['skill']]+=1;self.tokens_skill[pool+'/'+row['skill']]+=count
            self.seen[pool][row_key(row)]+=1
            if row.get('group'):self.groups[pool].add(row['group'])

    def report(self):
        return {'draws_by_category':dict(self.draws),'supervised_tokens_by_category':dict(self.tokens),
            'draws_by_original_skill':dict(self.by_skill),'supervised_tokens_by_original_skill':dict(self.tokens_skill),
            'pools':{p:{'draws':sum(v.values()),'unique_examples':len(v),'repeated_draws':sum(v.values())-len(v),
                'max_draws_same_example':max(v.values(),default=0),'unique_groups':len(self.groups[p])} for p,v in self.seen.items()}}


class BalancedImports:
    """Ciclo de categorias/grupos, sem repetir linhas até esgotar todo o pool."""
    def __init__(self,rows,rng):
        self.rows=rows;self.rng=rng;self.buckets=defaultdict(lambda:defaultdict(list));self.group_visits=Counter()
        for i,row in enumerate(rows):self.buckets[category(row)][row['group']].append(i)
        for groups in self.buckets.values():
            for group,indices in groups.items():groups[group]=[indices[i] for i in torch.randperm(len(indices),generator=rng).tolist()]
        self.cycle=[];self.used=set();self.cycles=0

    def sample(self,size):
        out=[]
        for _ in range(size):
            available={c:[g for g,ids in groups.items() if ids] for c,groups in self.buckets.items()}
            available={c:g for c,g in available.items() if g}
            if not available:raise ValueError('Todos os exemplos importados já foram usados; não repetir conjunto pequeno neste piloto.')
            self.cycle=[c for c in self.cycle if c in available]
            if not self.cycle:
                names=sorted(available);self.cycle=[names[i] for i in torch.randperm(len(names),generator=self.rng).tolist()];self.cycles+=1
            cat=self.cycle.pop();groups=available[cat]
            least=min(self.group_visits[g] for g in groups);groups=[g for g in groups if self.group_visits[g]==least]
            group=groups[int(torch.randint(len(groups),(),generator=self.rng))]
            idx=self.buckets[cat][group].pop();assert idx not in self.used
            self.used.add(idx);self.group_visits[group]+=1;out.append(self.rows[idx])
        return out


class ResponseObjective:
    """Mesmo x textual. Máscara de usuário/papéis; EOS nas fronteiras de resposta."""
    def __init__(self,replay,scenes,tokenizer):
        self.replay=replay;self.targets=[];self.boundaries=0;self.changed_rows=0
        mapping={scene:turns for scene,turns,*_ in scenes}
        for index,tokens in enumerate(replay.rows):
            y=tokens[1:].clone();meta=replay.provenance[index]
            if 'authored_scene' in meta:
                turns=mapping[meta['authored_scene']]
                text='\n'.join('Usuário: '+u+'\nTAC: '+a for u,a in turns)
                if tokenizer.encode(text)!=tokens.tolist():raise ValueError('Cena não corresponde ao snapshot textual congelado.')
                y.zero_();cursor=0
                for turn,(user,answer) in enumerate(turns):
                    prefix=('\n' if turn else '')+'Usuário: '+user+'\nTAC: '
                    start=cursor+len(prefix);end=start+len(answer)
                    a=1+len(tokenizer.encode(text[:start],special=False));b=1+len(tokenizer.encode(text[:end],special=False))
                    y[a-1:b-1]=tokens[a:b];y[b-1]=tokenizer.eos;cursor=end
                    self.boundaries+=1
                self.changed_rows+=1
            self.targets.append(y)

    def batch(self,descriptors,context=256,variant=False):
        x=torch.zeros((len(descriptors),context),dtype=torch.long);y=torch.zeros_like(x);rows=[]
        for i,(index,start) in enumerate(descriptors):
            seq=self.replay.rows[index][start:start+context+1];n=len(seq)-1
            x[i,:n]=seq[:-1]
            y[i,:n]=self.targets[index][start:start+n] if variant else seq[1:]
            meta=self.replay.provenance[index];kind='dialogo_autoral' if 'authored_scene' in meta else 'narrativa_autoral'
            rows.append({'messages':[['user',str(index)+':'+str(start)]],'answer':str(index),'skill':kind,'category':kind})
        if not (y!=0).any(dim=1).all():raise ValueError('Janela sem tokens supervisionados.')
        return x,y,rows


class MatchedRetention:
    """Mesma habilidade e quantidade de tokens-alvo por exemplo do controle."""
    def __init__(self,rows,tokenizer,retained):
        self.rows=rows;self.buckets=defaultdict(list);self.families=defaultdict(lambda:defaultdict(list))
        for row in rows:
            length=len(tokenizer.encode(' '+row['answer'],special=False))+1;key=(row['skill'],length)
            self.buckets[key].append(row)
            neg=any(re.search(r'\bnao\b',plain(t)) for role,t in row['messages'] if role=='user')
            family=(len(row['messages']),neg,row_key(row) in retained)
            self.families[key][family].append(row)
        self.decks={};self.family_decks={}

    def sample(self,control,tokenizer,rng,representative=False):
        out=[]
        for row in control:
            key=row['skill'],len(tokenizer.encode(' '+row['answer'],special=False))+1
            if not representative:
                choices=self.buckets[key];out.append(choices[int(torch.randint(len(choices),(),generator=rng))]);continue
            families=self.families[key];deck=self.family_decks.setdefault(key,[])
            if not deck:
                names=sorted(families);deck.extend(names[i] for i in torch.randperm(len(names),generator=rng).tolist())
            family=deck.pop();choices=families[family];rows=self.decks.setdefault((key,family),[])
            if not rows:rows.extend(torch.randperm(len(choices),generator=rng).tolist())
            out.append(choices[rows.pop()])
        return out


def tensor_hash(x,y):
    h=hashlib.sha256();h.update(x.cpu().numpy().tobytes());h.update(y.cpu().numpy().tobytes());return h.hexdigest()


def vector(model,grad=False,device='cpu'):
    return torch.cat([(p.grad if grad else p).detach().float().reshape(-1).to(device) for p in model.parameters()])


def relation(a,b):
    # Entradas FP32: não duplicar dois vetores de 10 milhões de posições em FP64.
    # O arredondamento da redução FP32 é suficiente para esta sonda direcional.
    a=a.float();b=b.float()
    # Mesma redução para norma e produto: evita cosseno > 1 por diferenças
    # de arredondamento entre norm() e dot() em vetores extensos.
    na=max(0.,float(torch.dot(a,a)))**.5;nb=max(0.,float(torch.dot(b,b)))**.5
    dot=float(torch.dot(a,b));cosine=max(-1.,min(1.,dot/(na*nb))) if na*nb>1e-18 else None
    return {'dot':dot,'cosine':cosine,'norm_a':na,'norm_b':nb}


def diagnostic_gradients(model,teacher,batches,device,storage_device='cpu'):
    named=list(model.parameters());vectors={};losses={}
    for pool,(x,y) in batches.items():
        logits,loss=model(x.to(device),y.to(device))
        if pool=='kl':
            with torch.no_grad():ref,_=teacher(x.to(device))
            mask=y.to(device)!=0
            loss=torch.nn.functional.kl_div(logits[mask].float().log_softmax(-1),ref[mask].float().softmax(-1),reduction='batchmean')
        gs=torch.autograd.grad(loss,named);vectors[pool]=torch.cat([g.detach().reshape(-1).to(storage_device) for g in gs]);losses[pool]=float(loss.detach())
    return vectors,losses


def vector_hash(tensor):
    """Hash sem criar cópia de 40 MiB de um vetor CUDA na RAM."""
    h=hashlib.sha256()
    for block in tensor.split(262144):h.update(block.cpu().numpy().tobytes())
    return h.hexdigest()
