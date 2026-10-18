"""Dados públicos da TAC v2: filtros locais, proveniência e árvores de conversa."""
from __future__ import annotations
import hashlib
import json
import re
import unicodedata
from array import array
from collections import Counter, defaultdict


def canonical(text):
    return ' '.join(unicodedata.normalize('NFC',text).casefold().split())


def fingerprint(text):return hashlib.sha256(canonical(text).encode()).hexdigest()


def clean(text):
    text=unicodedata.normalize('NFC',text or '').replace('\r\n','\n').replace('\r','\n')
    text=re.sub(r'[\t\u00a0\u200b\ufeff]+',' ',text)
    text=re.sub(r' +',' ',text)
    return re.sub(r'\n{4,}','\n\n\n',text).strip()


PRIVATE=[
    ('email',re.compile(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}')),
    ('cpf_cnpj',re.compile(r'\b(?:\d{3}[. -]?\d{3}[. -]?\d{3}[- ]?\d{2}|\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2})\b')),
    ('contact_number',re.compile(r'(?<!\w)(?:\+\d{1,3}[ -]?)?(?:\(\d{2,3}\)[ -]?)?\d{4,5}[ -]\d{4}(?!\w)')),
    ('personal_address',re.compile(r'\b(?:moro|resido|meu endereço|minha morada)\b.{0,80}\b(?:rua|avenida|travessa|cep|n[º°])\b',re.I)),
    ('personal_full_name',re.compile(r'\b(?:meu nome completo é|me chamo)\s+[A-ZÀ-Ý][a-zà-ÿ]+(?:\s+[A-ZÀ-Ý][a-zà-ÿ]+){1,4}')),
    ('secret',re.compile(r'\b(?:sk-[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|-----BEGIN [A-Z ]*PRIVATE KEY)')),
]
PT=set('de a o e que do da em um para com não uma os no se na por mais as dos como mas foi ao das tem seu sua é são também quando onde qual pelo pela você isso'.split())


def problem(text, *, document=False):
    if not text:return 'empty'
    if len(text)>(40000 if document else 16000):return 'too_long'
    if len(text)<(350 if document else 8):return 'too_short'
    if '\ufffd' in text or '\x00' in text:return 'broken_unicode'
    if len(re.findall(r'Ã[§£©¡³µºª]|â€|ðŸ',text))>=2:return 'mojibake'
    if any(ord(ch)<32 and ch not in '\n\t' for ch in text):return 'control_characters'
    for name,pattern in PRIVATE:
        if pattern.search(text):return name
    if re.search(r'ignore (?:all )?(?:previous|prior) instructions|ignore todas as instruções|desconsidere (?:todas )?as instruções anteriores',text,re.I):return 'instruction_injection'
    if re.search(r'\b(?:como estuprar|pornografia infantil|matar todos os|extermine todos os)\b',text,re.I):return 'extreme_abuse'
    if re.search(r'(.)\1{20,}',text):return 'repeated_characters'
    words=re.findall(r'\w+',text.casefold())
    if not words:return 'no_words'
    if document:
        if sum(c.isalpha() for c in text)/len(text)<.5:return 'low_alphabetic'
        if sum(w in PT for w in words)/len(words)<.025:return 'weak_portuguese_heuristic'
        if len(set(words))/len(words)<.14:return 'low_word_diversity'
        if max(Counter(words).values())/len(words)>.18:return 'repetitive_word'
        lines=[canonical(x) for x in text.splitlines() if len(x.strip())>10]
        if len(lines)>8 and 1-len(set(lines))/len(lines)>.3:return 'repeated_lines'
        if len(re.findall(r'<\/?[A-Za-z][^>]{0,100}>',text))>3:return 'html_residue'
        if sum(x in canonical(text) for x in ('aceitar cookies','política de cookies','todos os direitos reservados','carrinho de compras'))>=2:return 'navigation_or_commerce'
    return None


class NearDedup:
    """SimHash candidatos + Jaccard de 5-gramas; guarda somente fingerprints."""
    def __init__(self):self.buckets=defaultdict(list);self.rows=[];self.exact=set()
    def duplicate(self,text):
        norm=canonical(text);exact=hashlib.sha256(norm.encode()).digest()
        if exact in self.exact:return True
        words=re.findall(r'\w+',norm)
        grams={int.from_bytes(hashlib.blake2b(' '.join(words[i:i+5]).encode(),digest_size=8).digest(),'little') for i in range(max(0,len(words)-4))}
        # Amostra determinística para custo limitado; não certifica toda paráfrase.
        selected=sorted(grams,key=lambda n:n.to_bytes(8,'little'))[:128]
        sums=[0]*64
        for gram in selected:
            n=gram
            for b in range(64):sums[b]+=1 if (n>>b)&1 else -1
        signature=sum((v>=0)<<i for i,v in enumerate(sums));candidates=set()
        for band in range(4):candidates.update(self.buckets[(band,(signature>>(16*band))&65535)])
        for i in candidates:
            other,othergrams=self.rows[i]
            if (signature^other).bit_count()>3:continue
            intersection=len(grams.intersection(othergrams))
            if intersection/max(1,len(grams)+len(othergrams)-intersection)>=.85:return True
        i=len(self.rows);self.rows.append((signature,array('Q',sorted(grams))));self.exact.add(exact)
        for band in range(4):self.buckets[(band,(signature>>(16*band))&65535)].append(i)
        return False


def split_group(group):
    return 'validation' if int(hashlib.sha256(('TAC-v2-split-20261017:'+group).encode()).hexdigest()[:8],16)%100<3 else 'train'


def label_values(row):
    value=row.get('labels') or {}
    if isinstance(value,dict) and 'name' in value:
        return dict(zip(value['name'],value['value']))
    if isinstance(value,list):return {x['name']:x['value'] for x in value if isinstance(x,dict)}
    if isinstance(value,dict):return {k:v.get('value',0) if isinstance(v,dict) else v for k,v in value.items()}
    return {}


def oasst_examples(rows,counters):
    """Um caminho completo por resposta preferida; sem combinar irmãos."""
    lookup={r['message_id']:r for r in rows};best={}
    for row in rows:
        if row['role']!='assistant':continue
        rank=row.get('rank');rank=10**6 if rank is None else rank
        key=row.get('parent_id');old=best.get(key)
        if old is None or (rank,row['message_id'])<(old[0],old[1]['message_id']):best[key]=(rank,row)
    for _,row in best.values():
        path=[];seen=set();cursor=row;reason=None
        while cursor is not None:
            ident=cursor['message_id']
            if ident in seen:reason='cycle';break
            seen.add(ident);path.append(cursor)
            if cursor.get('lang') not in ('pt','pt-BR','pt-PT'):reason='non_pt_ancestor';break
            labels=label_values(cursor)
            if cursor.get('deleted') or cursor.get('synthetic') or cursor.get('review_result') is False:reason='rejected_or_synthetic';break
            if any(float(labels.get(k,0) or 0)>.1 for k in ('pii','spam','lang_mismatch','not_appropriate')):reason='annotation_flag';break
            if cursor['role']=='assistant' and 'quality' in labels and float(labels['quality'])<.4:reason='low_annotated_quality';break
            text=clean(cursor.get('text'));reason=problem(text)
            if reason:break
            parent=cursor.get('parent_id')
            if parent and parent not in lookup:reason='missing_ancestor';break
            cursor=lookup[parent] if parent else None
        if reason:counters['oasst_'+reason]+=1;continue
        path.reverse()
        if not path or path[0].get('parent_id') or any(r['role']!=('prompter' if i%2==0 else 'assistant') for i,r in enumerate(path)):
            counters['oasst_invalid_alternation']+=1;continue
        if len(path)<2 or len(path)%2:continue
        messages=[['user' if r['role']=='prompter' else 'assistant',clean(r['text'])] for r in path[:-1]]
        yield {'messages':messages,'answer':clean(row['text']),'source':'OpenAssistant/oasst2','source_id':row['message_id'],
               'group':'oasst:'+str(row.get('message_tree_id') or path[0]['message_id']),
               'ancestor_ids':[r['message_id'] for r in path[:-1]],'rank':row.get('rank'),
               'skill':'oasst_multiturn' if len(messages)>1 else 'oasst_instruction'}


def prompt_text(messages):
    if not messages or messages[-1][0]!='user':raise ValueError('Histórico precisa terminar no usuário')
    return '\n'.join(('Usuário: ' if r=='user' else 'TAC: ')+' '.join(t.split()) for r,t in messages)+'\nTAC:'


def load_jsonl(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():yield json.loads(line)
