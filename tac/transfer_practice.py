"""Prática autoral opcional; nomes/objetos de treino, sem ler a avaliação reservada."""
import random

from .conversation_lessons import canonical,prompt_text


def practice_rows(seed=302771,episodes=96):
    rng=random.Random(seed);rows=[]
    for i in range(episodes):
        a='Tre'+''.join(rng.choice(('la','vi','na','do','fe','ro','ma','si')) for _ in range(3))
        b='Tre'+''.join(rng.choice(('ze','lo','cu','ra','po','mi','be','to')) for _ in range(3))
        kind=i%4
        if kind==0:
            history=[['user',f'Pode me chamar de {a}.'],['assistant',f'Certo, {a}.'],['user','Como eu disse que me chamo?']]
            answer=f'Seu nome é {a}.'
        elif kind==1:
            history=[['user',f'Meu nome é {a}.'],['assistant',f'Entendido, {a}.'],
                     ['user',f'Não é {a}. Corrija meu nome para {b}.'],['assistant',f'Vou considerar {b}.'],['user','Qual é meu nome depois da correção?']]
            answer=f'Seu nome é {b}.'
        elif kind==2:
            history=[['user',f'Não coloquei a chave em {a.lower()}. A chave ficou em {b.lower()}.'],
                     ['assistant','Entendi.'],['user','Onde a chave está?']]
            answer=f'A chave está em {b.lower()}.'
        else:
            history=[['user',f'O caderno estava em {a.lower()}.'],['assistant','Certo.'],
                     ['user',f'Levei o caderno para {b.lower()}.'],['assistant','Entendi a mudança.'],['user','Onde ficou o caderno?']]
            answer=f'O caderno está em {b.lower()}.'
        rows.append({'messages':history,'answer':answer,'skill':f'practice_transfer_{i%24}',
            'mixture':'reviewed','authored_transfer_practice':True})
    return rows


def augment(data):
    from .neural.conversation import ConversationData
    forbidden={canonical(prompt_text(r['messages'])) for key,rows in data.sets.items() if key!='train' for r in rows}
    new=[r for r in practice_rows() if canonical(prompt_text(r['messages'])) not in forbidden]
    return ConversationData({**data.sets,'train':[*data.sets['train'],*new]},data.tokenizer)
