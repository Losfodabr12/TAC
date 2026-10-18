"""Diálogos autorais/sintéticos, estilos e correções locais (sem coleta externa).

Os formatos reservados são distintos dos de treino. O snapshot guarda todos os
exemplos concretos e os exemplos reservados não participam da amostragem.
"""
from __future__ import annotations

import random
import json
from pathlib import Path

from .conversation_lessons import canonical, prompt_text

VERSION = "TAC-natural-dialogue-v1"

FORMS = {
    "train": {
        "intro": ["{initial}", "ei, {initial}", "tac, {initial}", "olha, {initial} 🙂"],
        "color_change": ["Agora prefiro {new}.", "na real eu gosto mais de {new}", "mudei de ideia, agora é {new} 😊", "troca minha cor favorita pra {new}", "Não é mais {old}; agora prefiro {new}.", "Minha preferida é {new}, deixei {old} pra trás.", "corrigindo: a cor é {new}", "foi mal kkk, anota {new} no lugar de {old}", "Eu escolho {new}. {old} era a escolha antiga.", "pensando melhor, fico com {new}", "tô preferindo {new} hj", "na vdd minha favorita mudou pra {new}"],
        "name_change": ["Me chama de {new} agora.", "corrigindo meu nome: {new}", "na verdade eu me chamo {new}", "não é {old}, meu nome é {new}", "Pode usar {new}; {old} estava errado.", "troca o nome pra {new}, por favor", "foi mal, escrevi errado, sou {new} 😅", "o nome certo é {new}"],
        "topic_change": ["Vamos falar de {new} agora.", "na real, prefiro conversar sobre {new}", "mudei de assunto, quero falar de {new}", "troca o assunto pra {new}", "Não quero {old}; quero falar de {new}.", "Bora conversar sobre {new} kkk", "deixa {old} pra depois, vamos de {new}"],
        "question": ["{question}", "tac, {question}", "me lembra: {question}", "e então, {question}"],
    },
    "validation": {
        "intro": ["guarda isso pra nossa conversa: {initial}"],
        "color_change": ["pensando de novo, minha preferência passou a ser {new}.", "atualiza aí: eu escolhi {new}, esquece a escolha {old}."],
        "name_change": ["quero corrigir como me apresentei: pode usar {new}.", "meu nome informado estava errado; considere {new}."],
        "topic_change": ["vamos mudar o rumo da conversa para {new}.", "resolvi conversar de {new} em vez de {old}."],
        "question": ["só para confirmar, {question}"],
    },
    "test": {
        "intro": ["pra você se situar: {initial}"],
        "color_change": ["reconsiderei aquela preferência: fico com {new}.", "a escolha atual é {new}; a escolha {old} ficou para trás."],
        "name_change": ["minha apresentação precisa de um ajuste: o nome correto é {new}.", "a partir desta fala, me trate por {new}."],
        "topic_change": ["o assunto que quero daqui pra frente é {new}.", "vamos deixar {old} e seguir a conversa com {new}."],
        "question": ["com o que conversamos até aqui, {question}"],
    },
}

SLOTS = {
    "color": {"train": "azul verde amarelo branco preto rosa roxo cinza laranja marrom".split(),
              "validation": ["cobre", "anil", "creme"], "test": ["salmão", "lilás", "grafite"],
              "initial": "Minha cor favorita é {value}.", "ack": "Sua cor favorita é {value}.",
              "question": "qual é minha cor favorita?", "changed": "Agora sua cor favorita é {value}.",
              "recall": "Agora sua cor favorita é {value}."},
    "name": {"train": "Ana Lucas Carla Diego Maria Bruno Rita Sofia Paulo Lara Caio Bia Hugo Vera".split(),
             "validation": ["Ciro", "Cora", "Edna"], "test": ["Ivo", "Mara", "Hélio"],
             "initial": "Meu nome é {value}.", "ack": "Prazer, {value}.",
             "question": "qual é meu nome?", "changed": "Certo, vou chamar você de {value}.",
             "recall": "Você me disse que seu nome é {value}."},
    "topic": {"train": ["jogos", "histórias", "Python", "desenho", "música", "livros", "Roblox", "filmes"],
              "validation": ["animação", "pintura", "xadrez"], "test": ["quadrinhos", "astronomia", "fotografia"],
              "initial": "Quero conversar sobre {value}.", "ack": "Vamos conversar sobre {value}.",
              "question": "qual é o assunto de agora?", "changed": "Certo, vamos falar de {value}.",
              "recall": "Agora estamos falando de {value}."},
}

# Cada cena tem falas em três registros e respostas escritas para este projeto.
# Nenhuma cena é uma conversa humana coletada.
SCENES = [
    (["eae cheguei pra trocar ideia 😄", "olá, gostaria de conversar um pouco", "oieee bora papear kkk"],
     ["Oi! Como foi seu dia?", "Oi! Sobre o que você quer conversar?"],
     ["terminei meu desenho hj", "hoje consegui concluir um desenho", "acabei o desenho finalmente 😅"],
     ["Parabéns pelo desenho! Qual parte você mais gostou de fazer?", "Que bom! O que você desenhou?"], "natural_achievement"),
    (["to meio pra baixo hj", "hoje estou desanimado", "tô tristão 😕"],
     ["Poxa. Quer me contar o que aconteceu?", "Quer conversar sobre o que deixou seu dia difícil?"],
     ["tive um dia ruim na escola", "o dia na escola foi bem difícil", "a escola hoje me deixou mal"],
     ["O que aconteceu na escola?", "Qual parte do dia na escola foi mais difícil?"], "natural_empathy"),
    (["meu jogo tá dando erro", "estou com um problema no jogo", "esse jogo bugou de novo 😤"],
     ["Em que momento o erro aparece?", "O que acontece quando o jogo dá erro?"],
     ["ele fecha quando eu entro no mapa", "ao abrir o mapa, o jogo encerra", "entro no mapa e o jogo fecha sozinho"],
     ["Aparece alguma mensagem antes de fechar?", "Isso acontece em todos os mapas ou só nesse?"], "natural_help"),
    (["to escrevendo uma história", "estou criando uma narrativa", "bora falar da minha história kk"],
     ["Quem é o personagem principal?", "Sobre o que é a história?"],
     ["é sobre uma pessoa procurando o irmão", "o protagonista está procurando o irmão", "o personagem quer achar o irmão perdido"],
     ["O que levou o irmão a desaparecer?", "Que pista o protagonista encontra primeiro?"], "natural_writing"),
    (["comecei a aprender python", "estou começando em Python", "to estudando python agr 😄"],
     ["Qual parte de Python você está praticando?", "Você já tentou escrever algum programa pequeno?"],
     ["tô vendo laços for e while", "estou praticando for e while", "agora cheguei nos loops for e while"],
     ["Você quer praticar um exemplo com for ou com while?", "Qual dos dois laços ficou mais difícil de entender?"], "natural_help"),
    (["nem sei do que falar kkk", "estou sem assunto para conversar", "to sem assunto 😅"],
     ["Podemos falar de jogos, histórias ou do seu dia. Qual você prefere?", "O que você gosta de fazer no tempo livre?"],
     ["eu gosto de desenhar", "costumo desenhar nas horas livres", "curto fazer uns desenhos"],
     ["Que tipo de desenho você gosta de fazer?", "Você está trabalhando em algum desenho agora?"], "natural_hobby"),
    (["to cansado demais hoje", "hoje fiquei bastante cansado", "hj tô só o cansaço 😴"],
     ["Quer conversar sobre como foi seu dia?", "O que mais te cansou hoje?"],
     ["fiquei o dia todo trabalhando num projeto", "passei o dia trabalhando em um projeto", "trabalhei no projeto sem parar"],
     ["Qual parte do projeto tomou mais tempo?", "Você conseguiu concluir alguma etapa do projeto?"], "natural_empathy"),
    (["consegui fazer meu programa funcionar", "finalmente consegui executar meu programa", "deu certooo meu código rodou 😄"],
     ["Que bom! O que o programa faz?", "Parabéns! Qual era a dificuldade?"],
     ["é uma calculadora simples", "eu fiz uma calculadora básica", "uma calculadorazinha kkk"],
     ["Quais operações sua calculadora já faz?", "Qual operação você quer acrescentar depois?"], "natural_achievement"),
    (["nao entendi sua resposta", "sua explicação ficou confusa", "boiei nessa explicação 😅"],
     ["Qual parte ficou confusa?", "Quer que eu explique com um exemplo?"],
     ["quero um exemplo mais simples", "um exemplo curto ajudaria", "manda um exemplinho fácil"],
     ["Sobre qual assunto você quer esse exemplo?", "Qual assunto devo usar no exemplo?"], "natural_clarify"),
]

SCENE_HELDOUT = {
    "validation": [
        ("Resolvi vir conversar contigo.", "Consegui terminar uma ilustração que estava fazendo."),
        ("O dia de hoje me deixou sem ânimo.", "Uma coisa na escola me deixou chateado."),
        ("Preciso de ajuda com uma falha num jogo.", "O jogo encerra bem na hora de carregar o mapa."),
        ("Estou montando o enredo de uma narrativa.", "A pessoa principal da história quer encontrar o irmão desaparecido."),
        ("Comecei meus estudos na linguagem Python.", "Minha dúvida agora é sobre repetições com for e while."),
        ("Não tenho nenhum assunto em mente.", "Nas horas vagas gosto bastante de desenhar."),
        ("Hoje gastei todas as minhas energias.", "Fiquei ocupado com o mesmo projeto o dia inteiro."),
        ("Meu código passou a funcionar finalmente.", "O programa que criei é uma calculadora bem básica."),
        ("Pode esclarecer o que acabou de dizer?", "Acho que entenderia melhor com um exemplo pequeno."),
    ],
    "test": [
        ("Vim aqui para bater um papo contigo.", "Aquele desenho em que eu estava trabalhando ficou pronto."),
        ("Hoje o desânimo bateu por aqui.", "Saí da escola triste com o que rolou por lá."),
        ("Tem um erro atrapalhando minha partida.", "Assim que carrega o mapa, o jogo se fecha."),
        ("Tô inventando uma nova aventura para escrever.", "Meu protagonista procura o irmão que sumiu."),
        ("Decidi estudar programação usando Python.", "Cheguei no for e no while e queria praticar esses laços."),
        ("Quero papear mas não sei sobre o quê.", "Meu passatempo preferido é fazer desenhos."),
        ("Terminei o dia muito cansado mesmo.", "Dediquei o dia inteiro a um projeto que estou fazendo."),
        ("Consegui corrigir meu programa e ele rodou.", "Ele serve para fazer contas, é uma calculadora simples."),
        ("Me perdi na sua explicação anterior.", "Me ajudaria ver um exemplo bem curtinho."),
    ],
}


# Ampliação de treino após a primeira avaliação de desenvolvimento. O arquivo
# natural_holdout.json permanece congelado; nenhuma dessas frases o substitui.
TRAIN_EXPANSION = {
    'intro': ['Uma informação: {initial}', 'Anota aí: {initial}',
              'Durante esta conversa: {initial}', 'Olha só, {initial}',
              'Pra registrar: {initial}', 'Lembre desta informação: {initial}'],
    'color_change': [
        'Hoje minha preferência é {new}.', 'Minha preferência mudou para {new}.',
        'Gostava de {old}; passei a preferir {new}.', 'Eu escolhi {new} como minha cor favorita.',
        'Agora minha preferência é {new}, não {old}.', 'A escolha {old} foi alterada para {new}.',
        'Confere a anotação: minha preferência passou a {new}.',
        'Minha cor passou a ser {new} desde ontem.', 'Escolhi {new}; pode esquecer {old}.',
        'Pode atualizar: de todas as cores, gosto de {new}.',
        'Fiz outra escolha de cor: {new}.', 'Sobre minha preferência, considere {new}.',
        'A cor que eu quero usar daqui em diante é {new}.',
        'Reconsiderei a cor que tinha escolhido e agora quero {new}.',
    ],
    'name_change': [
        'Meu cadastro estava errado, sou {new}.', 'Para corrigir: considere {new} como meu nome.',
        'Acabo de me apresentar errado, me chamo {new}.', 'Use o nome {new} daqui em diante.',
        'Minha identificação correta é {new}.', 'Eu escrevi {old}, mas o certo seria {new}.',
        'A apresentação anterior estava errada; na realidade sou {new}.',
        'Ajustando o que disse: meu nome é {new}.', 'O nome passou a ser {new}.',
        'Preciso corrigir o nome informado para {new}.', 'Meu nome de preferência é {new}.',
    ],
    'topic_change': [
        'Quero levar nossa conversa para {new}.', 'Resolvi trocar {old} por {new} nesta conversa.',
        'O rumo mudou: agora falaremos de {new}.', 'Vamos conversar de {new}; não quero mais {old}.',
        'Minha escolha de assunto passou a ser {new}.', 'Podemos mudar o tema para {new}?',
        'Que tal conversarmos de {new} no lugar de {old}?', 'O próximo assunto será {new}.',
        'Agora meu interesse está em {new}.', 'Vamos deixar o tema antigo e conversar de {new}.',
        'A partir de agora o tema da conversa é {new}.', 'Prefiro o assunto {new} para este papo.',
        'Escolhi falar de {new}; esquece aquele assunto {old}.',
        'Resolvi mudar de tema e conversar sobre {new}.',
    ],
}
COMMON_CHANGES = ['Corrigindo: {new}.', 'A opção certa agora é {new}.',
                  'Mudei de ideia, vou ficar com {new}.', 'Atualização: {new}, em vez de {old}.',
                  'Não vale mais {old}. Considere {new}.', 'Pensando melhor: {new}.']


def make_natural_sets(fragments=()):
    sets = {split: [] for split in ("train", "validation", "test")}
    rng = random.Random(601733)
    reserved = {canonical(v) for slot in SLOTS.values() for split in ("validation", "test") for v in slot[split]}
    reserved.update("rafael marina gabriel camila noemi isadora renata tiago valentina murilo rebeca samuel lorena vicente amanda enzo turquesa violeta bege dourado prateado vermelho".split())
    pseudo = set()
    for _ in range(1300):
        if fragments and rng.random() < .5:
            value = "".join(rng.choice(fragments) for _ in range(rng.randint(1, 3))).lower()
        else:
            value = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 9)))
        if value.isalpha() and len(value) <= 18 and canonical(value) not in reserved:
            pseudo.add(value)
    pseudo = sorted(pseudo)

    def add(split, messages, answers, skill):
        if isinstance(answers, str):
            answers = [answers]
        row = {"messages": messages, "answer": answers[0], "skill": skill}
        if len(answers) > 1:
            row["acceptable_answers"] = answers
        sets[split].append(row)

    for split in sets:
        rng = random.Random(601733 + {"train": 0, "validation": 100, "test": 200}[split])
        forms = {key: list(value) for key, value in FORMS[split].items()}
        if split == 'train':
            for key, extra in TRAIN_EXPANSION.items():
                forms[key] += extra
            for slot_name in SLOTS:
                forms[slot_name + '_change'] += COMMON_CHANGES
        for slot_name, spec in SLOTS.items():
            for index in range(4000 if split == "train" else 9):
                values = spec[split]
                if split == "train" and index % 3 == 0:
                    values = pseudo
                old, new = rng.sample(values, 2)
                if slot_name == "name":
                    old, new = old.capitalize(), new.capitalize()
                initial = spec["initial"].format(value=old)
                first = rng.choice(forms["intro"]).format(initial=initial)
                ack = spec["ack"].format(value=old)
                history_ack = ack
                correction = rng.choice(forms[slot_name + "_change"]).format(old=old, new=new)
                if split == 'train' and index % 4 == 0:
                    correction += rng.choice([' Certo?', ' por enquanto.', ' por favor.', ' beleza?'])
                if split == 'train' and index % 5 == 0:
                    history_ack = rng.choice(['Entendido.', 'Anotei.', spec['changed'].format(value=old)])
                changed = spec["changed"].format(value=new)
                question = rng.choice(forms["question"]).format(question=spec["question"])
                add(split, [["user", first]], ack, "natural_" + slot_name + "_intro")
                prefix = [["user", first], ["assistant", history_ack], ["user", correction]]
                add(split, prefix, changed, "natural_" + slot_name + "_correction")
                add(split, [*prefix, ["assistant", changed], ["user", question]],
                    spec["recall"].format(value=new), "natural_" + slot_name + "_recall")
                if split == "train":
                    # A rede deve recuperar a correção mesmo se a própria resposta
                    # anterior confirmou o valor antigo incorretamente.
                    add(split, [*prefix, ["assistant", ack], ["user", question]],
                        spec["recall"].format(value=new), "natural_" + slot_name + "_recall")
        # Registros alternados, gírias/abreviações, emojis e falas com typos.
        for scene_index, (firsts, replies, seconds, answers, skill) in enumerate(SCENES):
            if split == "train":
                for first in firsts:
                    for reply in replies:
                        add(split, [["user", first]], replies, skill)
                        for second in seconds:
                            for prefix in ("", "tac, ", "então, "):
                                add(split, [["user", first], ["assistant", reply], ["user", prefix + second]], answers, skill)
                # Todas as respostas aceitáveis têm exemplos supervisionados,
                # com entradas diferentes, sem duplicar a mesma entrada/rótulo.
                for i in range(6):
                    first, second = firsts[i % 3], seconds[(i // 2) % 3]
                    for style in ("olha, ", "cara, ", "amiga, ", "ei, "):
                        add(split, [["user", style + first], ["assistant", replies[i % len(replies)]], ["user", second]],
                            [answers[i % len(answers)], *[a for a in answers if a != answers[i % len(answers)]]], skill)
            else:
                # Paráfrases reservadas não entram no treino.
                first, second = SCENE_HELDOUT[split][scene_index]
                for reply in replies:
                    add(split, [["user", first], ["assistant", reply], ["user", second]], answers, skill)

    frozen_path = Path(__file__).with_name("natural_holdout.json")
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if frozen.get("format") != VERSION:
            raise ValueError("Conjunto reservado de diálogos incompatível")
        sets['validation'], sets['test'] = frozen['validation'], frozen['test']
    # Os conjuntos reservados são fixos e precedem a deduplicação.
    seen = set()
    for split in ("validation", "test", "train"):
        clean = []
        for row in sets[split]:
            key = canonical(prompt_text(row["messages"]))
            if key not in seen:
                seen.add(key)
                clean.append(row)
        sets[split] = clean
    return sets
