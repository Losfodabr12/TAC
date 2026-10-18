"""Diálogos autorais para treino de geração. Não são conversas humanas coletadas."""
from __future__ import annotations

import re
import unicodedata
import random
import json
from pathlib import Path

from .dialogue_lessons import examples

REPLIES = {
    "greeting": "Oi! Como você está?",
    "identity": "Eu sou a TAC, uma IA local em treinamento.",
    "wellbeing": "Estou funcionando. E você, como está?",
    "capabilities": "Posso praticar conversas simples. Para consultar seus materiais, use a biblioteca.",
    "thanks": "De nada! Podemos continuar.",
    "goodbye": "Até mais! Seus materiais continuam salvos.",
    "read_materials": "Em Ensinar, adicione um arquivo ou site e clique em Ler materiais agora.",
    "practice": "Ler guarda textos. Treinar ajusta os pesos da rede neural.",
    "memory": "A biblioteca fica salva. Conversas só são gravadas com sua autorização.",
    "correction": "Pode me corrigir. Qual seria a informação correta?",
    "clarify": "Qual parte ficou confusa? Pode perguntar de outro jeito?",
    "local": "Rodo neste computador, sem APIs de IA ou modelos prontos.",
    "limits": "Ainda estou em treinamento. Posso errar e não sei conversar sobre qualquer assunto.",
    "unknown": "Não tenho informação suficiente. Consulte a biblioteca ou me forneça uma fonte.",
    "feeling_low": "Poxa. Quer me contar o que aconteceu?",
    "feeling_good": "Que bom! O que aconteceu de bom?",
    "frustration": "Entendo que isso frustra. Qual parte está dando errado?",
}


def canonical(text):
    return re.sub(r"\W+", " ", "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))).strip()


def prompt_text(messages):
    parts = []
    for role, text in messages:
        if role not in ("user", "assistant"):
            raise ValueError("Papel de conversa inválido")
        text = " ".join(text.split())
        parts.append(("Usuário: " if role == "user" else "TAC: ") + text)
    if not messages or messages[-1][0] != "user":
        raise ValueError("O contexto deve terminar com a mensagem do usuário")
    return "\n".join(parts) + "\nTAC:"


def make_sets():
    sets = {split: [] for split in ("train", "validation", "test")}
    for split in sets:
        for row in examples(split):
            if row["intent"] in REPLIES:
                sets[split].append({"messages": [["user", row["text"]]], "answer": REPLIES[row["intent"]], "skill": row["intent"]})
    names = {
        "train": "Ana Lucas Maria João Pedro Carla Diego Rita Bruno Sofia Luana Paulo Lúcia Miguel Lara Caio Nara Elisa Arthur Fernanda Igor Paula Davi Bia Júlio Helena Vítor Clara Marta Hugo Lia André Cecília Otávio Dani Gustavo Alice Bento Vera Eduardo".split(),
        "validation": "Rafael Marina Gabriel Camila Noemi Isadora Renata Tiago".split(),
        "test": "Valentina Murilo Rebeca Samuel Lorena Vicente Amanda Enzo".split(),
    }
    for split, values in names.items():
        for name in values:
            for intro in (f"meu nome é {name}", f"pode me chamar de {name}", f"eu me chamo {name}"):
                sets[split].append({"messages": [["user", intro]], "answer": f"Prazer, {name}.", "skill": "introduce_name"})
            for question in ("qual é meu nome?", "como eu me chamo?", "lembra meu nome?"):
                sets[split].append({"messages": [["user", f"Meu nome é {name}."], ["assistant", f"Prazer, {name}."], ["user", question]],
                                    "answer": f"Você me disse que seu nome é {name}.", "skill": "recall_name"})
    # Perguntas iguais, contextos diferentes: exige ler uma fala anterior.
    contexts = {
        "train": [
            ("Estou escrevendo uma história.", "Legal! Sobre o que é a história?", "Uma aventura com dragões.", "Quem é o personagem principal dessa aventura?", "writing"),
            ("Quero conversar sobre um jogo.", "Qual jogo você está jogando?", "É um jogo de exploração.", "O que você mais gosta de explorar nesse jogo?", "games"),
            ("Estou aprendendo programação.", "Qual linguagem você está estudando?", "Python.", "Qual parte de Python você está praticando?", "coding"),
            ("Hoje estou feliz.", "Que bom! O que aconteceu?", "Terminei meu projeto.", "Parabéns pelo projeto! Qual parte você mais gostou de fazer?", "achievement"),
            ("Meu dia foi difícil.", "Poxa. Quer contar o que aconteceu?", "Não consegui terminar meu trabalho.", "Qual parte do trabalho ficou mais difícil?", "difficulty"),
        ],
        "validation": [
            ("Eu estou criando uma história.", "Sobre o que é a história?", "É uma aventura com dragões.", "Quem é o personagem principal dessa aventura?", "writing"),
            ("Eu queria falar de um jogo.", "Qual jogo você está jogando?", "Um jogo de exploração.", "O que você mais gosta de explorar nesse jogo?", "games"),
            ("Estou estudando programação.", "Qual linguagem você estuda?", "Estou estudando Python.", "Qual parte de Python você está praticando?", "coding"),
        ],
        "test": [
            ("Queria te contar da minha história.", "Sobre o que ela é?", "Tem dragões e aventura.", "Quem é o personagem principal dessa aventura?", "writing"),
            ("Vamos falar do meu jogo?", "Que tipo de jogo?", "Eu exploro cenários.", "O que você mais gosta de explorar nesse jogo?", "games"),
            ("Estou começando a programar.", "Qual linguagem você escolheu?", "Escolhi Python.", "Qual parte de Python você está praticando?", "coding"),
        ],
    }
    for split, records in contexts.items():
        for first, reply, second, answer, skill in records:
            suffixes = ("", " 😊", ", sabe?", ", queria conversar sobre isso", " hoje") if split == "train" else ("",)
            for suffix in suffixes:
                sets[split].append({"messages": [["user", first], ["assistant", reply], ["user", second + suffix]], "answer": answer, "skill": skill})
    # Sem contexto suficiente, a resposta deve pedir esclarecimento.
    extras = {
        "train": ["e depois?", "continua", "me fala mais", "e isso?", "o que acha?", "tá", "sim", "não", "kkk", "hahaha"],
        "validation": ["pode continuar?", "e aí, o que acha?", "kkkkk"],
        "test": ["continua pra mim", "fala mais disso", "rsrs"],
    }
    for split, phrases in extras.items():
        for text in phrases:
            sets[split].append({"messages": [["user", text]], "answer": "Sobre qual assunto você quer conversar?", "skill": "ask_context"})
    # Muitas combinações novas ensinam cópia, em vez de decorar quarenta nomes.
    reserved = {canonical(name) for split in ("validation", "test") for name in names[split]}
    syllables = "ba be bi bo ca ce ci co da de di do fa fe fi fo ga ge gi go la le li lo ma me mi mo na ne ni no pa pe pi po ra re ri ro sa se si so ta te ti to va ve vi vo za ze zi zo".split()
    rng = random.Random(6719)
    synthetic_names = set()
    while len(synthetic_names) < 1200:
        name = "".join(rng.choice(syllables) for _ in range(rng.randint(2, 4))).capitalize()
        if canonical(name) not in reserved:
            synthetic_names.add(name)
    for name in sorted(synthetic_names):
        intro = rng.choice((f"meu nome é {name}", f"pode me chamar de {name}", f"eu me chamo {name}"))
        sets["train"].append({"messages": [["user", intro]], "answer": f"Prazer, {name}.", "skill": "introduce_name"})
        question = rng.choice(("qual é meu nome?", "como eu me chamo?", "lembra meu nome?"))
        sets["train"].append({"messages": [["user", f"Meu nome é {name}."], ["assistant", f"Prazer, {name}."], ["user", question]],
                              "answer": f"Você me disse que seu nome é {name}.", "skill": "recall_name"})
    # Cópias com consoantes finais e combinações arbitrárias evitam restringir
    # a tarefa aos nomes que terminam em sílabas consoante-vogal.
    for _ in range(500):
        name = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 9))).capitalize()
        if canonical(name) in reserved:
            continue
        sets["train"] += [
            {"messages": [["user", f"meu nome é {name}"]], "answer": f"Prazer, {name}.", "skill": "introduce_name"},
            {"messages": [["user", f"Meu nome é {name}."], ["assistant", f"Prazer, {name}."], ["user", "qual é meu nome?"]],
             "answer": f"Você me disse que seu nome é {name}.", "skill": "recall_name"},
        ]
    additional = {
        "unknown": ["qual foi o motivo do choro do personagem", "quem inventou o reino desse romance", "por que a rainha estava preocupada naquele capítulo", "qual era o aniversário do vilão", "qual foi o desfecho do conto que não mostrei", "quem é o autor da obra mencionada", "olá, qual é a massa do planeta Urano", "oi, o que é a radiação de Hawking", "boa tarde, como surgem as estrelas", "bom dia, quem descobriu a penicilina", "qual é a população atual dessa cidade", "como começou a guerra citada no livro", "o que aconteceu antes daquela cena", "por que o protagonista ficou nervoso", "quem participou do encontro citado nesse artigo", "olá, explique como um vulcão entra em erupção"],
        "greeting": ["um excelente dia para você", "passando para dar um oi", "um salve para você", "boa noite gente", "cheguei para conversar", "olá tudo bem por aí"],
        "identity": ["me fala seu nome por gentileza", "quem é a IA que conversa comigo", "qual nome eu uso para você", "diz aí como devo chamar você", "qual assistente está falando comigo"],
        "feeling_good": ["fiquei contente com isso", "estou muito contente hoje", "hoje estou super animado", "deu tudo certo e estou sorrindo", "acabei de conseguir algo legal", "aconteceu uma coisa ótima"],
        "feeling_low": ["hoje não estou me sentindo bem", "estou triste com o que aconteceu", "estou meio desanimado com a vida", "queria conversar porque fiquei mal", "nada me animou hoje"],
        "frustration": ["que raiva isso não funciona", "estou cansado de tentar e falhar", "como isso está difícil", "esse erro me deixa irritado", "fiquei sem paciência com essa dificuldade"],
        "local": ["essa inteligência funciona dentro do computador", "você depende de algum servidor", "qual máquina executa você", "você envia meus textos para serviços online", "a TAC usa modelos baixados"],
        "goodbye": ["estou indo embora até depois", "nos vemos outro dia", "preciso sair da conversa", "vou encerrar por agora", "depois a gente continua"],
        "capabilities": ["quais opções existem aqui", "mostra para mim o que pode fazer", "quero conhecer suas funções", "em que você pode me ajudar", "diga quais recursos você tem"],
        "memory": ["onde meus arquivos são preservados", "o que acontece com os materiais quando fecho", "você ainda terá meus documentos amanhã", "fechar o aplicativo apaga os estudos", "posso desligar e continuar depois"],
    }
    for skill, phrases in additional.items():
        for text in phrases:
            sets["train"].append({"messages": [["user", text]], "answer": REPLIES[skill], "skill": skill})
    for row in list(sets["train"]):
        if len(row["messages"]) == 1 and row["skill"] not in ("introduce_name", "recall_name"):
            for prefix in rng.sample(("ei, ", "tac, ", "olha, ", "então, ", "cara, ", "amiga, ", "opa, "), 2):
                sets["train"].append({**row, "messages": [["user", prefix + row["messages"][0][1]]]})
    # Ensina a responder a pergunta atual após uma conversa curta.
    previous = [("oi", REPLIES["greeting"]), ("qual é seu nome", REPLIES["identity"]), ("tudo bem", REPLIES["wellbeing"])]
    for row in list(sets["train"]):
        if len(row["messages"]) == 1:
            for earlier in rng.sample(previous, 2):
                sets["train"].append({**row, "messages": [["user", earlier[0]], ["assistant", earlier[1]], *row["messages"]]})
    # Elimina duplicatas entre splits, inclusive mudanças só de pontuação/acentos.
    seen = set()
    for split in sets:
        output = []
        for row in sets[split]:
            key = canonical(prompt_text(row["messages"]))
            if key not in seen:
                output.append(row)
                seen.add(key)
        sets[split] = output
    # O benchmark é congelado: uma nova paráfrase nunca pode consumir um item
    # antigo de validação/teste e removê-lo silenciosamente da avaliação.
    heldout = json.loads(Path(__file__).with_name("conversation_holdout.json").read_text(encoding="utf-8"))
    if heldout.get("format") != "TAC-conversation-heldout-v1":
        raise ValueError("Benchmark de conversa incompatível")
    sets["validation"], sets["test"] = heldout["validation"], heldout["test"]
    forbidden = {canonical(prompt_text(row["messages"])) for split in ("validation", "test") for row in sets[split]}
    forbidden_last = {canonical(row["messages"][-1][1]) for split in ("validation", "test") for row in sets[split] if len(row["messages"]) == 1}
    sets["train"] = [row for row in sets["train"] if canonical(prompt_text(row["messages"])) not in forbidden
                     and canonical(row["messages"][-1][1]) not in forbidden_last]
    return sets
