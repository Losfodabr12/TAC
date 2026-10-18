"""Currículo autoral verificável. Entidades e combinações reservadas por split.

Respostas são rótulos de treino/avaliação; nunca consultadas na inferência.
Não contém conversas humanas nem material baixado.
"""
import hashlib
import random
import json
from pathlib import Path


VERSION = "TAC-context-logic-v1"


def make_context_sets(fragments=()):
    result = {split: [] for split in ("train", "validation", "test")}
    values = {
        "train": "azul verde amarelo branco preto rosa roxo cinza laranja marrom".split(),
        "validation": ["turquesa", "violeta", "bege"],
        "test": ["dourado", "prateado", "vermelho"],
    }
    names = {
        "train": "Ana Lucas Carla Diego Pedro Maria Bruno Rita Sofia Caio".split(),
        "validation": ["Nilo", "Iara", "Celina"],
        "test": ["Tales", "Luiza", "Dora"],
    }
    rng = random.Random(10341)
    reserved = {v.casefold() for s in ("validation", "test") for v in values[s] + names[s]}
    reserved.update("Rafael Marina Gabriel Camila Noemi Isadora Renata Tiago Valentina Murilo Rebeca Samuel Lorena Vicente Amanda Enzo".lower().split())
    syllables = "ba be bi bo ca ce ci co da de di do fa fe fi fo ga ge gi go la le li lo ma me mi mo na ne ni no pa pe pi po ra re ri ro sa se si so ta te ti to va ve vi vo za ze zi zo".split()
    pool = set()
    attempt = 0
    while len(pool) < 1600:
        attempt += 1
        if fragments and attempt % 3 == 0:
            value = "".join(rng.choice(fragments) for _ in range(rng.randint(1, 3))).lower()
        elif attempt % 2:
            value = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 12)))
        else:
            value = "".join(rng.choice(syllables) for _ in range(rng.randint(2, 4)))
        if value not in reserved and len(value) <= 24:
            pool.add(value)
    pseudo = sorted(pool)
    def add(split, messages, answer, skill):
        result[split].append({"messages": messages, "answer": answer, "skill": skill})
    for split in result:
        count = 6000 if split == "train" else 18
        for i in range(count):
            name = rng.choice(names[split] if split != "train" or i % 4 == 0 else pseudo).capitalize()
            colors = values[split] if split != "train" or i % 4 == 0 else pseudo
            color, old = rng.sample(colors, 2)
            first = rng.choice((f"Minha cor favorita é {old}.", f"Eu gosto mais de {old}.", f"Anote minha preferência: {old}."))
            question = rng.choice(("Qual é minha cor favorita?", "De qual cor eu gosto?", "Qual foi a cor que eu falei?"))
            add(split, [["user", f"Minha cor favorita é {color}."], ["assistant", "Entendido."], ["user", question]],
                f"Sua cor favorita é {color}.", "context_color")
            messages = [["user", first], ["assistant", "Entendido."], ["user", f"Mudei de ideia. Agora minha cor favorita é {color}."],
                        ["assistant", "Certo, vou considerar a mudança."], ["user", question]]
            add(split, messages, f"Agora sua cor favorita é {color}.", "context_update")
            obj = rng.choice(("livro", "caderno", "telefone", "lápis", "vaso", "quadro"))
            add(split, [["user", f"{name} tem um {obj} {color}."], ["assistant", "Certo."], ["user", f"Qual é a cor do {obj} de {name}?"]],
                f"O {obj} de {name} é {color}.", "context_fact")
            a, b = rng.sample(("cozinha", "sala", "quarto", "jardim", "escritório", "varanda"), 2)
            add(split, [["user", f"{name} estava no local {a}. Depois foi para o local {b}. Onde {name} está agora?"]],
                f"{name} está no local {b}.", "context_location")
            # Um fato contrário é explicitado: não assumir a regra inversa.
            active = bool(i % 2)
            fact = "está ligado" if active else "não está ligado"
            add(split, [["user", f"Regra: se o botão {color} está ligado, a luz acende. O botão {color} {fact}. Podemos concluir que a luz acendeu?"]],
                "Sim, pela regra e pelo fato informado." if active else "Não. A regra não informa o resultado quando o botão está desligado.", "logic_condition")
        # Combinações aritméticas separadas de forma determinística.
        for a in range(25):
            for b in range(25):
                bucket = int(hashlib.sha256(f"{a}:{b}".encode()).hexdigest()[:8], 16) % 10
                chosen = "validation" if bucket == 0 else "test" if bucket == 1 else "train"
                if chosen != split:
                    continue
                for op in ("+", "-"):
                    answer = a + b if op == "+" else a - b
                    add(split, [["user", f"Quanto é {a} {op} {b}?"]], f"{a} {op} {b} = {answer}.", "arithmetic")
    heldout = json.loads(Path(__file__).with_name("context_holdout.json").read_text(encoding="utf-8"))
    if heldout.get("format") != "TAC-context-heldout-v1":
        raise ValueError("Benchmark de contexto incompatível")
    result["validation"], result["test"] = heldout["validation"], heldout["test"]
    # Deduplicação global, validação e teste permanecem reservados.
    seen = set()
    for split in ("validation", "test", "train"):
        clean = []
        for row in result[split]:
            key = tuple(tuple(message) for message in row["messages"])
            if key not in seen:
                clean.append(row)
                seen.add(key)
        result[split] = clean
    return result
