"""Material autoral revisado e mistura explícita; não são diálogos humanos."""
import json
import random
import re
from pathlib import Path

from .conversation_lessons import canonical, prompt_text

VERSION = 'TAC-fluency-balanced-v1'
SOURCE = Path(__file__).with_name('fluency_dialogues.json')


def lessons():
    data = json.loads(SOURCE.read_text(encoding='utf-8'))
    if data['format'] != 'TAC-reviewed-fluency-v1':
        raise ValueError('Material revisado incompatível')
    return data


def make_fluency_sets():
    source = lessons()
    result = {s:[] for s in ('train','validation','test')}
    for split in result:
        for entry in source[split]:
            scene, turns = entry[:2]
            rules = entry[2] if len(entry) > 2 else [[] for _ in turns]
            for style in (('', 'tac, ', 'olha, ', 'então, ', 'ei, ', 'uma coisa: ') if split == 'train' else ('',)):
                history = []
                for turn, ((user, answer), rule) in enumerate(zip(turns,rules)):
                    user = style + user
                    result[split].append({'messages':[*history[-6:],['user',user]], 'answer':answer,
                        'skill':'reviewed_' + scene, 'mixture':'reviewed', 'episode':scene,
                        'turn':turn, 'relevance_rules':rule})
                    history += [['user',user],['assistant',answer]]
    return result


def natural_legacy(row):
    """Mantém tarefas antigas com valores naturais; remove cópia de pseudopalavras."""
    from .natural_dialogues import SLOTS
    skill, answer = row['skill'], row['answer']
    names = set('Ana Lucas Carla Diego Pedro Maria Bruno Rita Sofia Paulo Lara Caio Bia Hugo Vera'.split())
    colors = set('azul verde amarelo branco preto rosa roxo cinza laranja marrom'.split())
    if skill in ('introduce_name','recall_name'):
        return any(re.search(r'\b' + name + r'\b',answer) for name in names)
    for slot, spec in SLOTS.items():
        if skill.startswith('natural_' + slot + '_'):
            return any(answer == spec[stage].format(value=v) for stage in ('ack','changed','recall') for v in spec['train'])
    if skill in ('context_color','context_update','context_fact'):
        return bool(set(re.findall(r'\w+',answer)) & colors)
    if skill == 'context_location':
        return answer.split()[0] in names
    if skill == 'logic_condition':
        return bool(set(re.findall(r'\w+', row['messages'][0][1])) & colors)
    return True


def batch_kind(step, micro, accumulation):
    # 11/20 diálogos revisados, 6/20 linguagem, 3/20 exercícios antigos.
    position = (step * accumulation + micro) % 20
    return 'reviewed' if position < 11 else ('language' if position < 17 else 'legacy')


NARRATIVES = [
    '{name} precisava entregar um trabalho, mas achava a tarefa grande demais. Em vez de tentar terminar tudo, separou uma parte pequena e começou por ela. Quando essa parte ficou pronta, conseguiu perceber melhor o que ainda faltava. O trabalho não terminou naquele instante, mas já havia um caminho para continuar.',
    '{name} procurou um caderno pela casa e não encontrou. Depois parou para lembrar a sequência do dia: primeiro estudou na cozinha, depois levou a mochila para o quarto. Ao conferir a mochila, encontrou o caderno entre os livros. Lembrar os acontecimentos ajudou a reduzir a procura.',
    '{name} queria sair para passear, mas a chuva mudou os planos. A comida já estava preparada e os amigos ainda queriam se encontrar. Eles combinaram um encontro em casa e escolheram uma atividade para fazer juntos. O passeio mudou de lugar, sem perder o motivo de reunir o grupo.',
    '{name} recebeu uma mensagem curta e achou que o amigo estava bravo. Antes de responder com irritação, perguntou o que ele quis dizer. O amigo explicou que estava com pressa e não tinha percebido o tom da mensagem. A conversa esclareceu uma intenção que o texto sozinho deixava ambígua.',
    '{name} fez um desenho e percebeu que as sombras não combinavam. Escolheu uma única fonte de luz e marcou quais partes deveriam ficar claras. Depois testou as mudanças numa cópia, preservando o desenho original. Comparar as duas versões ajudou a entender o efeito das sombras.',
    '{name} estava escrevendo uma história e explicou muitas vezes que o personagem tinha medo. Na revisão, retirou algumas repetições e mostrou o personagem hesitando diante de uma porta. A ação passou a sugerir a emoção. O leitor ganhou espaço para perceber o medo pelo comportamento.',
    '{name} tentou tocar uma música depressa e errou sempre a mesma passagem. Separou o trecho difícil e começou numa velocidade confortável. Quando conseguiu tocá-lo com mais estabilidade, ligou a passagem ao restante da música. Praticar a dificuldade específica foi mais útil do que voltar sempre ao começo.',
    '{name} esqueceu de responder a um convite e ficou sem jeito. Como ainda queria participar, enviou uma mensagem reconhecendo a demora e perguntou se ainda podia confirmar. A outra pessoa respondeu e os dois combinaram o horário. Uma pergunta direta resolveu o que a vergonha estava adiando.',
    '{name} e o amigo tinham ideias diferentes para um projeto. Antes de fazer duas versões completas, conversaram sobre o objetivo e dividiram as tarefas. Também combinaram um momento para comparar as partes. O acordo não eliminou todas as dúvidas, mas tornou mais claro o que cada um faria.',
    '{name} queria organizar todas as anotações de uma vez e acabou sem começar. Escolheu então uma única lista para reunir as ideias daquela semana. Depois de usar a lista, percebeu quais títulos ajudavam a encontrar as coisas. A organização foi sendo ajustada à rotina, em vez de nascer pronta.',
    '{name} achou que o programa estava quebrado, mas a mensagem indicava um nome de variável desconhecido. Ao comparar os nomes, encontrou uma letra trocada. Corrigiu a diferença e executou novamente um exemplo pequeno. O teste confirmou a correção daquele erro, sem provar que o programa inteiro estava livre de problemas.',
    '{name} ficou triste porque o primeiro resultado não saiu como esperava. Depois conseguiu identificar uma parte que tinha funcionado e outra que precisava de revisão. Escolheu melhorar um ponto concreto antes de tentar refazer tudo. Reconhecer a dificuldade e observar o progresso puderam acontecer ao mesmo tempo.',
    '{name} começou a ler uma história e encontrou uma palavra desconhecida. Primeiro observou a frase e o que estava acontecendo na cena. Em seguida procurou o significado e releu o trecho. O contexto ajudou a escolher entre sentidos diferentes da mesma palavra.',
    '{name} precisava explicar uma ideia, mas usava muitos detalhes antes de apresentar o objetivo. Reorganizou a explicação: começou pelo problema, mostrou um exemplo e só depois falou dos detalhes. Ao contar novamente, conseguiu acompanhar melhor a sequência. Quem ouvia também podia fazer perguntas em pontos mais claros.',
    '{name} prometeu fazer uma tarefa e percebeu que não conseguiria terminar no horário. Em vez de desaparecer, avisou sobre o atraso e explicou o que já tinha feito. As pessoas envolvidas puderam ajustar os planos. O aviso não apagou o atraso, mas evitou deixar os outros sem informação.',
    '{name} terminou uma primeira versão e logo enxergou vários problemas. Guardou uma cópia antes de revisar e escolheu uma mudança de cada vez. Assim pôde comparar os resultados e recuperar a versão anterior quando uma ideia não funcionou. Revisar passou a ser um processo de observar e testar.',
]


def make_language_corpus(legacy=None):
    """Narrativas causais + diálogos completos; reserva textual distinta."""
    rng = random.Random(804211)
    train = []
    provenance = []
    names = 'Ana Lucas Carla Diego Maria Bruno Rita Sofia Paulo Lara Caio Bia Hugo Vera Pedro Joana Clara Sara Davi'.split()
    for index, template in enumerate(NARRATIVES):
        for name in names:
            text = template.format(name=name)
            train.append(text)
            provenance.append({'authored_narrative':index,'name':name})
    for scene, turns, *_ in lessons()['train']:
        text = '\n'.join('Usuário: ' + u + '\nTAC: ' + a for u,a in turns)
        # Uma entrada por diálogo completo: sem inflar contagem por repetição.
        train.append(text)
        provenance.append({'authored_scene':scene})
    if legacy:
        # Amostras antigas já autorizadas, congeladas, em proporção pequena.
        choices = list(range(len(legacy['train'])))
        rng.shuffle(choices)
        for index in choices[:max(1,len(train)//7)]:
            train.append(legacy['train'][index])
            provenance.append(legacy['provenance'][index])
    validation = [
        'Uma pessoa pode conhecer as palavras de uma frase e ainda precisar do contexto para entender a intenção. Uma pergunta curta ajuda a esclarecer a dúvida antes de responder com uma suposição.',
        'Duas tarefas podem parecer igualmente urgentes. Comparar os prazos e o que depende de cada uma ajuda a definir uma ordem. O plano pode mudar quando chega uma informação nova.',
        'Um personagem quer atravessar uma floresta, mas não conhece o caminho. Ao encontrar uma trilha, ele tem uma possibilidade, não uma garantia. A história pode mostrar como ele verifica a direção antes de seguir.',
        'O primeiro teste revelou um problema que antes não estava visível. A equipe descreveu o erro, fez uma mudança e repetiu a verificação. Registrar as versões permitiu comparar os resultados.',
        'Uma explicação pode começar simples e ganhar detalhes conforme aparecem dúvidas. Um exemplo concreto ajuda a ligar uma palavra abstrata a uma situação compreensível.',
        'Uma pessoa disse que precisava de companhia, mas não queria explicar o motivo. O amigo respeitou esse limite e ofereceu ficar por perto. A presença não exigia conhecer todos os detalhes.',
    ]
    from .neural.data import Corpus
    return Corpus(train, validation, provenance)


def score_reply(text, rules):
    plain = canonical(text)
    drift = any(s in plain for s in ('sua cor favorita', 'vou chamar voce de', 'esta no local'))
    groups = [bool(re.search(r'\b(?:' + rule + r')',plain)) for rule in rules]
    words = plain.split()
    repeated = len(words) > 12 and len(set(zip(words,words[1:],words[2:]))) < len(words)*.5
    return {'topic_relevance':bool(groups) and all(groups), 'slot_drift':drift,
            'nonempty':len(words) >= 3, 'repeated':repeated,
            'passes_simple_rules':bool(groups) and all(groups) and not drift and not repeated and len(words) >= 3}


def fluency_ready(metrics, baseline):
    if metrics is None:
        return True
    target = max(.60, min(.90, baseline['rule_accuracy'] + .15), baseline['rule_accuracy'] - .02)
    return (metrics['rule_accuracy'] >= target
            and metrics['slot_drift_rate'] <= .05)
