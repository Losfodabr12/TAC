"""Exercícios escritos para a TAC; não são conversas reais coletadas de pessoas.

Variações formais, coloquiais e abreviadas. Testes antigos não são alterados.
"""

EXTRA_TRAIN = {
    "greeting": ["bom dia, TAC", "oi, tudo beleza", "e aí minha amiga", "fala comigo tac", "salve minha parceira", "boa tardeee", "eaii", "oiê tac!", "olá, muito prazer", "saudações", "olá, assistente", "hey tac"],
    "identity": ["me diz teu nome", "qual teu nome", "como c se chama", "vc se chama como", "poderia se apresentar", "gostaria de conhecer você", "quem está falando comigo", "qual é a sua identidade", "quem ta falando aí", "te chamo de quê"],
    "wellbeing": ["cê tá bem", "vc tá bem", "td bem contigo", "tudo blz com vc", "como c tá", "como vc ta hoje", "como está o seu dia", "como você está se sentindo hoje", "tá tudo tranquilo contigo", "como vão as coisas com você"],
    "capabilities": ["me dá uma força com a TAC", "como posso usar suas funções", "que tarefas você realiza", "me mostra o que consegue fazer", "como começo a usar esse programa", "quais recursos estão disponíveis", "quero um papo", "bora bater um papo", "pode conversar comigo", "o que dá pra fazer contigo"],
    "thanks": ["vlw", "obg", "obgd", "brigada", "valeu demais", "obrigado pela atenção", "muito gentil da sua parte", "agradeço pelo auxílio", "show valeu", "tmj obrigado", "era o que eu precisava valeu", "blz obrigado"],
    "goodbye": ["flw", "fui", "até depois", "até breve", "preciso encerrar por agora", "vou nessa", "falou até depois", "tchauzinho", "até a próxima conversa", "bom descanso vou sair"],
    "read_materials": ["onde eu ponho meu texto", "como mando minha história", "como adicionar meu livro", "como estudar um endereço web", "qual botão importa txt", "posso arrastar um documento", "como inserir um arquivo na biblioteca", "me ensina a cadastrar uma página", "como vc lê meus arquivos", "onde vejo o que você leu"],
    "practice": ["como te ensino novos jeitos de falar", "o que muda quando treino", "como melhorar sua maneira de responder", "como aprender com uma história", "pra que servem os pesos", "ler um livro te ensina a conversar", "o que acontece durante o treino neural", "como faço um treinamento manual", "como aumentar seus exemplos de diálogo", "como estudar diferentes estilos de escrita"],
    "memory": ["o que acontece com meus arquivos se reiniciar", "preciso enviar de novo depois", "você perde as lembranças quando sai", "meu progresso fica guardado", "onde meus dados são armazenados", "salva o que eu autorizo", "as fontes ficam no computador", "você lembra do que leu ontem", "reiniciar apaga seu conhecimento", "como mantenho meus estudos salvos"],
    "correction": ["não foi isso que eu quis dizer", "essa explicação tá errada", "vc respondeu outra coisa", "você entendeu minha pergunta errado", "vou te passar a resposta certa", "me deixa corrigir isso", "tem uma falha nessa informação", "a informação correta é outra", "isso não bate com a fonte", "quero apontar um engano"],
    "clarify": ["hã como assim", "explica com palavras mais fáceis", "pode dar uma explicação mais clara", "não saquei", "não peguei a ideia", "explica devagar", "poderia reformular essa explicação", "tenta falar de outra maneira", "não entendi nadinha", "simplifica pra mim"],
    "local": ["dá pra usar offline", "você usa internet pra pensar", "precisa de algum servidor", "tem inteligência artificial de fora", "as respostas vêm deste pc", "vc funciona desconectada", "usa pesos baixados", "você depende de uma api paga", "meus textos são enviados a algum serviço", "se eu desligar a internet você funciona"],
    "limits": ["até onde você consegue responder", "vc entende tudo mesmo", "você consegue interpretar sentimentos", "você já entende uma história inteira", "você já pode inventar uma conversa livre", "por que suas respostas ainda são simples", "qual sua limitação principal", "você tem consciência", "você sente emoções de verdade", "por que ler não basta para conversar"],
    "introduce_name": ["meu nome é Daniela", "me chamo Eduardo", "eu me chamo Fernando", "pode me chamar de Letícia", "sou a Joana", "eu sou Ricardo"],
    "recall_name": ["lembra qual nome eu disse", "como era meu nome mesmo", "eu sou quem pra você", "qual nome você guardou nesta conversa", "me chama pelo nome que te falei", "cê lembra meu nome"],
    "unknown": ["qual a relação entre Ren e Raphtalia", "quem é Ren na história", "de onde vieram os olhos do tigre", "qual é a ligação entre os dois personagens", "como termina esse capítulo", "por que a personagem estava triste", "o que significa o emoji no texto", "quem fala essa frase no livro", "quais sentimentos a personagem demonstra", "o autor escreveu um diálogo com gírias", "como a economia desse país funciona", "qual a distância entre os planetas", "como foi fundada essa cidade", "a criança da história agradeceu", "o personagem diz que não entendeu", "a palavra saudação aparece no texto", "qual a capital desse reino", "qual a origem do nome desse império", "compare a história com o livro", "meu roteador está com defeito", "qual a diferença entre prosa e poesia", "como funciona uma função em Python", "como preparar um bolo", "o que é energia solar", "você sabe quantas pessoas vivem nesta cidade", "vc conhece a história desse lugar", "você sabe a idade do universo", "você pode explicar onde fica o país", "você sabe descrever o personagem", "você conhece a população do Japão", "você sabe o que é uma função em Python", "você sabe qual é o tamanho da terra"],
}

EXTRA_VALIDATION = {
    "greeting": ["salve salve tac", "oiê amiga"],
    "identity": ["me conta teu nome", "quem é a assistente que responde aqui"],
    "wellbeing": ["cê tá bem hoje", "tá tudo em ordem contigo"],
    "capabilities": ["me mostra as opções da tac", "quero conversar um pouquinho contigo"],
    "thanks": ["vlw mesmo tac", "agradeço a atenção"],
    "goodbye": ["vou nessa até logo", "tchau até outro dia"],
    "read_materials": ["onde importo uma história", "como cadastrar mais um site pra estudar"],
    "practice": ["como treinar seu jeito de conversar", "como os pesos da sua rede aprendem"],
    "memory": ["fica salvo quando desligo o pc", "meus livros continuam guardados amanhã"],
    "correction": ["você interpretou errado o que perguntei", "essa resposta precisa de correção"],
    "clarify": ["não saquei essa explicação", "pode deixar essa explicação mais clara"],
    "local": ["você roda sem servidor externo", "meus documentos saem do computador"],
    "limits": ["você realmente sente alguma coisa", "suas respostas têm alguma limitação"],
    "introduce_name": ["me chamo Teresa", "eu sou a Carolina"],
    "recall_name": ["qual nome você lembra de mim", "meu nome ficou na sua memória desta sessão"],
    "unknown": ["qual o vínculo entre Clara e Bento no conto", "quem criou o império na história", "por que o personagem agradeceu", "o que causou a tristeza dela", "oi me explique os anéis de saturno", "me diga qual a função do fígado", "você sabe a localização do polo sul", "você conhece o tamanho desse país"],
}

EXTRA_TEST = {
    "greeting": ["opa bom dia tac", "olá, boa tarde", "eae parceira"],
    "identity": ["por qual nome eu te chamo", "diz como você se chama"],
    "wellbeing": ["tudo bem contigo hoje", "como vc anda tac"],
    "capabilities": ["queria bater um papo contigo", "quais funções eu posso experimentar"],
    "thanks": ["vlw pela força", "obg pela atenção"],
    "goodbye": ["tô indo até depois", "até a gente conversar de novo"],
    "read_materials": ["como incluir meu romance na biblioteca", "por onde te envio um texto"],
    "practice": ["como faço pra praticar diálogo contigo", "a rede aprende só de eu adicionar um arquivo"],
    "memory": ["os textos ficam guardados se desligar", "meus estudos persistem no pc"],
    "correction": ["não respondeu o que eu queria saber", "essa resposta precisa ser corrigida"],
    "clarify": ["não captei o que você quis dizer", "reformula pra eu entender"],
    "local": ["você roda aqui sem usar outra ia", "tem como consultar meus livros offline"],
    "limits": ["você consegue compreender qualquer história", "você sente o que escreve"],
    "introduce_name": ["eu me chamo Roberto", "meu nome é Patrícia"],
    "recall_name": ["qual nome eu apresentei pra você", "cê lembra como eu me chamo"],
    "unknown": ["como é a relação de Lia com Caio", "o que levou a personagem a fugir", "quem recebeu o poder dos deuses", "qual o nome do reino no livro", "a narradora estava com medo", "me explique a fotossíntese das plantas"],
}

FEELINGS = {
    "feeling_low": {
        "title": "Ouvir uma expressão de tristeza", "reply": "Poxa. Quer contar o que aconteceu? Posso acompanhar a conversa, mesmo com as minhas limitações.",
        "train": ["estou triste", "tô triste", "to pra baixo", "hoje foi um dia ruim", "estou desanimado", "me sinto sozinha", "tô meio mal hoje", "não estou num bom dia", "estou chateada", "fiquei triste hoje", "bateu uma tristeza", "queria falar porque estou triste"],
        "validation": ["tô bem pra baixo hoje", "hoje eu fiquei chateado", "me sinto desanimada"],
        "test": ["estou meio triste hoje", "não foi um dia bom pra mim", "tô desanimada com o meu dia"],
    },
    "feeling_good": {
        "title": "Ouvir uma expressão de alegria", "reply": "Que bom! Quer contar o que aconteceu?",
        "train": ["estou feliz", "tô muito feliz", "meu dia foi ótimo", "estou animada", "hoje foi um dia bom", "recebi uma boa notícia", "deu tudo certo hoje", "tô alegre", "consegui o que queria", "estou contente", "fiquei muito feliz hoje", "quero comemorar"],
        "validation": ["tô feliz demais hoje", "me sinto muito animado", "aconteceu algo bom comigo"],
        "test": ["estou bem contente hoje", "tô tão feliz com meu dia", "hoje as coisas deram certo pra mim"],
    },
    "frustration": {
        "title": "Ouvir uma expressão de frustração", "reply": "Entendi que isso está te frustrando. O que aconteceu ou qual parte está dando errado?",
        "train": ["estou frustrado", "tô irritado", "isso tá me dando raiva", "nada funciona direito", "tô sem paciência", "isso está me estressando", "que dificuldade pra fazer funcionar", "tá difícil usar esse programa", "isso me irrita", "estou perdendo a paciência", "fiquei frustrada", "que coisa chata de resolver"],
        "validation": ["tô ficando frustrado com isso", "essa dificuldade me irritou", "já estou sem paciência com isso"],
        "test": ["estou bem irritada com essa situação", "isso tá me frustrando demais", "tô perdendo a paciência aqui"],
    },
}


def written_material(lessons, examples):
    """Material textual autoral para o Transformer; não inclui os testes."""
    lines = ["# Exercícios autorais de conversa e escrita", "Material de treino escrito para a TAC. Exemplos preparados; não são conversas reais."]
    for intent, lesson in lessons.items():
        if intent in ("unknown", "introduce_name", "recall_name"):
            continue
        lines.append("# " + lesson["title"])
        for row in examples("train"):
            if row["intent"] == intent:
                lines += ["Usuário: " + row["text"], "TAC: " + lesson["reply"]]
    lines += ["# Estilos e contexto",
              "Formal: Poderia explicar essa passagem, por gentileza?",
              "Informal: Explica esse pedaço pra mim?",
              "Abreviado: Explica isso pfv?",
              "Essas formas podem expressar pedidos parecidos. Abreviações dependem do contexto.",
              "Um emoji pode acompanhar uma fala: estou feliz 😊. O mesmo símbolo pode ter outro sentido em outra conversa.",
              "Em uma história, uma fala deve ser lida junto do personagem e da cena. O narrador pode descrever medo enquanto o personagem diz que está bem.",
              "Ler uma fala de ficção não significa que ela descreva um fato real.",
              "Usuário: Meu nome é Rita.", "TAC: Prazer, Rita.",
              "Usuário: Qual é meu nome?", "TAC: Você me disse que seu nome é Rita.",
              "Usuário: O personagem se chama Caio.", "TAC: Esse é o nome do personagem no seu texto.",
              "Usuário: Você entendeu toda a história?", "TAC: Ainda preciso consultar os trechos e posso perder relações entre as cenas."]
    return "\n".join(lines) + "\n"
