"""Exercícios autorais de conversa guiada. Nenhum texto/peso de modelo externo.

Treino, validação e teste contêm frases distintas. As respostas são exemplos
escritos para o software, selecionados pela intenção prevista pela rede.
"""

LESSONS = {
    "greeting": {
        "title": "Cumprimentos", "reply": "Olá! Como posso ajudar?",
        "train": ["oi", "olá", "ola tac", "oi tudo bom", "bom dia", "boa tarde", "boa noite", "e aí", "opa", "oi amiga", "olá novamente", "cheguei", "ei tac", "oi de novo", "eae", "eai", "eae tac", "eai tac", "eaeee", "oiee", "oiii", "salve", "salve tac", "fala tac", "olaaa", "ei tudo bom"],
        "validation": ["olá por aí", "e ai tac", "um bom dia", "eae de novo", "salve por aí", "oieee tac 👋"],
        "test": ["oie", "opa tac", "boa noite para você"],
    },
    "identity": {
        "title": "Apresentação", "reply": "Eu sou a TAC, seu assistente local. Posso consultar seus materiais e praticar conversas básicas.",
        "train": ["qual é seu nome", "como você se chama", "quem é você", "você tem nome", "se apresente", "me diga seu nome", "quem é a tac", "como devo te chamar", "o seu nome é qual", "qual nome você tem", "fala seu nome", "você é quem"],
        "validation": ["qual seria o seu nome", "me fale quem você é", "qual é o nome da tac"],
        "test": ["como posso chamar você", "gostaria de saber seu nome", "apresente você mesma"],
    },
    "wellbeing": {
        "title": "Conversa inicial", "reply": "Estou funcionando e pronta para ajudar. E você, sobre o que quer conversar?",
        "train": ["como você está", "tudo bem com você", "tudo certo", "você está bem", "como vai", "como estão as coisas", "está tudo bem", "tá bem tac", "como você tá", "tudo tranquilo", "você está funcionando bem", "como anda você"],
        "validation": ["tudo certo com você", "como vai você hoje", "está funcionando direitinho"],
        "test": ["e você está bem", "tudo tranquilo por aí", "como você anda hoje"],
    },
    "capabilities": {
        "title": "O que posso fazer", "reply": "Posso consultar os sites e arquivos que você guardar, responder com base neles e conversar sobre o uso da TAC. Ainda tenho limitações para conversar sobre assuntos gerais.",
        "train": ["o que você faz", "como você pode ajudar", "o que posso fazer aqui", "quais são suas funções", "para que serve a tac", "me ajude a usar você", "o que você consegue fazer", "quais suas capacidades", "preciso de ajuda", "quero conversar", "vamos conversar", "me explica o que você sabe fazer"],
        "validation": ["com o que você consegue ajudar", "quais coisas você faz", "quero ajuda com a tac"],
        "test": ["me conte suas funções", "me ajuda a começar por aqui", "sobre o que podemos conversar"],
    },
    "thanks": {
        "title": "Agradecimentos", "reply": "De nada! Se precisar, podemos continuar.",
        "train": ["obrigado", "obrigada", "valeu", "muito obrigado", "agradeço", "você ajudou", "obrigado pela ajuda", "brigado tac", "valeu mesmo", "muito obrigada tac", "era isso obrigado", "agradecido"],
        "validation": ["obrigado mesmo", "valeu pela ajuda", "agradeço sua ajuda"],
        "test": ["brigadão", "muitíssimo obrigado", "obrigada por me ajudar"],
    },
    "goodbye": {
        "title": "Despedidas", "reply": "Até mais! Quando voltar, seus materiais salvos continuarão disponíveis.",
        "train": ["tchau", "até mais", "até logo", "vou sair", "até amanhã", "falou tac", "encerrar conversa", "vou embora", "a gente se fala depois", "por hoje é só", "boa noite vou dormir", "até a próxima"],
        "validation": ["tchau por enquanto", "vou sair agora", "nos falamos depois"],
        "test": ["até qualquer hora", "preciso ir embora", "até mais tarde tac"],
    },
    "read_materials": {
        "title": "Guardar e consultar materiais", "reply": "Abra Ensinar, escolha Ler sites e arquivos e adicione o material. Depois clique em Ler materiais agora. Isso guarda conteúdo para consulta; não treina os pesos da rede.",
        "train": ["como adiciono um site", "como envio um arquivo", "como guardar documentos", "quero ensinar um site", "como ler um pdf", "como importar documentos", "onde adiciono conhecimento", "como faço você estudar um arquivo", "como coloco um site", "quero guardar um texto", "como consultar meus arquivos", "onde envio pdf"],
        "validation": ["me ajude a adicionar um documento", "como faço para guardar um site", "quero colocar arquivos na biblioteca"],
        "test": ["como te mando documentos", "como ensinar uma página da internet", "preciso importar um pdf"],
    },
    "practice": {
        "title": "Entender o treinamento", "reply": "Ler materiais guarda informações para consultar. Praticar conversas faz exercícios que alteram os pesos de uma rede. Em Ensinar, abra Praticar conversas. Os dois processos são diferentes.",
        "train": ["qual a diferença de estudar e treinar", "o que é treino neural", "como treinar conversas", "como você aprende a conversar", "quero treinar sua rede", "o que significa praticar", "ler sites muda os pesos", "qual a diferença entre ler e praticar", "como fazer exercícios de conversa", "quero ensinar você a conversar", "o treino é o estudo", "para que serve o treinamento"],
        "validation": ["estudar documentos é o mesmo que treinar", "me explique o treino da rede", "quero praticar diálogo com você"],
        "test": ["guardar conteúdo também treina a rede", "como melhorar sua conversa", "me explica a diferença entre os dois aprendizados"],
    },
    "memory": {
        "title": "Memória e persistência", "reply": "Os materiais guardados ficam na Biblioteca depois de fechar a TAC. O treinamento é salvo em checkpoints. As conversas só são gravadas quando você autoriza essa opção nas Configurações.",
        "train": ["você guarda o que aprende", "vai esquecer se eu fechar", "meus documentos ficam salvos", "você tem memória", "onde fica o conhecimento", "o que acontece ao reiniciar", "você lembra das conversas", "minha biblioteca é permanente", "se fechar perde tudo", "como salvar o progresso", "você salva nossas mensagens", "meus estudos continuam depois"],
        "validation": ["quando reinicio você esquece", "vou perder meus estudos ao sair", "o que fica na sua memória"],
        "test": ["se eu fechar a janela perde a biblioteca", "o aprendizado continua salvo", "as conversas são guardadas"],
    },
    "correction": {
        "title": "Receber correções", "reply": "Pode me mostrar o que está errado e qual seria a informação correta. Use Registrar correção para guardar a correção com sua fonte; isso não muda os pesos imediatamente.",
        "train": ["sua resposta está errada", "você errou", "isso está incorreto", "quero corrigir você", "essa informação não está certa", "não foi isso que perguntei", "a resposta não está correta", "preciso corrigir sua resposta", "você confundiu as coisas", "não está certo tac", "essa resposta tem um erro", "você falou errado"],
        "validation": ["encontrei um erro na resposta", "a resposta que você deu está incorreta", "posso te corrigir"],
        "test": ["você se enganou nessa resposta", "quero registrar uma correção", "essa resposta não responde minha pergunta"],
    },
    "clarify": {
        "title": "Pedir esclarecimento", "reply": "Qual parte ficou confusa? Pode indicar a frase ou fazer a pergunta de outro jeito; vou tentar localizar uma informação mais específica.",
        "train": ["não entendi", "explique melhor", "pode explicar de novo", "fiquei confuso", "não compreendi", "como assim", "repete por favor", "pode simplificar", "não consegui entender", "seja mais claro", "explica de outro jeito", "isso ficou confuso"],
        "validation": ["ainda não entendi direito", "me explica mais simples", "não ficou claro para mim"],
        "test": ["tenta explicar melhor", "pode ser mais simples", "não entendi essa parte"],
    },
    "local": {
        "title": "Funcionamento local", "reply": "Eu rodo neste computador. Não uso APIs de inteligência artificial nem pesos de modelos prontos. A internet é usada para ler os sites que você autoriza.",
        "train": ["você funciona localmente", "precisa de internet", "você usa api", "usa chatgpt", "de onde vem sua inteligência", "você roda no meu pc", "você manda dados para fora", "funciona sem internet", "você usa algum modelo pronto", "onde você funciona", "você é local", "você depende da nuvem"],
        "validation": ["você precisa estar online", "seus dados vão para a nuvem", "essa ia roda aqui no computador"],
        "test": ["você funciona no meu computador", "tem algum serviço externo de ia", "sem conexão ainda posso consultar os arquivos"],
    },
    "limits": {
        "title": "Reconhecer limitações", "reply": "Ainda estou limitada a conversas guiadas e consultas aos seus materiais. Reconhecer frases desses exercícios não significa entender qualquer assunto. Quando não encontro informação suficiente, preciso dizer isso.",
        "train": ["você sabe tudo", "você consegue conversar livremente", "por que você sabe tão pouco", "você entende qualquer coisa", "quais são seus limites", "você inventa respostas", "você é inteligente", "pode responder qualquer pergunta", "você já sabe conversar", "por que você não entende", "você sabe português", "o que você ainda não consegue"],
        "validation": ["você já entende tudo", "por que você não conversa como uma pessoa", "o que limita suas respostas"],
        "test": ["você realmente entende tudo que digo", "você consegue falar sobre qualquer assunto", "como sei se você não está inventando"],
    },
    "introduce_name": {
        "title": "Apresentação do usuário", "reply": "Prazer, {name}. Vou usar esse nome nesta conversa.",
        "train": ["meu nome é Lucas", "eu me chamo Ana", "pode me chamar de João", "sou a Maria", "me chama de Pedro", "meu nome é Carlos", "eu sou Beatriz", "pode me chamar de Rita", "me chamo Diego", "sou o Bruno", "meu nome é Sofia", "eu me chamo Luana"],
        "validation": ["meu nome é Rafael", "me chamo Marina", "pode me chamar de Paula"],
        "test": ["meu nome é Gabriel", "eu me chamo Camila", "pode me chamar de André"],
    },
    "recall_name": {
        "title": "Contexto da sessão", "reply": "Você me disse que seu nome é {name}.",
        "train": ["qual é meu nome", "como eu me chamo", "você lembra meu nome", "diga meu nome", "quem sou eu", "qual nome eu te disse", "lembra como me chamar", "como você deve me chamar", "me fala meu nome", "você sabe meu nome", "eu te disse meu nome", "qual é o nome que falei"],
        "validation": ["você lembra como eu me chamo", "o meu nome é qual", "repita meu nome"],
        "test": ["qual foi o nome que te falei", "lembra do meu nome", "meu nome você sabe"],
    },
    "unknown": {
        "title": "Separar perguntas de conhecimento", "reply": "",
        "train": ["qual o nome do brasil", "qual é a capital do brasil", "o que é python", "como instalar linux", "quem escreveu este livro", "como funciona a fotossíntese", "qual a população da frança", "me explique a história do brasil", "quanto é sete vezes oito", "escreva um programa", "onde fica portugal", "qual moeda usa o japão", "oi qual é a capital da itália", "obrigado mas o que é astronomia", "qual o nome da argentina", "como se chama o autor do livro", "qual é o nome do presidente", "quem é machado de assis", "meu computador não conecta na internet", "a internet é uma rede mundial", "como uma rede neural calcula gradientes", "a memória ram funciona como", "me fale sobre treinar cães", "o site da nasa é qual", "qual a origem da palavra português", "desenhe um mapa", "o que devo comer", "quero informações sobre rios", "qual é o nome de aurória", "qual a capital de boreália", "eae como funciona a gravidade", "eai me conte sobre literatura", "salve qual é a população do brasil", "oi qual é o nome oficial de portugal", "na história a personagem disse oi e saiu", "ele estava triste e chorou 😢", "a personagem sorriu e disse olá", "eae qual é a capital da espanha"],
        "validation": ["qual é a capital da alemanha", "como funciona o sistema solar", "explique programação em python"],
        "test": ["qual o nome oficial do brasil", "quem descobriu o brasil", "onde fica a cidade de brasília", "me conte sobre dom casmurro", "o que é uma árvore", "qual o tamanho de júpiter", "oi qual a moeda do canadá", "o que aconteceu em portugal", "você sabe a população da china"],
    },
}


from .language_styles import EXTRA_TRAIN, EXTRA_VALIDATION, EXTRA_TEST, FEELINGS

# A avaliação anterior permanece disponível para medir retenção nos mesmos 54 itens.
LEGACY_TEST = [{"text": text, "intent": intent} for intent, lesson in LESSONS.items() for text in lesson["test"]]
LESSONS.update(FEELINGS)


def examples(split):
    rows = []
    extra = {"train": EXTRA_TRAIN, "validation": EXTRA_VALIDATION, "test": EXTRA_TEST}[split]
    for intent, lesson in LESSONS.items():
        for text in [*lesson[split], *extra.get(intent, [])]:
            rows.append({"text": text, "intent": intent})
            if split == "train":
                rows.append({"text": text + ", por favor", "intent": intent})
                rows.append({"text": "tac, " + text, "intent": intent})
                # Exemplos explícitos: emojis acompanham a frase, sem mudar o rótulo.
                rows.append({"text": text + " 😊", "intent": intent})
                if intent == "unknown":
                    # Cumprimentos antes de perguntas não devem ocultar seu conteúdo.
                    rows.append({"text": "eae, " + text, "intent": intent})
    return rows
