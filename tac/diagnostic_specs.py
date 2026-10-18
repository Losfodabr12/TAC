"""Descrição verificável do código atual, separada dos registros históricos."""
from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from pathlib import Path


def source_ref(obj):
    obj=inspect.unwrap(obj)
    path=Path(inspect.getsourcefile(obj))
    return {'arquivo':'tac/'+path.relative_to(Path(__file__).parent).as_posix(),
            'funcao':obj.__qualname__,'linha':inspect.getsourcelines(obj)[1],
            'sha256_codigo':hashlib.sha256(path.read_bytes()).hexdigest()}


def rules():
    from .neural.conversation import ConversationEngine,context_ready,natural_ready
    from .neural.training import Trainer
    from .neural.dialogue import DialogueEngine
    from .fluency_lessons import fluency_ready,score_reply
    result={}
    for name,obj in (('dialogo',ConversationEngine.train),('leitura',Trainer.train),('classificador',DialogueEngine.train)):
        tree=ast.parse(textwrap.dedent(inspect.getsource(obj)))
        expressions=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('checks','accepted','selection_loss','target_steps') for t in node.targets):
                expressions.append(ast.unparse(node))
            if isinstance(node,ast.AugAssign) and isinstance(node.target,ast.Name) and node.target.id=='selection_loss':
                expressions.append(ast.unparse(node))
            if isinstance(node,ast.FunctionDef) and node.name=='acceptable':expressions.append(ast.unparse(node))
        result[name]={'fonte':source_ref(obj),'expressoes_codigo_atual':expressions}
    for obj in (context_ready,natural_ready,fluency_ready,score_reply):
        result[obj.__name__]={'fonte':source_ref(obj),'codigo':textwrap.dedent(inspect.getsource(obj))}
    return result


def implementation():
    from .neural.conversation import ConversationEngine,decode_reply
    from .neural.training import Trainer
    from .neural.model import TACTransformer,CausalBlock
    from .neural.bpe import LocalBPETokenizer
    from .neural.data import Corpus
    from .neural.dialogue import features,DialogueNetwork
    from .dialogue_imports import imported_kind
    defaults={k:v.default for k,v in inspect.signature(ConversationEngine.train).parameters.items()
              if v.default is not inspect.Parameter.empty and isinstance(v.default,(int,float,str,type(None)))}
    return {
        'escopo':'Código instalado atualmente. Não preenche parâmetros históricos ausentes com estes valores.',
        'fontes':[source_ref(o) for o in (TACTransformer,CausalBlock,LocalBPETokenizer,Corpus.from_memory,
                                         decode_reply,features,DialogueNetwork,imported_kind)],
        'transformer':{'tipo':'decoder causal próprio, pre-LayerNorm, residual, GELU',
            'atencao':'scaled_dot_product_attention causal; QKV e projeção lineares',
            'dropout':0.,'inicializacao':'Normal(0,0.02), vieses zero',
            'memoria_recorrente':False,'kv_cache':False,
            'memoria':'Contexto de tokens da conversa atual. Biblioteca SQLite/FTS5 é externa aos pesos.',
            'posicao':'RoPE quando rope=true; embeddings posicionais aprendidos nos demais modelos.'},
        'tokenizacao':{'leitura':'UTF-8 por bytes: PAD=0, BOS=1, EOS=2, bytes+3; vocabulário 259.',
            'dialogo':'BPE local com fallback de bytes; tamanho real vem do checkpoint. Nenhum vocabulário externo.',
            'preprocessamento':'prompt_text normaliza espaços e intercala Usuário:/TAC:. Somente tokens da resposta e EOS entram na perda do diálogo.',
            'limite_resposta':'len(encode(" "+answer))+1 deve ser menor que context-4; máximo context-5.',
            'vocabulario_completo':'Não exportado; pode revelar fragmentos dos dados de aprendizado.'},
        'classificador_guiado':{'entrada':'2048 atributos de hashing; não é tokenizador de linguagem autoregressivo.',
            'features':'Palavras peso 2, bigramas peso 1, n-gramas de caracteres 2/3/4 com pesos 0.25/0.5/0.5. Hash blake2b com sinal e normalização L2; nomes substituídos por pessoa nos padrões reconhecidos.',
            'rede':'Linear(2048,128) → GELU → Linear(128,n_classes). Respostas autorais selecionadas após limiar; não gera tokens.'},
        'geracao':{'dialogo':{'padrao':'greedy; escolha experimental controlada em conversation_generation',
                    'greedy_default':True,'max_new_tokens':128,'temperatura_default_model':.7,
                    'temperatura_afeta_argmax':False,
                    'controlada_opcional':{'temperature':.65,'top_k':30,'top_p':.9,'repetition_penalty':1.08,
                                          'seed':'config.seed; penaliza apenas tokens já gerados'}},
                  'leitura':{'greedy':False,'max_new_tokens':100,'temperatura':.7,'seed':42,
                    'top_k':None,'top_p':None,'penalidade_repeticao':None},
                  'fim':'EOS ou limite; PAD/BOS são bloqueados. Sem nova avaliação durante a exportação.'},
        'historico':{'chat':'Core.chat mantém as últimas 6 mensagens e acrescenta a pergunta: até 7 falas no prompt. É estado temporário, não treino.',
                    'tokens':'decode_reply mantém BOS e os últimos context-1 tokens do prompt.',
                    'treino_importado':'Últimas 7 falas; prefixo ainda pode ser recortado para caber junto da resposta.',
                    'avaliacao_sessoes':'Usa respostas geradas pela própria rede; exemplos ouro não são dados como respostas anteriores nesses testes.'},
        'biblioteca':{'consulta':'Recuperação textual FTS5/BM25 e extração por regras; guardar não altera pesos.',
            'leitura':'Corpus.from_memory usa documentos e correções validadas. Divide cada trecho em 80% treino/20% validação, com quotas por fonte.',
            'dialogo':'Biblioteca, histórico autorizado e correções salvas não viram diálogo automaticamente. Importador tem dados próprios. Opcionalmente, prepare textos de fontes escolhidas para os lotes de escrita da rede de diálogo; isso não cria pares pergunta/resposta.',
            'textos_opcionais':'prepare_document_replay cria snapshot separado com prática autoral e até 10% de trechos adicionais escolhidos. Proporção de trechos, não de tokens. Preparar não treina; corpora documentais anteriores não selecionados e fontes research_web são excluídos.',
            'wikipedia':'Consulta documental pode fornecer a resposta; pesquisa não é inferência da rede. Corpus exclui fontes marcadas research_web.'},
        'treino':{'defaults_funcao_dialogo':defaults,'otimizador_transformers':'AdamW, weight_decay=0.01, grad_clip_norm=1',
            'precisao_politica_atual':'fp16 autocast e GradScaler em CUDA; fp32 em CPU. Dtype dos pesos não comprova a precisão histórica das operações.',
            'scheduler_dialogo':'warmup linear × cosseno até fator 0.1; warmup=min(100,max(1,steps//10)), ou min(300,...) no perfil grande.',
            'scheduler_leitura':'Sem warmup/scheduler no código atual. Usa lr configurada no AdamW; uma retomada restaura o otimizador salvo.',
            'taxa_dialogo':'conversation_learning_rate é independente de learning_rate da leitura; padrão novo 0.00002. Retomadas restauram a taxa registrada, sem substituí-la pelo padrão atual.',
            'epocas_dialogo':'Padrão: amostragem balanceada com reposição. Opção deck_v1 embaralha grupos e exemplos sem reposição dentro de seus ciclos. epoch=ceil(step/1000) não representa passagens completas pelo corpus.',
            'loss_dialogo_registrada':'train_loss é o último microlote mostrado, não a média de todo o conjunto. Pode ser outra tarefa ou incluir KL. val_loss é a perda básica, não a soma usada na seleção.',
            'retencao':'Replay e KL para a própria versão anterior congelada. No currículo fluency, a política anterior aplica KL somente nos 15% de microlotes legacy. A força realmente usada está no config de cada execução.',
            'ancora_opcional':'anchor_v1: 0.5 × mistura anterior + 0.5 × (CE antiga + retention_weight × KL) em cada passo. Desligada por padrão; não comprovou aprovação nos pilotos.',
            'retomada':'Execuções novas salvam original_baseline/original_natural_baseline. A retomada compara aprovação com o início original, não com um candidato já degradado. Checkpoints legados sem referência usam dados disponíveis explicitamente identificados.',
            'selecao_importada':'protected-imported-v1 em execuções novas com importações: candidatos com loss básica < 1.02 × referência vêm primeiro; entre eles, menor loss de validação importada e depois soma interna. Teste importado nunca seleciona. Todos os critérios finais anteriores continuam obrigatórios.',
            'mistura_sem_importacoes':{'reviewed':.55,'language':.30,'legacy':.15},
            'mistura_com_importacoes':{'imported':.275,'reviewed':.275,'language':.30,'legacy':.15},
            'classificador':'AdamW lr=0.002, weight_decay=0.01; lotes 64; corte de gradiente 1. Não gera texto.',
            'reforco':'Adam lr=0.02; política 8→48(Tanh)→7; REINFORCE, recompensa 1 se soma correta; entropia 0.01. Avalia as mesmas 16 somas 0..3.',
            'overfitting':'Separações reservadas, weight decay e escolha por validação. Não existe garantia de generalização ou parada por platô dedicada.'},
        'regras_aprovacao_codigo_atual':rules(),
        'avaliacao':{'exatas':'Igualdade textual com answer ou acceptable_answers. Não há julgador semântico.',
            'fluencia':'score_reply usa regras lexicais simples, 3+ palavras, repetição e deriva para slots. Não mede fluência geral.',
            'contexto':'Tarefas fechadas de nome, cor, local, mudança e condição; não comprova memória geral.',
            'negacao':'Há novos conjuntos sintéticos em reports/research_0120 com negação, correção, nomes e contexto. Rubricas literais limitadas; semântica geral exige revisão humana. Não provam compreensão ampla.',
            'importadas':'Os resultados exatos por habilidade continuam informativos e não afrouxam aprovação. Em novas execuções, a loss da validação importada participa da seleção entre versões protegidas. Teste permanece apenas informativo. Execuções históricas mantêm sua política registrada.',
            'transferencia':'Novo teste final congelado, separado dos dados de treino e ainda não consumido nesta fase. Desenvolvimento novo avaliado uma única vez após seleção. Notas semânticas pendentes são null, não zero.',
            'zero':'Zero com amostras avaliadas difere de ausência de métrica ou conjunto vazio. O relatório registra estes estados separadamente.',
            'reutilizacao':'Testes internos foram usados repetidamente no desenvolvimento; não são avaliação cega externa.'}}
