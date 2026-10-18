"""Relatório a partir das respostas efetivamente geradas e dos logs de treino."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from tac.config import atomic_json
OUT=ROOT/'reports/conversacao_v2/rodada_20261018_01';BASE=ROOT/'data/tac_v2_conversation_20261018'


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def cell(text):return text.replace('|','\\|').replace('\n',' ')


def main():
    data=read(BASE/'ready.json');run=read(BASE/'result.json');before=read(OUT/'eval_00000.json');after=read(OUT/'eval_04000.json')
    suite=read(OUT/'expanded_comparison.json');valid=read(OUT/'validacao_final.json');native=read(OUT/'interface_verificada.json')
    examples=[]
    judgments=['Saudação agora apropriada; ganho pontual.',
               'Continua falsa: descreve o lobo como local; não explica o animal.',
               'Continua incorreta: a resposta é 4, não 2.',
               'Não produz uma história coerente nem cumpre a instrução.',
               'Recuperação correta de Carlos em conversa limpa.']
    for b,a,judgment in zip(before['known_probes_not_used_for_selection'],after['known_probes_not_used_for_selection'],judgments):
        assert b['question']==a['question']
        examples.append({'question':a['question'],'before':b['text'],'after':a['text'],'diagnosis':judgment,
                         'before_stop':b['stop_reason'],'after_stop':a['stop_reason'],'type':'known_probe_not_selection'})
    for b,a in zip(before['open_generalization_tasks_for_human_review'],after['open_generalization_tasks_for_human_review']):
        examples.append({'question':a['question'],'before':b['text'],'after':a['text'],'type':'open_development_task',
                         'diagnosis':'Cópia de Lívia correta; outra formulação.' if a['category']=='context' else 'Não certificada como correta; respostas permanecem fracas/incoerentes.'})
    atomic_json(OUT/'ANTES_E_DEPOIS.json',examples)
    table=['| Pergunta | Referência anterior | Experimento 4.000 | Avaliação |','|---|---|---|---|']
    table.extend('| '+ ' | '.join(cell(x[k]) for k in ('question','before','after','diagnosis'))+' |' for x in examples[:5])
    extended=['| Passos adicionais | Aritmética (80) | Contexto (80) | F1 lexical leitura (60) | F1 lexical natural (18) | F1 lexical retenção (24) |',
              '|---:|---:|---:|---:|---:|---:|']
    for x in suite:
        s=x['summary'];extended.append(f"| {x['step']} | {s['arithmetic']['structured_correct']}/80 | {s['context']['structured_correct']}/80 | {s['reading']['reference_word_f1']:.4f} | {s['natural']['reference_word_f1']:.4f} | {s['retention']['reference_word_f1']:.4f} |")
    detail={}
    for step in (0,4000):
        value=read(OUT/f'expanded_{step:05d}.json');groups={}
        for r in value['outputs']:
            if r['category']=='context':
                groups.setdefault(r['skill'],[0,0]);groups[r['skill']][1]+=1
                groups[r['skill']][0]+=int(bool(r['score']['structured_correct']))
        detail[step]=groups
    skills=['| Habilidade contextual | Antes | Depois |','|---|---:|---:|']
    for k in sorted(detail[0]):
        b=detail[0][k];a=detail[4000][k];skills.append(f'| {k} | {b[0]}/{b[1]} | {a[0]}/{a[1]} |')
    periodic=['| Passos | Loss validação por resposta | Loss retenção por resposta | Loss linguagem | Proxy de geração | Passou proteção de retenção? |',
              '|---:|---:|---:|---:|---:|---|']
    periodic.append(f"| 0 | {before['losses']['validation']:.4f} | {before['losses']['retention']:.4f} | {before['losses']['language']:.4f} | {before['selection_proxy']:.4f} | Referência |")
    for x in run['evaluations']:
        l=x['losses'];periodic.append(f"| {x['step']} | {l['validation']:.4f} | {l['retention']:.4f} | {l['language']:.4f} | {x['score']:.4f} | {'Sim' if x['eligible'] else 'Não'} |")
    text='''# Continuação real de conversação — TAC v2 — 18/10/2026

Executada uma única continuação dos **10.004.544 parâmetros existentes**, com **4.000 atualizações** em CUDA/RTX 3050, sem reconstrução. O orçamento era de até 8.000; a parada antecipada ocorreu porque nenhum checkpoint passou pelo critério de retenção, após pelo menos três avaliações sem melhora elegível e o mínimo de 4.000 passos.

**Resultado parcial: a conversação geral coerente ainda não foi atingida.** Melhoraram algumas tarefas de cópia/contexto e a saudação conhecida. Aritmética, explicações, histórias e mudanças de assunto continuam falhando. Houve regressão em retenção e em recuperação do nome atual. O candidato está disponível na interface como **“Experimento — não aprovado”**, com aviso e seletor para a referência anterior. Não é aprovação de melhoria geral.

## Execução e preservação

- Referência preservada: `64f45f52fee54053a8f0c6b16d9df172`, passo original 14.247. SHA-256 `5291a1c9a3768baec4a1f3f2279bfb90cb170ee716fc101e2fda3266ccc2b311`, idêntico ao fechamento anterior.
- Selecionado para experimentação: `aeed01e3d5cf46c287e3a8f719d50587`, 4.000 passos adicionais, total 18.247. Hash de parâmetros `714db89d53bc78ca594cce556006fd3d1d76da185a24164ab9a3a2ee2aec9ea5`. O último checkpoint recuperável da mesma função/pesos é `77e829c696d2466b95c900988276933b`.
- Uma execução, run `d03a4e71570e495284b9c53da1ca1623`. **30.400 apresentações de 9.306 exemplos únicos**, 1.902.731 alvos de resposta e 409.600 tokens de repetição de linguagem FineWeb2. Cada exemplo de instrução apareceu **3 ou 4 vezes**; não houve superamostragem de poucas saudações. Foram salvos **10 checkpoints novos**, incluindo início, intervalos de 500 e fechamento. Nenhum anterior foi excluído.
- Mesmo Transformer e tokenizer da referência, contexto 1.024. Continuação a partir dos pesos carregados, com AdamW novo para a fase: LR máximo 1e-4, warmup 100, cosine até 0,15 do máximo, betas (0,9; 0,95), weight decay 0,01, clipping 1. BF16 na RTX 3050 de 6 GB; limite de alocação do processo de aproximadamente 3 GB. Lote efetivo 8, microbatch 4. Uma atualização de linguagem a cada 20.
- Loss aplicada somente à resposta e EOS; prompt/padding mascarados. Nesta rodada, média da CE de tokens válidos por resposta, depois média por exemplo. Essa ponderação difere da média global por token da etapa original e foi declarada no plano anterior ao primeiro passo. Portanto o resultado não isola causalmente o efeito dos dados ou da ponderação.
- Todas as 4.000 perdas e normas de gradientes registradas foram finitas, sem OOM observado. O tempo registrado da fase principal, incluindo avaliações periódicas e excluindo a avaliação inicial, foi aproximadamente 323 s.
- Pesos de referência, corpus original e tokenizer conferidos por SHA-256. Código/interface, biblioteca e documentos pessoais preservados. Nenhum modelo externo, API de geração, RAG, classificador ou corretor de conteúdo foi usado nas respostas avaliadas. Os testes finais reservados não foram abertos.

Evidências: [plano congelado](PLANO.json), [referência](reference.json), [execução](training_result.json), [checagens finais](validacao_final.json).

## Conjunto adicional e revisão

Foram preparados 9.306 exemplos de treino, 1.198 de validação e 366 de retenção, em uma pasta nova, mantendo os dados anteriores. O treino contém Aya (4.631), OASST2 (204), OASST1 adicional sem duplicatas (5), SQuAD português (2.500), diálogos públicos selecionados (31), aritmética calculada em Python (1.331) e extração/contexto novos (604). Aproximadamente 79% dos exemplos vêm dos materiais públicos; o componente calculado/extraído é minoritário. Há 213 exemplos classificados como escrita e 49 como conversação; essa cobertura de conversa natural é limitada e não deve ser confundida com milhares de diálogos completos.

Aya/OASST: Apache 2.0. OASST1 é fonte adicional, mas quase todo o conteúdo português se sobrepunha ao OASST2; duplicatas não foram anunciadas como novas. SQuAD português: repositório de tradução MIT, respeitando os textos/dados SQuAD sob CC BY-SA 4.0. Chatterbot-corpus: BSD 3-Clause; usados apenas dados contribuídos, sem instalar seu motor nem selecionar suas respostas em inferência. A tradução SQuAD já existia; nenhuma tradução por outra IA foi executada. Fontes, revisões e condições completas em [licenças](../../../data/tac_v2_conversation_20261018/LICENSES.md).

O filtro anterior rejeitava textos com menos de oito caracteres, prejudicando respostas como números e cumprimentos curtos. A preparação nova aceita esses casos quando válidos. Foram aplicados filtros de PII, texto quebrado, qualidade anotada, contexto, duplicatas e prompts com alvos diferentes. A remoção desses últimos é conservadora: respostas diferentes nem sempre são contradições factuais. Quizzes repetitivos, traduções ruins detectadas e parte de instruções inadequadas foram excluídos. A leitura foi limitada a 2.500 exemplos para não dominar o conjunto.

Houve inspeção de uma amostra estratificada de **55 registros**, complementada por verificações automáticas e revisão de trechos de cumprimentos. Foram corrigidos rótulos e concordância nos históricos calculados e removidos registros inadequados; isso não certifica todos os fatos públicos. Aritmética tem alvos calculados, e contexto tem alvos extraídos literalmente, sem respostas criadas por um modelo externo. Nenhum exercício da v1 foi reutilizado.

As famílias de formulação do novo treino e da validação são disjuntas; os artigos SQuAD são separados e árvores OASST não são fracionadas entre os dois. As tarefas calculadas reservam formulações inteiras; a extração reserva também um vocabulário de nomes. Essa separação não demonstra que todo nome/conceito seja desconhecido no pré-treinamento original. Exemplos já estudados pela referência não são chamados de generalização inédita e entram no painel de retenção. Não há sobreposição de prompts completos entre o novo treino e a validação. Contagens de famílias foram recalculadas após a curadoria, sem alterar arquivos ou hashes usados no treino.

Evidências: [dados e proveniência](DADOS.json), [amostra inspecionada](amostra_revisao.json), [registro da revisão](REVISAO.json).

## Comparação em geração livre

Todos os checkpoints usaram as mesmas entradas e decodificação: temperatura 0,55, top-k 30, top-p 0,9, penalidade 1,12, seed 42; mesma proteção contra repetição. Os painéis de desenvolvimento usam 96 tokens novos e as sondas abertas até 128, de modo igual entre versões. Cortes por repetição são falhas de geração e não acertos.

Durante o treino, foram avaliados referência e passos 1.000/2.000/3.000/4.000. O painel fixo tinha 50 casos de generalização, além de seis de retenção, cinco sondas conhecidas e oito instruções abertas para inspeção. A seleção automática usava um proxy explícito e proteção de retenção: loss de linguagem até 1,3× a referência e F1 lexical das seis respostas de retenção sem queda superior a 0,05. **Nenhum candidato passou. Os critérios e notas anteriores não foram alterados.**

'''+ '\n'.join(periodic)+'''

A avaliação ampliada posterior comparou referência, 1.000 e 4.000 passos em **262 casos idênticos por checkpoint**: 80 de aritmética, todos os 80 de contexto, 60 de leitura, todas as 18 perguntas públicas inéditas e 24 de retenção, com amostragem por seed fixa. Foram acrescentadas sessões com respostas anteriores geradas pela própria rede. São dados de desenvolvimento, usados para selecionar e descrever candidatos; não constituem teste final independente.

'''+ '\n'.join(extended)+'''

**F1 lexical é sobreposição de palavras com a referência, não taxa de respostas factualmente corretas.** Não existe certificação semântica automática para perguntas gerais e histórias. Em aritmética, são aceitas respostas com redação diferente contendo somente o resultado, ou a equação correta; uma sequência de contas falsas terminando no número esperado é rejeitada. Em contexto, a rubrica verifica valores, papéis e ordem; é restrita e pode rejeitar paráfrases válidas. As saídas brutas estão preservadas.

O checkpoint de 4.000 superou o de 1.000 nos quatro indicadores ampliados e teve menos cortes por repetição. Essa comparação fundamenta sua disponibilização como melhor candidato **entre os comparados nessa ampliação**. As diferenças de 1–2 acertos entre os dois não provam vantagem estatística robusta. Como ambos regrediram na retenção, a disponibilização solicitada é experimental e sinalizada; não é aprovação nem reinterpretação do critério anterior.

O agregado de contexto esconde uma regressão importante no nome atual:

'''+ '\n'.join(skills)+'''

Evidências: [evolução](evolucao.json), [comparação ampliada](expanded_comparison.json), [referência completa](expanded_00000.json), [candidato completo](expanded_04000.json), [decisão de disponibilidade](SELECAO.json).

## Respostas reais antes e depois

Conversas limpas e configuração idêntica; resultados confirmados pelo envio assíncrono real da janela Windows. As sondas conhecidas abaixo não entraram no proxy de seleção.

'''+ '\n'.join(table)+'''

Na tarefa aberta “Pode me chamar de Lívia. Qual nome acabei de informar?”, a referência respondeu “O nome de Lívia é uma decisão que representa a decisão do juiz que não poderia ser feito com a decisão.”; o candidato respondeu **“Lívia.”**. É evidência pontual de recuperação com outra formulação, sem demonstrar compreensão geral.

Todos os oito casos abertos adicionais e seus resultados permanecem em [antes e depois](ANTES_E_DEPOIS.json). As respostas ruins também foram preservadas. Não foi executada correção por frase fixa.

## Histórico e interface

Instâncias de `Core` mantêm históricos distintos; limpar a conversa remove o histórico da entrada neural. Não foi encontrado vazamento entre instâncias ou persistência oculta da sessão limpa. Foi demonstrado um caso separado: trocar o checkpoint conservava respostas geradas pelo modelo anterior no histórico. O Core agora reinicia a sessão ao detectar outra versão, e o seletor da interface limpa também a transcrição. A ação **Nova conversa** ficou visível.

Dentro da mesma conversa, permanece um defeito da função aprendida: depois de “Opaa, como vai você?”, a pergunta sobre lobo recebeu “Opaa, para você, etc.”. Na sessão com nomes, a rede recuperou “Arlete”, mas ao receber “Agora meu nome é Noemi.” continuou respondendo “Arlete”; à pergunta seguinte sobre nome atual respondeu “Claro”. Os históricos efetivos contêm as mensagens corretas. Logo, a montagem da sessão não explica essa confusão remanescente.

Foram verificados na janela Windows o envio das cinco perguntas, o carregamento do checkpoint escolhido, a troca para a referência e a volta ao experimento, com história limpa. No mesmo prompt Carlos, a referência respondeu “O meu nome é meu nome.” e o experimento “Carlos”. O aviso de regressão está visível. [Registro nativo](interface_verificada.json) e [captura](interface_conversacao.png).

## Verificação, limites e encerramento

**54 testes de software passaram**, incluindo desligamento do aprendizado durante backward sem atualização, retomada exata de otimizador/amostragem em cópia pequena, prevenção de execução simultânea e de repetição silenciosa de treino concluído, isolamento de sessões e troca das versões. Esses testes não atestam inteligência. Um teste antigo esperava consulta documental no modo neural; sua chamada foi corrigida para o modo documental, mantendo a verificação separada de que o modo neural não usa RAG. [Resultado dos testes](pytest_final.xml).

O aprendizado alterou de fato os pesos, mas **não atingiu o objetivo de responder de forma coerente a perguntas simples em geral**. Existem ganhos em saudação e algumas recuperações, matemática segue em 6,25% nessa amostra, o nome atual piorou e a retenção caiu. Não se atribui causalidade única ao tokenizer, arquitetura ou quantidade de parâmetros: a rodada mudou dados, objetivo de ponderação e iniciou um otimizador de fase, sem experimento de ablação. O treinamento não foi ampliado após a parada justificada.

Próxima prioridade recomendada, sem execução automática: uma continuação concentrada em conversas naturais completas e manutenção de contexto com respostas próprias, protegendo a retenção e reduzindo a predominância de alvos muito curtos; comparar com esta referência e medir geração livre. Não há demonstração nesta rodada de que aumentar parâmetros ou reconstruir a rede resolveria o problema. O experimento atual e a referência estão disponíveis para comparação, com as limitações expostas.
'''
    (OUT/'RELATORIO.md').write_text(text,encoding='utf-8')
    (OUT/'FALHAS.md').write_text(f'''# Falhas observadas após a continuação

- Aritmética: somente 5/80 acertos; “Quanto é 2 + 2?” → “2”.
- Nome atual: 4/11 → 1/11, apesar da melhora agregada de contexto.
- Retenção: F1 lexical nas seis sondas fixas {before['retention_free_word_f1']:.4f} → {after['retention_free_word_f1']:.4f}. Na amostra ampliada de 24, 0,2056 → 0,1570.
- Definições: lobo tratado como local; explicações sobre chuva e energia continuam sem sentido.
- Escrita: pedido de história recebe frase falsa sobre uma igreja, sem narrativa.
- Multiturno: assunto e nome anterior dominam respostas, mesmo com histórico corretamente montado.
- Critério de aceitação: nenhum checkpoint elegível. Redução da loss não aprovou candidatos.
- Cobertura: poucas conversas naturais/saudações e poucos exemplos abertos de escrita na validação; revisão de amostra não certifica todo o corpus.

Os registros brutos e os resultados negativos permanecem no relatório e nos JSON. Nenhum erro foi substituído por resposta preparada.
''',encoding='utf-8')


if __name__=='__main__':main()
