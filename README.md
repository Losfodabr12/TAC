<!-- TAC-V3-STATUS -->
# TAC v3 — treinamento concluído, qualidade insuficiente

**Publicação no GitHub:** somente código, scripts e testes. `data/`, `reports/`, pesos, bancos, materiais e configurações locais ficam fora do repositório. Os links de resultados abaixo se referem à instalação local. Veja [PUBLICACAO_GITHUB.md](PUBLICACAO_GITHUB.md) para o escopo e os limites de uma clonagem nova.

Nova linha separada: **29.414.400 parâmetros** próprios, treinados desde pesos aleatórios em CUDA/BF16. Concluídas **20.696 atualizações** em aproximadamente 98 minutos: 120 milhões de tokens de pré-treinamento, replay adicional e duas apresentações de cada uma das 21.770 instruções. Os 11 checkpoints novos são recuperáveis e foram conferidos; os 96 arquivos originais da v2 verificados permanecem iguais.

**O objetivo de conversação coerente não foi atingido.** Na revisão das 27 tarefas novas, a primeira passagem supervisionada atende integralmente 1, o final 0 e a v2 de conversação 2. Acertos parciais de cópia/resumo ficam separados; a melhora de previsão de linguagem não é aprovação de compreensão. Nenhum candidato substituiu a v2.

Para comparar: **INICIAR_TAC.bat → Conversar → TAC v3 — experimental, 29,4 milhões**. A v2 de conversação e a referência anterior continuam no seletor; trocar a versão limpa a conversa. Interface nativa verificada com geração neural real, sem respostas fixas, biblioteca ou API externa. Não há outro treinamento em andamento nem extensão automática do orçamento.

**[Relatório final, resultados e limitações](reports/tac_v3/rodada_20261018_01/RELATORIO.md)**; [comparação de todas as respostas](reports/tac_v3/rodada_20261018_01/COMPARACAO_MODELOS.md); [falhas](reports/tac_v3/rodada_20261018_01/FALHAS.md); [estado concluído](reports/tac_v3/rodada_20261018_01/ESTADO_EXECUCAO.md). Fontes e condições: [LICENSES.md](data/tac_v3/LICENSES.md). Resultado do orçamento em `data/tac_v3/training_result.json`; `live.json` contém apenas a última amostra periódica, anterior ao encerramento. Uma próxima hipótese está recomendada no relatório, sem execução automática.

---

<!-- TAC-V2-CONCLUIDA -->
# TAC v2 experimental

Continuação de conversação em 18/10/2026: **4.000 atualizações reais** sobre os pesos existentes, com 9.306 exemplos únicos. O checkpoint experimental está disponível no seletor do chat, junto da referência anterior preservada. Contexto passou de 15/80 para 53/80, mas matemática ficou em 5/80, o nome atual piorou e houve regressão de retenção. **O objetivo de conversação geral coerente ainda não foi atingido.** A interface identifica o experimento como não aprovado e permite comparar as versões em conversas limpas. [Resultados, respostas reais e limitações](reports/conversacao_v2/rodada_20261018_01/RELATORIO.md).

Reconstrução concluída: **10.004.544 parâmetros novos**, pré-treinamento FineWeb2 em português e ajuste Aya/OASST2. Abra `INICIAR_TAC.bat` e escolha **Conversar com a TAC v2 — experimental**. A conversa neural usa a rede própria; a biblioteca continua preservada no modo documental.

Execução: 14,247 atualizações confirmadas em CUDA, com checkpoints recuperáveis da geração nova. Relatório e limitações: [fechamento](reports/reconstrucao_zero/execucao_v2/FECHAMENTO.md). Fontes/condições: [licenças](data/tac_v2/LICENSES.md). A geração livre observada ainda é fraca, com repetições e erros factuais; a versão serve para testes e não tem boa conversação certificada.

O treino automático antigo está desligado. O botão de retomada v2 e `scripts/train_tac_v2.py` continuam somente a execução nova. Nenhum exercício antigo é a base desse treino. Foram excluídos apenas os artefatos generativos antigos autorizados; livros, documentos, biblioteca e programa foram preservados.

Correção de geração em 18/10/2026: a conversa v2 agora interrompe ciclos detectados, informa respostas parciais e usa no máximo 128 tokens novos. O perfil conservador foi ativado; os pesos continuam iguais. As cinco sondas em conversas limpas produziram saídas distintas, mas persistem erros graves de compreensão e contaminação pelo histórico. Não houve novo treinamento. Evidências e limites: [correção de geração](reports/correcao_geracao_v2/rodada_20261018_01/RELATORIO.md).

---

# TAC — Inteligência Artificial com Aprendizado Contínuo

Versão 0.13.0. Aplicação desktop local para estudo documental, consulta com fontes e experimentos neurais reproduzíveis. Criada para Windows 10/11, Python 3.13, 16 GB de RAM e RTX 3050 de 6 GB. Há caminho CPU funcional e aceleração CUDA.

## Generalização de nomes e correções

Foram executados **6 pilotos de 120 passos**, com três seeds e duas formas de supervisão, usando cópias do ativo original de 10.004.544 parâmetros. O dataset sintético tem 928 exemplos em 256 conversas; nomes e roteiros da validação principal são separados do treino. O teste novo de 56 exemplos e os testes reservados anteriores continuam fechados.

O treino normal passou de 3,33% para **36,67%** de acerto factual restrito na validação com nomes e roteiros inéditos (média de três seeds). No painel de situações estudadas, chegou a 94,25%. Dar peso 3 aos tokens dos nomes piorou a média inédita para 33,89%, portanto essa variante não foi incorporada. A cópia isolada de nomes melhorou, mas as conversas ainda falham sobretudo em recuperar e distinguir nomes antigos/atuais. A coerência é medida por uma regra factual limitada, não por um avaliador semântico geral.

**Não houve promoção, treino longo, mudança de arquitetura/tokenizer ou integração ao treino principal.** O chat normal segue com o modelo anterior. O candidato da seed mediana fica separado nos relatórios, com pesos próprios e sem estados do otimizador para retomada exata. **45 testes automatizados passaram**, incluindo separação dos dados, máscara/EOS, preservação de caracteres, geração com histórico próprio e bloqueio de atualizações ao desligar o aprendizado.

Relatório: **[reports/generalizacao_nomes_0130/RELATORIO.md](reports/generalizacao_nomes_0130/RELATORIO.md)**. Comparação por seed: **[COMPARACAO.md](reports/generalizacao_nomes_0130/COMPARACAO.md)**. Sondas e exemplos de falhas: **[FALHAS.md](reports/generalizacao_nomes_0130/FALHAS.md)**. Gráfico: **[generalizacao.png](reports/generalizacao_nomes_0130/generalizacao.png)**. Objetos, negação e preferências ficam para depois de demonstrar generalização consistente em nomes.

## Experimento com as 36 conversas fornecidas

A cópia de 10.004.544 parâmetros treinada **somente com `TAC_TREINO_CONVERSAS.jsonl`** reproduziu 85/85 respostas estudadas após 120 passos, contra 1/85 do ativo. Também acertou 85/85 usando suas próprias respostas anteriores no histórico. O treino parou no primeiro bloco; não misturou exercícios internos, documentos ou RAG. Os arquivos originais, configurações, datasets e todos os checkpoints do diretório de dados foram verificados por hash e preservados. O teste reservado não foi interpretado nem usado.

**Isso mede aquisição/memorização de um conjunto pequeno, não generalização.** Não foram avaliadas perguntas inéditas nem retenção das habilidades antigas. O candidato não foi aprovado para produção e o chat normal continua com o modelo ativo anterior. O checkpoint experimental guarda pesos e configuração, sem estados do otimizador para retomada exata.

Para testar manualmente, feche a janela principal e abra **`TESTAR_APRENDIZADO_EXPERIMENTAL.bat`**: opção 1 usa o candidato; opção 2 compara com o ativo. Digite `/novo` para começar outra conversa e `/sair` para fechar. Esse chat gera diretamente pela rede, sem busca e sem gravar suas falas.

Relatório, comparação por resposta, curvas, cópia compatível e pesos experimentais: **[reports/aprendizado_conversas_0130/RELATORIO.md](reports/aprendizado_conversas_0130/RELATORIO.md)**. Os testes novos verificam adaptação, separação de grupos, máscara, amostragem e interrupção de atualizações quando o aprendizado é desligado.

## Pilotos de preservação de habilidades — 0.13.0

Esta versão acrescenta experimentos isolados de objetivo de resposta, exposição às conversas importadas e cobertura da revisão de habilidades antigas. A rede continua com **10.004.544 parâmetros**. O treinamento normal, o vocabulário, a geração do chat e os critérios de aprovação existentes foram preservados.

Os pilotos usam o mesmo ponto de partida e alteram uma variável por comparação. A instrumentação registra exemplos únicos, repetições, tokens supervisionados, gradientes de todas as camadas e a direção real das atualizações do AdamW. Nenhum piloto pode promover um modelo automaticamente. Os testes finais reservados não são utilizados para escolher variantes.

Abra **INICIAR_TAC.bat → Ferramentas avançadas → Treinamento e versões → Ver resultados dos experimentos de conversa**. O painel traz gráficos, resultados por execução, exposição aos dados e os motivos das proteções não atingidas. Ele apenas lê os relatórios locais; não inicia treinamento. Os relatórios anteriores continuam disponíveis.

Resultados e estado da rodada: **[RELATORIO.md](reports/experimentos_0130/RELATORIO.md)**. Tabela: **[COMPARACAO.md](reports/experimentos_0130/COMPARACAO.md)**. Causas e limites da evidência: **[PROBLEMAS_CONFIRMADOS.md](reports/experimentos_0130/PROBLEMAS_CONFIRMADOS.md)**. Não se atribui nota semântica automática a respostas abertas; a ficha de revisão mantém os campos não avaliados como `null`.

O runner de pesquisa é `.venv\Scripts\python.exe -X utf8 scripts\run_controlled_experiments.py`. Exige aprendizado autorizado, impede execução simultânea com a aplicação e respeita os limites de recursos. As execuções concluídas são preservadas. Checkpoints experimentais de pesos ficam fora do cadastro de modelos ativos; não contêm momentos para retomada exata. Configurações, biblioteca e datasets originais não são alterados.

Rodada concluída: **15 pilotos × 120 passos, três seeds, nenhum promovido**. Todos ficaram em 128/142 respostas básicas, contra 131/142 do ativo, e 0/60 importadas exatas no painel de desenvolvimento. A variante de objetivo melhorou o encerramento em uma sonda de respostas ouro, sem melhora consistente na geração livre. O balanceamento aumentou exemplos únicos, mas não resolveu a falta de variedade por habilidade. **217 testes automatizados passaram**, e a abertura pelo BAT foi verificada. O modelo ativo permanece intacto; não houve treinamento longo posterior. Para reproduzir sem substituir resultados, use `--output reports/experimentos_0130_repeticao`.

## Fase de generalização — 0.12.0

**O modelo ativo de 10.004.544 parâmetros foi preservado. Nenhum dos seis pilotos novos foi aprovado.** Esta fase melhorou a auditoria, o controle dos experimentos e a proteção do aprendizado; ainda não produziu uma rede capaz de conversar livremente. O conjunto final novo de avaliação continua reservado, sem execução.

- Taxa de aprendizado da conversa independente da escrita, com padrão conservador de **0,00002**. Retomadas mantêm seus parâmetros antigos.
- Opções experimentais de repetir habilidades antigas em cada passo, embaralhar sem reposição e gerar com sorteio controlado. Começam desligadas; os pilotos não justificaram ativá-las automaticamente.
- Validação importada passa a participar da escolha entre candidatos protegidos nas **novas** execuções. Todos os critérios anteriores de aprovação continuam obrigatórios. Teste importado não escolhe candidatos.
- A retomada de novos treinos conserva a referência de avaliação original, evitando comparar uma versão degradada com ela mesma.
- Auditoria de duplicatas preservando operadores, acentos e histórico; sem apagar dados. Novo registro separa perdas por objetivo e a retenção, facilitando investigar interferência.
- Prática autoral opcional de nomes, negação, correções e localização. Textos escolhidos da Biblioteca podem ser preparados, separadamente e com autorização, para os lotes de escrita da rede de conversa. Preparar não treina; guardar material continua sendo apenas memória documental.

Abra **INICIAR_TAC.bat → Ferramentas avançadas → Treinamento e versões → Ver resultados dos experimentos de conversa**. As taxas e estratégias ficam em **Ajustes e manutenção**. Os relatórios mostram regressões; não há ativação oculta de candidatos. As alterações entram em uma janela aberta novamente pelo BAT.

Diagnóstico das causas e protocolo: **[DIAGNOSTICO_E_PLANO.md](reports/research_0120/DIAGNOSTICO_E_PLANO.md)**. Resultados, limites, caminhos do código e próximos experimentos: **[RELATORIO.md](reports/research_0120/RELATORIO.md)**. As capturas estão na pasta `reports/research_0120/capturas`.

## Exportar diagnóstico técnico

Ative **Ferramentas avançadas → Treinamento e versões → Exportar diagnóstico completo**. Escolha onde salvar o ZIP, inclusive fora da pasta TAC. A coleta ocorre em segundo plano, funciona com o aprendizado desligado e não exige um treinamento em andamento. Aguarde a atividade atual terminar antes de exportar.

O pacote contém `resumo.md`, `arquitetura.json`, `treinamento.json`, `checkpoints.json`, `avaliacoes.json`, `exemplos_de_respostas.json`, `datasets_estatisticas.json` e `logs.txt`. Um `manifesto.json` adicional permite conferir tamanhos e hashes SHA-256. Todos os JSON são UTF-8 e válidos, com valores ausentes representados por `null` ou estado explícito.

A exportação lê metadados dos checkpoints em CPU e snapshots existentes. **Não treina, não gera respostas, não executa testes novos e não altera pesos.** As avaliações e os resultados de testes de software são históricos e identificados como tal. O diagnóstico inclui referências às funções e regras atuais do código, separadas das decisões históricas.

**Privacidade:** não inclui pesos, SQLite, livros, documentos, histórico do chat ou texto de conversas importadas. Textos que não coincidam com os exercícios autorais conhecidos são substituídos por marcadores anônimos. Nomes de materiais, caminhos pessoais, campos de credenciais e mensagens livres dos logs são omitidos. Os logs preservam classes de exceção, categorias literais e referências verificáveis ao código. A chave dos marcadores não é exportada. Não existe envio automático do ZIP para serviços externos.

As respostas privadas ou não verificáveis ficam anonimizadas, mesmo se isso reduzir a análise textual de alguns erros. O pacote mantém contagens e notas originais; não recalcula acertos depois da anonimização. Ausência de avaliação, conjunto vazio e zero medido são estados diferentes. Contagens de entradas repetidas não comprovam ausência de vazamento semântico. Picos históricos de RAM/VRAM que não foram registrados permanecem indisponíveis.

Guia detalhado e validação: **[reports/diagnostico_110/RELATORIO.md](reports/diagnostico_110/RELATORIO.md)**. Alternativa opcional por terminal: `.venv\Scripts\python.exe scripts\export_technical_diagnostic.py --output TAC_Diagnostico.zip` (recusa substituir um arquivo existente, a menos que `--overwrite` seja informado).

## Importar conversas para ensinar a conversar

1. Abra **INICIAR_TAC.bat → Melhorar a TAC → Conversas de exemplo → Importar conversas**.
2. Escolha **TXT ou JSONL** no formato indicado em **Ver formatos e exemplos**. Confira as quantidades, avisos e rejeições antes de importar. Importar guarda exemplos; não inicia treino.
3. Clique em **Ensinar a conversar com os exemplos escolhidos**. A TAC mostra quantas conversas suas e quantos exercícios do programa serão usados antes de pedir para iniciar.
4. No resultado, confira a decisão e a versão que continua em uso. A comparação de respostas exatas por habilidade nas suas conversas reservadas de **validação** e **teste** é **somente informativa**. Na versão 0.12, a perda de validação importada ajuda a selecionar candidatos protegidos em novas execuções; não afrouxa a aprovação. Teste fica fora da seleção.

Formato completo, limites e funcionamento: **[docs/CONVERSAS.md](docs/CONVERSAS.md)**. Arquivos prontos para editar: **[JSONL](examples/conversas_exemplo.jsonl)** e **[TXT](examples/conversas_exemplo.txt)**. Os dois contêm as mesmas 3 conversas, com 7 respostas; são exemplos de formato, não um curso de conversação.

As conversas importadas são misturadas aos exercícios antigos e à prática de escrita, mantendo a retenção da versão anterior. Cada `group` fica inteiro em treino, validação ou teste, inclusive em arquivos diferentes. Retomadas preservam os dados daquela execução. Os critérios de aprovação anteriores continuam intactos. Um treino rejeitado não troca a versão ativa.

O importador não usa automaticamente Biblioteca, correções ou histórico salvo como diálogo. Importar exige um modelo de conversa local aprovado com o vocabulário BPE, já presente nesta instalação. Nenhum treino novo é iniciado ao abrir a aplicação por causa deste recurso.

Resultados e verificações desta entrega: **[reports/importador_100/RELATORIO.md](reports/importador_100/RELATORIO.md)**.

## Interface reorganizada

Abra **INICIAR_TAC.bat**. A primeira tela é **Início**, com o quadro **Onde cada coisa vai** e o assistente **Como começo?**.

- **Conversar:** escolha **Conversar com a TAC** ou **Consultar meus materiais**. Cada resposta informa sua origem: rede da TAC, documentos, Wikipédia ou resposta preparada. A consulta aos seus textos não ensina nada à rede.
- **Meus materiais:** adicione arquivos ou sites, clique em **Guardar materiais agora** e consulte os conteúdos na guia **Guardados**. Guardar conteúdo não treina a conversa. Arquivos aceitos: PDF, TXT, DOCX, Markdown e HTML. A mensagem para JSONL aponta para o importador de conversas. Um TXT que parece conter conversas oferece o caminho para ensinar a conversar.
- **Melhorar a TAC:** o treino de escrita usa textos guardados e correções autorizadas; o treino de conversa usa os exercícios internos e as conversas importadas habilitadas. Cada cartão explica seu efeito e apresenta duração observada quando há histórico suficiente. **Conversas de exemplo** permite importar, conferir formatos e escolher os arquivos do próximo treino.
- **Preferências:** aparência, histórico autorizado e cópias de segurança. A opção **Consultar a Wikipédia quando faltar informação** é a mesma do Chat, salva em uma única configuração e sincronizada nas duas telas.
- **Ferramentas avançadas:** preservam parâmetros, métricas, gráficos, checkpoints, retomada/restauração, treino de respostas guiadas, monitoramento, ajustes, automação e reforço. A continuação de texto fica acessível ao ativar essas ferramentas.

Depois de um treino, o resultado explica aprovação/rejeição, motivo, comparações disponíveis e versões efetivamente em uso. A reorganização inicial da interface não alterou o treinamento. A versão 0.10 adiciona o importador descrito acima, preservando os critérios anteriores. Nenhuma dessas métricas comprova conversa livre ou compreensão geral.

Capturas e verificação desta alteração: **[reports/interface_090/RELATORIO.md](reports/interface_090/RELATORIO.md)**. Se a janela antiga estiver aberta, feche e abra novamente pelo BAT para carregar a interface atualizada.

As seções seguintes preservam o histórico das entregas e seus resultados. Nomes antigos de telas devem ser interpretados pelo guia atual acima.

## Consultar a Wikipédia — versão 0.7

No Chat, **Consultar Wikipédia quando faltar informação** habilita uma consulta em segundo plano para perguntas públicas curtas que não tenham resposta com fonte na biblioteca. Nesta instalação, a opção foi autorizada pelo usuário; em instalações novas começa desligada. Requer **novos estudos permitidos**. O botão **Consultar Wikipédia** permite informar o assunto/título; **Cancelar pesquisa** interrompe a operação. Nenhuma consulta é iniciada só por abrir o programa. O histórico fica em **Mais opções → Consultas à web**.

Esta primeira versão **consulta artigos pelo título e referências citadas no artigo**, sem chave. Não é um buscador geral nem consulta o Google. Exemplo: `O que é fotossíntese?`. Perguntas fora dos formatos reconhecidos podem usar o botão com um título mais preciso. URLs de artigos, dados de acesso, data da consulta e trechos ficam na biblioteca. Histórico, documentos e arquivos pessoais não são enviados à pesquisa; o envio automático aceita somente assuntos curtos em padrões limitados e bloqueia indicadores de dados pessoais. O usuário deve usar o campo manual apenas para assuntos públicos.

A TAC tenta consultar referências externas públicas dentro dos limites configurados. Mostra **fonte única**, **concordância textual**, **valores divergentes** ou **sem comparação factual**. A comparação automática de valores limita-se aos campos explícitos já extraídos, como capital e nome oficial; outros assuntos não recebem uma confirmação semântica. Sites diferentes podem copiar a mesma informação, por isso concordância não é prova de verdade nem de independência editorial. As respostas continuam documentais e vêm identificadas como consulta à web.

Os textos pesquisados **não entram automaticamente no treino neural**, inclusive após atualizar a fonte. Os pesos do modelo de 10.004.544 parâmetros permanecem os mesmos. A consulta real à Wikipédia e a recuperação posterior com aprendizado desligado foram verificadas. Consulte **[reports/CONSULTA_WEB_070.md](reports/CONSULTA_WEB_070.md)** para os testes, limitações e arquivos.

## Diálogos variados e correções — versão 0.6

O gerador continua com **10.004.544 parâmetros** e recebeu duas rodadas de treino real na GPU, com 8.000 e 12.000 passos. A segunda rodada foi aprovada; foi selecionado seu checkpoint do passo 8.000 pela perda combinada de validação. O conjunto combinado passou a **82.512 exemplos autorais/sintéticos de treino**. Inclui jeitos formais e informais de falar, abreviações, emojis, dúvidas e conversas sobre escrita, desenho, escola, jogos e programação. Correções de nome, cor e assunto agora supervisionam tanto a confirmação imediata quanto a resposta à pergunta seguinte.

Nos novos exercícios reservados, a versão anterior passou de **7/65 para 58/65** na validação e de **7/62 para 47/62** no teste separado. Em conversas que usam as respostas reais da própria TAC como histórico, acertou **19/19** confirmações imediatas e recuperações na validação; no teste separado, **14/18** confirmações e **16/18** recuperações. Esses números medem tarefas pequenas com respostas autorais de referência, não fluência ou compreensão geral. O teste básico anterior teve pequena queda de 125/144 para 123/144; os checkpoints anteriores foram preservados.

O chat real em CPU confirmou as mudanças **azul → verde**, **Pedro → Ana** e **jogos → música**, inclusive ao perguntar novamente a informação. Isso é memória da conversa atual: não altera pesos por mensagem nem grava automaticamente preferências permanentes. Há erros em formas e assuntos novos, especialmente na troca de assunto, ajuda e passatempos. **116 testes do software passaram**, incluindo retenção e retomada exata em CPU. Veja **[reports/DIALOGOS_E_CORRECOES_060.md](reports/DIALOGOS_E_CORRECOES_060.md)** para arquivos, resultados, regressões e limites.

## Diálogo neural: marco de 10 milhões

O novo perfil possui **10.004.544 parâmetros únicos**, 5 blocos causais, embeddings de dimensão 384, 6 cabeças, MLP de 1.728 neurônios, posições relativas RoPE, contexto de **512 tokens** e vocabulário BPE local de 1.024 tokens. Os pesos foram inicializados no projeto. O BPE é treinado nos exercícios locais e inclui fallback de bytes para acentos e emojis; não usa modelos ou vocabulários baixados. Embeddings de entrada e saída compartilham os mesmos parâmetros. A primeira variante grande, também preservada, tinha 10.016.592 parâmetros e posições aprendidas.

Há três modos distintos no seletor do Chat:

- **Consultar biblioteca / conversa guiada:** pesquisa textos com fontes ou escolhe uma resposta pelo classificador de intenções.
- **Conversar com a rede neural (experimental):** gera texto a partir dos pesos de diálogo e das últimas falas. Não consulta uma tabela de respostas durante a inferência. Fica disponível depois de aprovar um checkpoint de diálogo; é selecionado na abertura quando existe uma versão aprovada.
- **Continuação de texto (experimental):** usa o leitor de bytes anterior, preservado separadamente, com 495.808 parâmetros.

Para fatos dos documentos, escolha o modo biblioteca ou **Biblioteca → Usar no chat**. A geração experimental pode inventar, confundir nomes e perder contexto. O histórico neural é temporário, limitado às últimas seis mensagens e aos 512 tokens do modelo grande; reiniciar a aplicação o limpa. Isso não apaga a biblioteca ou os pesos.

**Ferramentas avançadas → Treino avançado → Treinar diálogo e contexto · 10 milhões** executa treino supervisionado de respostas. Configurações separadas controlam passos/duração de diálogo. Lote, acumulação, RAM, VRAM e CPU também são respeitados; pausa, interrupção e retomada ficam na mesma tela. A preparação linguística com replay textual pode ser executada pelo script manual, conforme o relatório. Abrir o programa não inicia esse script.

Os exercícios são autorais e sintéticos, com separações reservadas para validação e teste. Incluem apresentação, lembrança de nomes, preferências, mudança de informação, fatos curtos de contexto, deslocamentos, uma regra condicional e somas/subtrações pequenas. Métricas de reprodução exata desses exercícios **não medem conversa livre nem compreensão geral**. O treino de resposta supervisiona somente os tokens da resposta; textos dos materiais locais são repetidos separadamente para previsão de linguagem. Checkpoints guardam também otimizador, tokenizador e RNG.

Na entrega **0.5**, foram executadas 39 mil atualizações nas rodadas grandes. O checkpoint então aprovado acertou **125/144 (86,8%)** no teste básico reservado, contra 90/144 da variante pequena anterior. Acertou 56/56 tarefas contextuais preparadas, mas falhou em uma mudança de preferência formulada de outro jeito e acertou apenas 71/128 operações aritméticas. Naquela entrega passaram **110 testes do software**. Esse checkpoint foi preservado; a versão ativa atual e seus resultados estão descritos na seção 0.6 acima.

Consulte **[reports/TREINO_10M.md](reports/TREINO_10M.md)** para os resultados da entrega 0.5. Os relatos abaixo descrevem as versões anteriores preservadas; seus números não devem ser somados à capacidade do gerador de diálogo. Os resultados atuais estão no relatório 0.6 acima.

## Interface simples e conversa guiada

As áreas principais agora são **Conversar**, **Ensinar** e **Biblioteca**. Marque **Ferramentas avançadas** para ver o Transformer experimental, monitoramento e configurações. Detalhes técnicos do estado, fila e histórico ficam recolhidos até serem solicitados. As fontes não são despejadas na conversa: clique em **Ver fontes** para conferir a evidência e a origem da última resposta.

Em **Ensinar**, há duas opções explicadas na própria tela:

- **Ler sites e arquivos:** guarda conteúdo para perguntas posteriores. Não altera pesos.
- **Praticar conversas:** executa exercícios que alteram uma rede neural de intenções. Não acrescenta fatos à biblioteca.

O botão **Como funciona?** mostra essa distinção em linguagem simples. A geração livre do Transformer continua experimental e separada dessas duas opções.

Foram preparados **1.854 exemplos de treino em 18 grupos de conversa**, mais uma categoria para perguntas que devem ir à biblioteca. Os exercícios incluem linguagem formal, coloquial e abreviada, emojis, cumprimentos, apresentação, ajuda, memória, correções, esclarecimento e expressões simples de alegria, tristeza ou frustração. São exemplos autorais, não conversas humanas coletadas. Há 98 frases de validação e 100 no teste ampliado, distintas das frases de treino. As **54 perguntas antigas continuam sendo avaliadas separadamente**. O dataset está em `tac/dialogue_lessons.py` e `tac/language_styles.py`; os snapshots auditáveis ficam em `data/datasets/dialogue_*.json`.

Histórias e diálogos podem ser importados como TXT em UTF-8, preservando emojis. Um texto de 400 mil caracteres cabe no limite padrão de 1 milhão por documento. Ler esse arquivo o coloca na biblioteca; o Transformer pode usar amostras dele no treino avançado. O treino de conversa guiada usa os exercícios autorais acima: não converte histórias importadas em pares de conversa automaticamente. Aprender padrões de escrita, prever emojis e compreender sentimentos são capacidades diferentes. Não há compreensão geral de emoções implementada.

Essa rede é um classificador PyTorch próprio, agora com **264.723 parâmetros**: transforma palavras e trechos de caracteres em atributos numéricos e aprende a reconhecer intenções. **Ela escolhe respostas escritas para os exercícios; não gera conversa livre.** No treino desta atualização acertou **49/54** perguntas antigas (90,7%), contra 45/54 antes, e **81/100** no teste ampliado. Não houve falso aceite entre as 15 perguntas de conhecimento do teste ampliado. Esses são benchmarks de desenvolvimento reutilizados, não uma avaliação cega externa. Há erros de classificação e não há garantia para frases arbitrárias.

O modelo de conversa fica em `active_dialogue`, separado do `active_checkpoint` do Transformer. Os checkpoints anteriores continuam carregáveis; ao ampliar as intenções, o treino copia os pesos antigos e inicializa apenas as linhas adicionais da saída. O nome informado pelo usuário pode ser usado na sessão, sem memória permanente automática. Ao reiniciar, esse contexto temporário é apagado; a gravação autorizada de histórico continua obedecendo às preferências.

A divisão dos textos preserva falas e ações próximas. Linhas curtas e nomes como `Ren:` não são tratados automaticamente como títulos. A Biblioteca oferece **Reorganizar contexto** para reconstruir os índices derivados de uma fonte com a permissão de aprendizado ativa, mantendo o texto e a origem. Os identificadores dos trechos podem mudar; snapshots antigos de treinamento permanecem congelados. Nesta atualização, a história do usuário passou de 2.538 para 233 trechos, sem perda de conteúdo.

Selecionar um documento em **Biblioteca → Usar no chat** dá prioridade à consulta desse documento e impede o classificador de desviar perguntas para instruções sobre a TAC. Cumprimentos e agradecimentos curtos continuam funcionando nesse contexto. Perguntas sobre a relação entre dois nomes podem mostrar um trecho em que ambos aparecem, com evidência; isso não é inferência de relações nem análise narrativa geral. Perguntas presentes em roteiros de conversa são ignoradas na extração de fatos como capitais.

A revisão 0.1.2 torna as consultas documentais curtas. Perguntas de nome oficial e capital usam relações explícitas presentes na fonte, sem respostas de países embutidas no código. Outras consultas selecionam uma frase por correspondência textual. Quando a relação não é encontrada ou fontes apresentam valores diferentes, isso é informado. Para visualizar blocos completos, peça **Mostre os trechos sobre…** ou use a Biblioteca. Essas respostas continuam sendo extração documental por regras/seleção lexical; não demonstram compreensão geral da rede neural.

A revisão 0.1.1 esclarece a operação: um painel permanente mostra se há material aguardando, estudo em andamento, fontes prontas ou erros. O botão **Ativar e estudar fila** autoriza e inicia em uma ação explícita. Adicionar um site sozinho prepara a fila; nenhum acesso é feito nessa etapa. A Central mostra as etapas reais de conexão, recebimento, extração e gravação, o resultado final e o botão **Consultar último material estudado**. Tentativas de treino sem corpus ou geração sem checkpoint ficam bloqueadas com explicação. As configurações e os dados existentes são preservados.

## Iniciar

Abra **INICIAR_TAC.bat** com dois cliques. O ambiente virtual e as dependências já estão instalados neste computador. Não é preciso digitar comandos Python.

Se transferir o código para outra máquina, instale Python 3.13 de 64 bits e execute **INSTALAR_TAC.bat**. O instalador baixa bibliotecas matemáticas/interface e o runtime CUDA do PyTorch, sem modelos ou pesos pré-treinados. Uma instalação CUDA pode ocupar vários GB. Se o download CUDA falhar, o instalador tenta a versão CPU. Nenhuma biblioteca é instalada automaticamente pela aplicação.

Use **DIAGNOSTICO_TAC.bat** para conferir dependências, FTS5, tokenização, rede neural e CUDA. **TESTAR_TAC.bat** executa os testes automatizados. A instalação exata está em `requirements-lock.txt`; `requirements.txt` contém as dependências diretas. A versão CUDA instalada é `torch 2.7.1+cu126`.

## Primeiro uso

1. Abra **Conversar** e experimente `oi`, `qual é seu nome` ou uma pergunta sobre seus materiais. Em uma instalação sem modelo de conversa, primeiro use **Ensinar → Praticar conversas**. A instalação atual já recebeu esse treino, conforme os relatórios.
2. Em **Ensinar → Ler sites e arquivos**, abra **Adicionar documento ou exemplo** e clique em **Testar com exemplo local**, selecione documentos ou arraste arquivos. Para sites, use **Adicionar site**, cole a URL no campo de endereço e confirme os termos. Adicionar à fila não estuda o conteúdo nem altera o original.
3. Clique no botão verde **Autorizar e ler materiais**, ou **Ler materiais agora** se a permissão já estiver ativa. O clique autoriza e inicia o estudo; ativar somente a permissão não inicia tarefas. O estado do material passa de **Aguardando início** para **Estudando agora**, e finalmente **Disponível para consulta** ou uma explicação de erro.
4. No **Conversar**, pergunte sobre o material. A resposta fica curta, e a evidência pode ser conferida em **Ver fontes**.
5. Na **Biblioteca**, selecione um documento e clique em **Usar no chat**. Pergunte `O que você aprendeu com esse documento?` para obter uma visão extrativa. **Limpar contexto** volta à pesquisa geral.
6. Para praticar os exemplos de diálogo, use **Ensinar → Praticar conversas**. Para o treino experimental de continuação de texto, marque **Ferramentas avançadas**, abra **Treino avançado** e clique em **Iniciar treinamento da rede**. Esse Transformer usa documentos estudados e correções validadas, separa treino/validação e registra métricas. A permissão de treinamento automático do Transformer também fica nessa área avançada.
7. Para interromper, use **Interromper e salvar** ou desligue o aprendizado. Depois selecione o checkpoint de interrupção e clique em **Retomar selecionado**.

Fechar a janela durante uma tarefa solicita finalização segura e aguarda o término. Uma requisição em andamento pode levar alguns segundos até atingir seu timeout. Após encerramento forçado do processo, a fila é recuperada na próxima abertura; o treino retoma do último checkpoint já gravado, com possível perda dos passos posteriores a ele.

## Funcionalidades implementadas

- Interface PySide6 com seis áreas: Chat, Central de Aprendizado, Biblioteca, Treinamento Neural, Monitoramento e Configurações. Tema escuro padrão, tema claro e tamanho de fonte configuráveis.
- Estudo local de TXT, Markdown, HTML, DOCX e PDFs textuais, incluindo tabelas DOCX, normalização, seções, partes sem sobreposição, metadados, assunto, hashes e detecção de duplicatas.
- SQLite com WAL, transações, chaves estrangeiras, FTS5 e ranking BM25. Busca tolerante a acentos e respostas extrativas com referências. Comparação mostra trechos dos documentos nomeados; possíveis contradições são uma heurística limitada de negação explícita.
- Fila persistente aprovada pelo usuário, progresso, histórico de URLs/documentos, repetição de itens com erro e cancelamento. O desligamento bloqueia novos commits de conhecimento e passos do otimizador sob o mesmo lock.
- Acesso HTTP/HTTPS público, individual ou documentação dentro da origem e caminho fornecidos; profundidade, quantidade, tamanho, intervalos, robots.txt, timeouts, redirecionamentos e duplicatas controlados. Atualizar uma fonte substitui seus trechos e índices, mantendo o mesmo registro de origem.
- DNS validado e conexão fixada no IP público validado, certificado/SNI verificados, proxies de ambiente ignorados. Endereços locais, privados, reservados e faixas de transição IPv6 são bloqueados, inclusive após redirecionamentos. Não há navegador, execução de JavaScript, login ou contorno de proteções.
- Memória documental, conversas autorizadas, correções com fonte/data/validação, configurações e memória de treinamento separadas. Validação de correção é feita pelo usuário. Conversas não são usadas automaticamente como exemplos de treino.
- Transformer causal escrito no projeto, tokenização UTF-8 por bytes, embeddings, atenção, feed-forward, entropia cruzada, AdamW, retropropagação, clipping, lotes pequenos, acumulação e precisão mista CUDA. Nenhum peso de terceiros.
- Métricas antes/depois, conjunto fixo de padrões, replay de dados anteriores, comparação com a versão ativa e rejeição de regressões. Os pesos ativos anteriores permanecem disponíveis se a nova execução falhar na avaliação.
- Checkpoints atômicos com modelo, otimizador, scaler, estado dos geradores, configuração e identidade do snapshot. Carregamento com `weights_only=True`; retomada reproduzida com igualdade exata de pesos em CPU nos testes. CUDA pode variar entre hardware/versões de kernels.
- Restauração manual de checkpoints, gráfico de loss real, tabela comparativa e histórico de métricas/parâmetros. Selecionar uma versão rejeitada para restauração manual é uma decisão explícita do usuário e exige aprendizado ativado.
- Experimento separado REINFORCE em 16 somas de 0 a 3, tentativas e recompensas objetivas persistidas, avaliação antes/depois e checkpoint próprio. Não altera o Transformer documental.
- Monitoramento em segundo plano de CPU, RAM, documentos, disco e operações; GPU/utilização/VRAM via `nvidia-smi`, CUDA e memória dos tensores via PyTorch quando carregado. A interface continua processando eventos durante as tarefas.
- Backups ZIP com manifesto SHA-256, cópia consistente SQLite, snapshots, checkpoints, arquivos da fila e livros com seus originais e licenças. Importação valida caminhos/integridade e restaura em outra subpasta. A pasta anterior é preservada. Permissões de aprendizado e treino automático ficam desativadas após importar.
- Backups periódicos locais com retenção, migração do diretório de dados para outra subpasta da TAC, logs rotativos e diagnóstico de inicialização.

## Rede neural e avaliação

O modelo padrão inicial tem **285.888 parâmetros**, 2 blocos, dimensão 96, 4 cabeças, contexto de 128 bytes e vocabulário de 259 símbolos. Nesta instalação, a ampliação autorizada da versão 0.4.0 tem **495.808 parâmetros**, contexto de **256 bytes** e 896 neurônios em cada camada intermediária dos dois blocos. Continua abaixo de 500 mil parâmetros. A rede de intenções permanece separada, com 264.723 parâmetros. O tamanho pequeno não fornece conhecimento linguístico inicial.

A expansão conserva os embeddings e a atenção anteriores, copia os neurônios já treinados e inicializa os novos no próprio projeto. As novas conexões de saída começam em zero: antes do treino, as previsões no contexto antigo são preservadas. Todos os parâmetros da rede ampliada podem ser treinados. Os checkpoints antigos continuam disponíveis.

O treino autorizado usa os materiais locais, validação separada e repetição de exemplos anteriores. A opção técnica `retention_weight` acrescenta repetição funcional: a própria rede antiga fornece distribuições de referência para prefixos sintéticos aleatórios e motivos aleatórios repetidos. Esses alvos não são respostas factuais nem evidência de compreensão. O teste fixo continua reservado à avaliação. A referência anterior à expansão permanece fixa entre rodadas, evitando acumular regressões de 2% a cada atualização. O checkpoint registra a versão do amostrador de retenção; experimentos iniciais sem essa versão permitem restaurar os pesos e iniciar um treino incremental, mas não retomar exatamente aquele experimento.

Overflows de gradiente em FP16 descartam o lote, reduzem a escala e repetem a tentativa sem executar um passo do otimizador. Erros persistentes ainda interrompem o treinamento e preservam o estado. A validação continua em FP32 e compara as duas arquiteturas nas mesmas janelas de 128 bytes; o treino ampliado usa até 256 bytes.

O relatório desta ampliação fica em `reports/TREINO_500K.md`, com execuções rejeitadas, métricas, consumo de recursos e amostras reais. `scripts/train_500k.py --execute --show` inicia uma nova execução manual com os controles de pausa, interrupção e permissão da interface. Não é executado automaticamente ao abrir a TAC. `scripts/verify_500k.py` audita a entrega sem treinar nem acessar a rede. **A ampliação não torna a geração livre confiável; consulta documental e conversa guiada continuam sendo os modos principais.**

Treino e validação usam intervalos diferentes do texto, sem janelas que cruzem a fronteira. São intervalos do mesmo corpus, portanto a validação mede previsão de texto semelhante e pode ser otimista. A avaliação usa quatro lotes fixos por conjunto para limitar o custo; não avalia toda a distribuição linguística. As métricas não equivalem a qualidade de conversação ou verdade factual.

Uma execução é promovida se a loss de validação atual, a do teste fixo e, quando disponível, a validação do corpus anterior não piorarem mais que 2%. **Metade dos microlotes** usa exemplos anteriores quando o snapshot existe. A identidade desse conjunto e os parâmetros efetivos também são preservados ao retomar um treino. Se o passo final regredir, épocas anteriores são reavaliadas pelos mesmos critérios; uma delas pode ser promovida, mantendo o checkpoint final rejeitado. O relatório distingue passos executados e passo selecionado, com contagens das amostras realmente vistas pela versão escolhida.

O corpus divide seu limite de caracteres entre as fontes, com amostras determinísticas ao longo dos textos. Fontes pequenas liberam espaço para as maiores. Há seleção explícita de fontes para treinos focados; a atualização 0.3.0 combinou a história autorizada e o novo material autoral, repetindo o corpus anterior. Um limite muito pequeno ainda pode impedir a representação de todas as fontes. Esses mecanismos reduzem riscos de esquecimento, sem garantir sua eliminação.

O material `materials/exercicios_conversa_e_escrita_tac.txt`, com cerca de 216 mil caracteres, está na Biblioteca e na cópia gerenciada da fila. Ele contém somente exemplos preparados para este projeto. A execução aprovada da revisão 0.3.0 escolheu o passo 40 entre 1.200 executados: na amostra da história, a loss passou de 3,1634 para 2,9764. O teste fixo piorou 1,4%, dentro do limite de 2%. Não houve promoção das versões posteriores que pioraram mais. Essa melhora de previsão de bytes não demonstra compreensão da história ou conversa livre.

Em **Ferramentas avançadas → Monitoramento** é possível conferir a contagem real de parâmetros dos modelos ativos. A rede de intenções e o Transformer têm funções diferentes; ler mais materiais não aumenta a quantidade de parâmetros.

O lote solicitado de **20 livros em português** fica em `data/books`: originais com licenças/créditos, textos extraídos e `manifest.json` com autores, URLs e hashes. `reports/treino_20_livros.json` registra as execuções e uma avaliação adicional com 16 janelas de validação por livro. O script `scripts/study_books.py --execute` é uma operação manual explícita, nunca executada na abertura da aplicação. Usa o espelho listado oficialmente pelo Gutenberg para downloads, verifica robots.txt e os mesmos bloqueios de rede pública da TAC. Indexar o texto integral permite consultas; o treino utiliza amostras dentro do limite do corpus e não equivale a compreender integralmente as obras.

Checkpoints não convertem a biblioteca em conhecimento neural automaticamente. O Chat documental é utilizável desde o primeiro documento. **Amostra neural experimental** exige um checkpoint treinado ativo e apresenta uma continuação de texto explicitamente identificada. Essa amostra pode conter texto sem sentido ou incorreto.

No teste de padrões cíclicos executado nesta máquina, 50 passos em CUDA reduziram a loss de validação de **5,5715 para 0,2471**. O conjunto continha sequências como `abcde` e dígitos repetidos: a melhoria demonstra aprendizado desse padrão, sem medir raciocínio ou conversação. Os hashes dos pesos iniciais e finais são diferentes e estão no relatório. No reforço, o acerto passou de **18,75% para 87,5%** nas mesmas 16 somas treinadas; não é avaliação de generalização.

Os experimentos da entrega e da revisão estão isolados em `reports/validation_...`, `reports/revisao_site_...` e `reports/workflow_real_...`. A instalação nova começa sem materiais ou pesos de demonstração ativos. Atualizações preservam os materiais que o usuário já adicionou e autorizou.

## Permissões e recursos

O estado do aprendizado é persistido. Com ele desligado, a TAC permite consultas, leitura de histórico, preparação da fila e ações administrativas explícitas (exclusão, backup e preferências), mas não grava novos conhecimentos/correções/conversas nem executa passos de treino. Checkpoints e diagnósticos de interrupção ainda podem ser salvos para preservar o estado confirmado.

**Permitir treinamento neural automático** é independente do estudo e começa falso. Ao reabrir a aplicação, nenhuma execução automática começa sem a opção adicional **Autorizar treino automático também ao abrir a TAC**. Sem essa opção, é necessário armar o automático nesta sessão, marcando-o ou processando uma fila. Frequência e duração são respeitadas. Inicializar um treino manual é autorização para aquela execução.

Padrão: teto verificado de 3 GB de RAM da TAC, reserva mínima de 768 MB livres para o sistema, limite do alocador CUDA de 3 GB, 4 threads, lote 8 e acumulação 2. Esses limites são cooperativos e verificados entre etapas; não constituem isolamento rígido por processo do Windows. O runtime CUDA, bibliotecas, drivers e outros processos também usam memória. O alocador PyTorch não cobre toda a memória do driver. Erros de RAM/VRAM são apresentados e preservam checkpoint quando possível; reduza lote/contexto ou use CPU.

Arquivos têm limite de tamanho e DOCX tem limite de descompactação. PDF tem checagem de tamanho de streams extraídos, mas descompactação/parsers podem alocar memória antes da checagem. Não há sandbox de processo ou limite rígido de RAM para os parsers nesta versão. Não trate essas checagens como garantia contra todo documento malformado.

Todo o código, ambiente, dados, cópias gerenciadas, backups e relatórios permanecem dentro de TAC. Configurações aceitam outra **subpasta** do projeto. A migração conserva a pasta anterior e reinicia a interface; não move nem exclui documentos originais. O acesso à rede da aplicação se limita ao estudo explícito das URLs da fila, seus robots.txt e links dentro do escopo de documentação. A instalação manual de dependências usa os repositórios públicos de pacotes.

## Exclusão e retenção

Excluir uma fonte elimina registro, trechos FTS e snapshots de treino que a referenciem. A cópia do arquivo na fila pode ser removida separadamente com **Remover item da fila**. Conversas autorizadas e correções podem ser excluídas pelas telas de gerenciamento. Histórico de estudos, logs, checkpoints e backups anteriores podem conservar referências ou cópias do material: a exclusão da biblioteca não promete limpeza de arquivos históricos. Administre os backups e arquivos históricos conscientemente.

**Excluir um documento não remove sua influência de pesos que já foram treinados com ele.** Não há esquecimento neural automático. Retreinar a partir de um estado anterior à exposição ou da inicialização, com avaliação verificável, seria necessário para investigar essa remoção. Checkpoints históricos permanecem disponíveis; um snapshot excluído não pode ser retomado, mas o estado dos pesos pode ser restaurado.

## Arquivos e arquitetura

```text
TAC/
  INICIAR_TAC.bat         execução desktop
  INSTALAR_TAC.bat        criação do venv e dependências
  DIAGNOSTICO_TAC.bat     verificação local
  TESTAR_TAC.bat          suíte automatizada
  requirements*.txt      dependências diretas e versões instaladas
  config.json            preferências persistentes
  tac/
    __main__.py          entrada, diagnóstico e instância única
    ui.py                seis painéis, workers, gráficos e monitoramento
    core.py              coordenação e fila
    dialogue_lessons.py  exercícios autorais de conversa, validação e teste
    config.py            validação e gravação atômica de preferências
    db.py                esquema v1, transações, FTS5 e migrações futuras
    memory.py            recuperação, fontes e correções
    answers.py           respostas extrativas curtas e relações explícitas
    presentation.py      estados legíveis, pré-requisitos e explicações
    learning.py          permissão, cancelamento e pausa
    documents.py         extração e divisão de texto
    web.py               acesso público e crawler controlado
    backups.py           backup, restauração e migração local
    diagnostics.py       recursos e logs
    neural/
      tokenizer.py       bytes UTF-8
      model.py           Transformer causal próprio
      data.py            corpus, separação, replay e teste fixo
      training.py        treino, avaliação, checkpoints e inferência
      reinforcement.py  experimento REINFORCE separado
      dialogue.py        classificador neural de intenções e conversa guiada
  data/                  SQLite, inbox, datasets, checkpoints, logs e backups
  examples/              documento demonstrativo escrito para o projeto
  scripts/               validação reproduzível da entrega
  tests/                 testes funcionais, neurais, rede e GUI
  reports/               JUnit, capturas e resultados reais
  .venv/                 ambiente Python isolado
```

Mudanças futuras do esquema devem acrescentar migrações explícitas e preservar `schema_version`. As versões desta entrega rejeitam esquemas incompatíveis; não há migrações inventadas para versões ainda inexistentes.

## Limites e itens ainda incompletos

- Conversação livre, resumo abstrativo, comparação semântica, verificação de fatos e detecção geral de contradições não estão resolvidos. A primeira versão entrega pesquisa/extratos e uma heurística de negação.
- A conversa guiada classifica intenções e usa respostas escritas para os exercícios. Não substitui treinamento de um modelo de linguagem capaz de produzir diálogo livre. Seu teste é um benchmark fixo usado durante o desenvolvimento, e não uma avaliação cega externa. Um novo treino de conversa parte da versão aprovada e repete os exercícios anteriores; checkpoints interrompidos são salvos, mas ainda não têm retomada por lote na interface. A retomada exata implementada continua sendo a do Transformer.
- As respostas curtas não constituem um leitor semântico geral. As regras diretas abrangem nome oficial e capital, com evidências explícitas; campos curtos de capital são reconhecidos apenas no início do documento para reduzir confusão com títulos de referências. Perguntas com outras relações dependem da coincidência lexical de uma frase e podem não encontrar uma resposta existente. Acrescentar arquivos não resolve limitações de interpretação do mecanismo.
- OCR de PDFs digitalizados, imagens, áudio, código de projetos e ferramentas autônomas futuras não foram implementados.
- O crawler lê HTML/texto retornado pelo servidor. Sites dependentes de JavaScript, login, CAPTCHA ou paywall não são suportados. Termos são verificados pelo usuário; robots.txt não substitui autorização jurídica. Duas páginas reais de `docs.python.org/3/` foram validadas nesta entrega; outros sites podem ter comportamento distinto.
- O reforço é um experimento finito, reiniciado a cada execução. Seu checkpoint é salvo, mas a interface ainda não retoma esse experimento de reforço; a retomada implementada é a do Transformer.
- Ainda não há treinamento distribuído, tokenizador aprendido, limpeza automática de checkpoints, deduplicação semântica ou sandbox rígido dos parsers. Os checkpoints e o histórico podem aumentar o espaço ocupado.
- O corpus antigo só pode ser reavaliado se seu snapshot ainda existir. Se ele foi removido, o teste fixo continua, mas não há garantia de retenção do conteúdo removido.
- Não há criptografia de banco/backups. São arquivos locais acessíveis a quem tem acesso à conta Windows.
- Não foi reiniciado fisicamente o Windows. A persistência foi validada fechando e abrindo processos independentes; não se afirma que houve teste de reinicialização física.

Consulte `reports/VALIDACAO.md`, `reports/REVISAO_0.1.1.md`, `reports/revisao-tests.xml`, `reports/release_validation.json` e `data/diagnostics.json` para evidências. Para reproduzir a validação ampliada, execute no ambiente do projeto `python scripts/validate_release.py`: esse script faz treino real e acessa apenas a URL de documentação Python fornecida no pedido original. `python scripts/review_workflow.py` verifica pela interface o artigo Wikipédia fornecido pelo usuário, em uma pasta de teste separada.

Referências técnicas usadas na implementação: [instalação oficial PyTorch](https://pytorch.org/get-started/locally/), [threads e sinais Qt](https://doc.qt.io/qtforpython-6/PySide6/QtCore/QThread.html). Nenhum serviço de IA é consultado pela TAC.
