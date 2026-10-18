# Conversas para o treino de diálogo

Importe em **Melhorar a TAC → Conversas de exemplo → Importar conversas**. Estes arquivos são diferentes dos materiais da Biblioteca. Importar não treina imediatamente. O histórico salvo e as correções registradas não são convertidos automaticamente neste formato.

## JSONL

UTF-8, com ou sem BOM. Um objeto por linha, somente com os cinco campos abaixo. O arquivo completo com várias conversas está em [conversas_exemplo.jsonl](../examples/conversas_exemplo.jsonl).

```jsonl
{"messages":[{"role":"user","content":"Tenho duas tarefas para amanhã."},{"role":"assistant","content":"Qual delas tem o prazo mais próximo?"},{"role":"user","content":"A primeira precisa ser entregue de manhã."}],"answer":"Comece pela primeira e depois reserve um horário para a segunda.","skill":"organizacao","group":"estudo_001","split":"validation"}
```

- `messages`: de 1 a 101 falas anteriores. Cada fala tem somente `role` e `content`. Use `user` para o usuário e `assistant` para a TAC. Comece com o usuário, alterne os papéis e termine com o usuário. O histórico fica nesta lista; não existe um campo `history` adicional.
- `answer`: resposta final da TAC. Todas as respostas intermediárias da TAC em `messages` também geram exemplos de treino ou avaliação, no destino definido pelo grupo.
- `skill`: habilidade que aparecerá no resultado, como `organizacao` ou `conversa_cotidiana`. Até 64 caracteres: letras minúsculas sem acento, números, `_`, `-` e `.`.
- `group`: identificador da conversa e de suas variações. Até 80 caracteres, com os mesmos caracteres permitidos em `skill`. Use o mesmo identificador nas paráfrases da mesma conversa, inclusive em arquivos diferentes. A TAC não reconhece automaticamente todas as paráfrases.
- `split`: `train`, `validation`, `test` ou `auto`.

Não use uma lista JSON envolvendo o arquivo inteiro, Markdown envolvendo cada linha, campos adicionais ou papéis `system`/`tool`. JSON comum não é aceito: converta para um objeto por linha em `.jsonl`.

## TXT

UTF-8, com ou sem BOM. Cada conversa começa com `=== conversa ===`. Inclua os três campos de cabeçalho e termine com uma fala da TAC.

```text
=== conversa ===
group: estudo_001
skill: organizacao
split: validation

Usuário: Tenho duas tarefas para amanhã.
TAC: Qual delas tem o prazo mais próximo?
Usuário: A primeira precisa ser entregue de manhã.
TAC: Comece pela primeira e depois reserve um horário para a segunda.

=== conversa ===
group: treino_002
skill: escrita
split: train

Usuário: Quero escrever uma cena curta.
  Ela começa numa casa vazia.
TAC: Escolha um detalhe que mostre por que a casa parece vazia.
```

Para continuar uma fala em outra linha, comece com **dois espaços**. Linhas vazias não separam conversas. Os marcadores são exatamente `Usuário:` e `TAC:`. O exemplo completo equivalente ao JSONL está em [conversas_exemplo.txt](../examples/conversas_exemplo.txt).

## Separação e conferência

`train` atualiza pesos. `validation` e `test` ficam reservados e não são usados para atualizar pesos nem para escolher ou aprovar a nova versão. A comparação dessas conversas é feita depois da escolha do candidato pelos testes internos anteriores.

O destino é fixado por **grupo**, globalmente. Um grupo já registrado não pode mudar de destino em outra importação, mesmo se seu arquivo estiver desabilitado. Um conflito recusa os novos registros desse grupo. Em `auto`, o identificador determina o destino de forma estável, com proporção alvo de 80%/10%/10%; arquivos pequenos podem ficar sem validação ou teste. O resultado não depende da ordem das linhas ou do nome do arquivo.

A conferência informa arquivo, linha, grupo, quantidades e motivos de rejeição. Papéis inválidos, ordem incorreta das falas ou uma resposta grande demais recusam o registro inteiro. Duplicatas são ignoradas; sobreposições de entradas com destinos diferentes ou avaliações internas protegidas são recusadas e explicadas. Podem entrar os outros exemplos válidos do arquivo. Registros ilegíveis são contados separadamente, pois não é possível afirmar quantas respostas eles continham.

Limites: tamanho configurado de arquivo (20 MiB por padrão), 20.000 registros, 50.000 exemplos de resposta e 1 MiB por linha JSONL. Cada texto de fala tem limite de 200.000 caracteres. Respostas devem caber no codificador existente: com contexto 512, o máximo é **507 tokens, incluindo espaço inicial e encerramento**. Uma resposta com 508 já é recusada; o limite anterior não foi afrouxado. A contagem usa o vocabulário local ativo, não o número de palavras. A resposta nunca é cortada para ser aceita.

O chat gera até 128 novos tokens. Respostas maiores recebem aviso na importação, mesmo quando cabem no treino. O contexto considera as últimas sete falas terminando no usuário e pode recortar texto para caber nos 512 tokens. O arquivo completo fica preservado na cópia gerenciada.

## Treino e resultado

Antes de iniciar aparece: **“Este treino vai usar X conversas suas e Y exercícios do programa.”** X é a quantidade de grupos habilitados destinados a treino, e não a soma de todos os arquivos. As quantidades de respostas, validação e teste são mostradas separadamente. Nenhum arquivo é treinado só por ser importado ou habilitado.

Com conversas importadas de treino, a mistura tem 27,5% de lotes de suas conversas, 27,5% de conversas revisadas do programa, 30% de escrita e 15% de exercícios antigos. A retenção por KL continua usando a rede própria anterior congelada nos lotes antigos, com o peso existente 2. Sem conversas importadas de treino, a mistura anterior permanece 55%/30%/15%. Este caminho exige o currículo de conversas revisadas e o mesmo vocabulário BPE da versão anterior.

O resultado mostra, por habilidade, **anterior N/total → nova N/total** na validação e no teste importados. A conversa é reconstruída com as respostas geradas pelo próprio modelo no histórico. As respostas esperadas não são dadas ao modelo durante essa avaliação. **O acerto exige redação idêntica à referência:** uma resposta correta escrita de outro jeito pode não contar. Isso não mede compreensão geral, verdade factual ou fluência livre.

Essa comparação de respostas completas é informativa e fica fora dos critérios de aprovação. Na versão 0.12, novas execuções usam também a perda de tokens da validação importada para escolher entre candidatos com perda básica menor que 102% da referência. A soma das perdas internas desempata; nenhum dado do teste participa dessa escolha. A aprovação final continua exigindo todos os critérios internos anteriores. Retomadas antigas mantêm sua política de seleção registrada. Uma falha na comparação de respostas é informada como indisponível; não vira nota zero nem muda a decisão dos testes internos. Um pedido de interrupção continua impedindo a ativação de uma versão nova.

Em Ferramentas avançadas há estratégias experimentais de retenção, amostragem e geração, todas opcionais. O padrão mantém a mistura e a geração anteriores. A taxa de aprendizado da conversa tem ajuste separado, inicialmente 0,00002. Novos treinos salvam a referência original de avaliação e a conservam ao retomar; isso impede tratar uma perda acumulada como ponto de partida aceitável. Consulte `reports/research_0120/RELATORIO.md` para os pilotos e suas regressões.

O conjunto, o vocabulário, o estado do treino, a mistura e a versão de referência ficam congelados no progresso salvo. Retomar usa esses dados, mesmo que você habilite outros arquivos depois; os novos materiais entram em uma nova etapa. A comparação da retomada usa a referência do início dessa execução.

## Guardar e recuperar

As conversas ficam nas tabelas `dialogue_imports`, `dialogue_groups` e `dialogue_examples` do SQLite; os originais são copiados para `data/dialogue_imports`. Os arquivos que você forneceu não são alterados. O banco é migrado para a versão 2 com uma cópia de segurança antes da migração de uma instalação existente. A exportação de backup inclui o acervo importado e continua aceitando backups antigos.

**Escolher arquivos para o próximo treino** permite habilitar ou desabilitar cada importação. Isso não apaga o material nem desfaz sua influência em pesos já treinados. Não existe exclusão seletiva de influência neural ou avaliação semântica automática neste importador.
