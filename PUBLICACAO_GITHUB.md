# Código da TAC no GitHub

Este repositório contém o código-fonte, a interface, os scripts e os testes de software da TAC. A linha experimental v3 tem 29.414.400 parâmetros, inicializados no próprio projeto; a v2 tem 10.004.544 parâmetros. Não há pesos de modelos externos.

## O que permanece no computador

Conforme solicitado, **`data/` e `reports/` não são enviados**. Também ficam fora pesos (`.pt`, `.pth`, `.safetensors`), bancos SQLite, configurações locais, documentos em `materials/`, arquivos ZIP, caches e conjuntos congelados de avaliação. Os arquivos originais continuam no computador, sem exclusão ou alteração para este envio.

O `config.json` local não é versionado. `config.example.json` contém os padrões do código, com aprendizado e treino automático desligados. A aplicação cria a configuração quando ela não existe; não é necessário sobrescrever uma configuração já instalada.

Os relatórios e resultados citados pelo README pertencem à instalação local e não acompanham esta publicação. Os conjuntos congelados usados por partes dos testes antigos também permanecem locais; uma clonagem não contém todos os insumos para reproduzir esses benchmarks.

## Executar e analisar

No Windows, com Python 3.13 de 64 bits, use `INSTALAR_TAC.bat` para instalar as dependências e `INICIAR_TAC.bat` para abrir a interface. Uma clonagem nova **não contém os checkpoints nem o corpus treinado** e, portanto, não reproduz as respostas da instalação original. Nenhum treinamento é iniciado como parte desta publicação.

Arquivos principais para análise:

- `tac/neural/model.py`: Transformer causal e geração.
- `tac/neural/conversation.py`: carregamento e execução da conversa neural.
- `tac/neural/training.py`, `v2_training.py`, `v3_training.py`: fluxos de treinamento.
- `tac/neural/bpe.py` e `v3.py`: tokenizers locais e configuração v3.
- `tac/neural/decoding.py`: decodificação e parada de repetições.
- `tac/*_lessons.py`: exercícios e serialização de mensagens das gerações anteriores.
- `scripts/`: preparação, treinamento, avaliação e ferramentas de diagnóstico. Os scripts são publicados como código; não são executados pelo envio ao GitHub.

O treinamento da v3 foi concluído, mas não atingiu conversação coerente. Ela permanece experimental e a v2 foi preservada. A publicação do código não representa aprovação de qualidade nem autorização para iniciar outra rodada.
