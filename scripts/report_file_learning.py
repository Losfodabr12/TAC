"""Organiza resultados já medidos, sem treinar e sem abrir testes reservados."""
from __future__ import annotations

import json
import sqlite3
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tac.config import atomic_json
from tac.dialogue_imports import ConversationImports
from tac.neural.bpe import LocalBPETokenizer
from scripts.run_controlled_experiments import LiveSettings

OUT = ROOT / 'reports/aprendizado_conversas_0130'


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


class ReadOnlyCatalog:
    def __init__(self, path):
        self.path = path

    def query(self, sql):
        with sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True) as db:
            db.row_factory = sqlite3.Row
            return [dict(r) for r in db.execute(sql)]


def charts(curve, steps):
    # QtCharts é o componente de gráficos já distribuído com PySide6.
    import os
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPainter, QFont, QFontDatabase
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCharts import QChart, QChartView, QLineSeries, QScatterSeries
    app = QApplication.instance() or QApplication([])
    # No plugin offscreen do Windows, a fonte padrão pode ficar sem glifos.
    font_path = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts/segoeui.ttf'
    if font_path.exists():
        QFontDatabase.addApplicationFont(str(font_path))
    app.setFont(QFont('Segoe UI', 10))
    for name, title in [('curva_aprendizado.png', 'Perda no próprio material estudado'),
                        ('respostas_exatas.png', 'Respostas estudadas reproduzidas literalmente (85 exemplos)')]:
        chart = QChart()
        chart.setTitle(title)
        chart.setTheme(QChart.ChartTheme.ChartThemeDark)
        chart.setTitleFont(QFont('Segoe UI', 13))
        chart.legend().setFont(QFont('Segoe UI', 9))
        if name.startswith('curva'):
            series = QLineSeries()
            series.setName('Média dos dois lotes de cada atualização')
            for item in steps:
                series.append(item['step'], item['loss'])
            chart.addSeries(series)
            dataset = QScatterSeries()
            dataset.setName('Arquivo inteiro: antes e depois (FP32)')
            dataset.setMarkerSize(12)
            for item in curve:
                dataset.append(item['step'], item['loss'])
            chart.addSeries(dataset)
        else:
            for key, label in [('exact_gold', 'Histórico original'), ('exact_own', 'Respostas anteriores da própria rede')]:
                series = QLineSeries()
                series.setName(label)
                series.setPointsVisible(True)
                for item in curve:
                    series.append(item['step'], item[key])
                chart.addSeries(series)
        chart.createDefaultAxes()
        for axis in chart.axes():
            axis.setLabelsFont(QFont('Segoe UI', 10))
            axis.setTitleFont(QFont('Segoe UI', 10))
        chart.axes(Qt.Orientation.Horizontal)[0].setTitleText('Passos de atualização dos pesos')
        chart.axes(Qt.Orientation.Vertical)[0].setTitleText('Perda (menor é melhor)' if name.startswith('curva') else 'Respostas exatas')
        if not name.startswith('curva'):
            chart.axes(Qt.Orientation.Vertical)[0].setRange(0, 85)
        view = QChartView(chart)
        view.setRenderHint(QPainter.RenderHint.Antialiasing)
        view.resize(1100, 620)
        view.show()
        app.processEvents()
        if not view.grab().save(str(OUT / name)):
            raise OSError('Falha ao salvar o gráfico.')
        view.close()


def main():
    config = read('configuracao.json')
    result = read('resultado.json')
    before = read('avaliacao_0000.json')
    after = read(f"avaliacao_{result['final_step']:04d}.json")
    preservation = read('preservacao_verificada.json')
    validation = read('validacao_importacao.json')
    tests_path = OUT / 'testes.xml'
    tests_note = 'Os resultados automatizados serão registrados em testes.xml.'
    if tests_path.exists():
        suite = ET.parse(tests_path).getroot().find('testsuite')
        if suite is not None:
            tests_note = (f"{suite.attrib['tests']} testes executados; {suite.attrib.get('failures', '0')} falhas, "
                          f"{suite.attrib.get('errors', '0')} erros. Resultado JUnit: `testes.xml`. "
                          'Cobertura: adaptação e limites do importador, grupos e splits, contexto, '
                          'máscara/EOS, amostragem reprodutível, histórico gerado, '
                          'bloqueio de atualização ao desligar o aprendizado e cancelamento, '
                          'além dos testes existentes de importação, treino importado e pilotos controlados.')
    # Conferência pelo próprio importador em cima de catálogo REAL, somente leitura.
    settings = LiveSettings(ROOT)
    tokenizer = LocalBPETokenizer.from_state(read('dataset.json')['tokenizer_state'])
    importer = ConversationImports(settings, ReadOnlyCatalog(settings.data / 'tac.sqlite3'), None)
    preview = importer.preview(OUT / 'TAC_TREINO_COMPATIVEL.jsonl', tokenizer, config['model_config']['context'])
    atomic_json(OUT / 'conferencia_importador_real.json', preview['report'])
    result['active_preserved'] = preservation['checks']['active']
    result['all_original_files_preserved'] = preservation['all_preserved']
    atomic_json(OUT / 'resultado.json', result)
    comparisons = []
    for mode in ('gold_history', 'own_history'):
        aa, bb = before[mode]['outputs'], after[mode]['outputs']
        assert len(aa) == len(bb) == 85
        for a, b in zip(aa, bb):
            assert (a['group'], a['turn'], a['expected']) == (b['group'], b['turn'], b['expected'])
            comparisons.append({'history_mode': mode, 'skill': a['skill'], 'group': a['group'], 'turn': a['turn'],
                                'user': a['messages'][-1][1], 'expected': a['expected'],
                                'active': a['actual'], 'candidate': b['actual'],
                                'exact_active': a['exact_literal'], 'exact_candidate': b['exact_literal'],
                                'same_history': a['messages'] == b['messages'],
                                'context_active': a['messages'], 'context_candidate': b['messages']})
    atomic_json(OUT / 'comparacao_respostas.json', comparisons)
    rows = []
    for skill, b in after['gold_history']['by_skill'].items():
        a = before['gold_history']['by_skill'][skill]
        ao = before['own_history']['by_skill'][skill]
        bo = after['own_history']['by_skill'][skill]
        rows.append(f"| {skill} | {a['examples']} | {a['exact_literal']} → {b['exact_literal']} | {ao['exact_literal']} → {bo['exact_literal']} |")
    examples = '\n\n'.join(f"Pergunta estudada: **{r['user']}**  \nAtivo: {r['active']}  \nCandidato: {r['candidate']}  \nEsperado: {r['expected']}"
                             for r in comparisons if r['history_mode'] == 'gold_history' and
                             (r['skill'], r['turn']) in {('nome_e_correcao', 3), ('negacao', 1), ('raciocinio_simples', 1)})
    text = f'''# Aquisição de conversas — experimento isolado da TAC

A cópia experimental de **{config['parameter_count']:,} parâmetros** reproduziu **85/85 respostas estudadas**, contra **1/85 no ativo**, em geração neural direta. Nenhum documento, RAG, busca, classificador ou resposta preparada foi consultado. Não há generalização demonstrada por este resultado.

## Material e compatibilidade

- Fonte: `TAC_TREINO_CONVERSAS.jsonl`, SHA-256 `{config['input_sha256']}`.
- 36 conversas completas, 85 exemplos únicos. Categoria virou `skill`; `group` estável foi derivado do conteúdo; a resposta final saiu de `messages` para `answer`. As 36 conversas ficaram explicitamente em `train`, conforme a finalidade deste experimento.
- Cópia compatível: `TAC_TREINO_COMPATIVEL.jsonl`. Arquivo original e TXT equivalente não foram alterados. O TXT equivalente não entrou no treino em duplicidade.
- Validação com funções reais `normalize_record` e `derive_examples`: zero rejeições, zero históricos cortados, zero entradas iguais com respostas conflitantes. Conferência adicional pelo importador real em `conferencia_importador_real.json`, sem efetivar importação no banco.
- Sequência máxima: {validation['maximum_training_sequence_tokens']} tokens; resposta máxima: {validation['maximum_answer_tokens_including_eos']} tokens incluindo EOS; 1.595 tokens supervisionados por passagem. Todo o contexto necessário cabe.
- Não foi criado um conjunto independente a partir desses poucos exemplos. Treino e medição de aquisição usam o mesmo material, explicitamente. Nenhum arquivo reservado foi interpretado ou avaliado; foram apenas calculados hashes para conferir preservação.

## Protocolo e execução real

- Origem: ativo `{config['parent']}`, SHA-256 `{config['parent_sha256']}`.
- Mesma arquitetura, BPE próprio de 1.024 tokens, contexto 512, 5 camadas, dimensão 384, 6 cabeças, feed-forward 1.728, RoPE e embeddings compartilhados. Não houve aumento de parâmetros nem ajuste do tokenizer.
- AdamW novo, LR 0,00008, aquecimento linear de 10 passos e taxa constante depois; betas 0,9/0,999; epsilon 1e-8; weight decay 0,01; corte de gradiente 1; dropout 0.
- Batch 8, acumulação 2: 16 exemplos por atualização; seed 42; permutações completas antes de repetir; parâmetros/otimizador FP32 e autocast FP16 com GradScaler em CUDA. Avaliação FP32 e greedy, máximo de 128 tokens, como o chat neural.
- Somente as 85 respostas deste arquivo. **Zero exercícios internos, zero replay, zero KL e zero textos da biblioteca.** Isso isola aquisição; não é uma configuração aprovada para preservar habilidades antigas.
- Loss mascarada no histórico: apenas tokens da resposta da TAC e EOS contribuem. Cada resposta intermediária da conversa também é supervisionada, conservando as falas anteriores como contexto.
- Plano registrado antes: blocos de 120 até no máximo 360 passos ou 180 segundos de treino. Parada ao reproduzir literalmente todos os exemplos com os dois tipos de histórico. **Parou no primeiro bloco: {result['steps']} passos, {result['train_seconds']:.3f} segundos de atualizações**, sem contar avaliação e auditorias de disco. Não foram ajustados hiperparâmetros após ver resultados.
- {result['exposure']['presentations']} apresentações, 85 exemplos únicos, {result['exposure']['repetitions']} repetições; {result['exposure']['effective_passes']:.6f} passagens equivalentes. Cada exemplo apareceu 22 ou 23 vezes. Repetição é deliberada neste teste pequeno de aquisição/memorização.
- GPU: {config['gpu']}. Pico do alocador PyTorch: {result['peak_cuda_allocated_mb']:.2f} MiB alocados e {result['peak_cuda_reserved_mb']:.2f} MiB reservados. RAM do processo nas amostras: máximo {max(r['process_mb'] for r in result['resources']):.2f} MiB; não é medição contínua do pico. Limites respeitados: RAM/VRAM de 3.072 MiB e reserva de RAM de 768 MiB.

## Resultados medidos

| Medição no próprio treino | Ativo (0 passos) | Cópia (120 passos) |
|---|---:|---:|
| Respostas literalmente exatas, histórico original | {before['gold_history']['exact_literal']}/85 | {after['gold_history']['exact_literal']}/85 |
| Respostas literalmente exatas, histórico gerado pela rede | {before['own_history']['exact_literal']}/85 | {after['own_history']['exact_literal']}/85 |
| Igualdade normalizada, histórico original | {before['gold_history']['exact_normalized']}/85 | {after['gold_history']['exact_normalized']}/85 |
| Loss ponderada por tokens, arquivo inteiro | {before['loss']:.9f} | {after['loss']:.9f} |
| Próximo token correto com resposta anterior fornecida | {before['teacher_forced_token_accuracy']:.4%} | {after['teacher_forced_token_accuracy']:.4%} |
| EOS gerado, histórico original | {before['gold_history']['eos']}/85 | {after['gold_history']['eos']}/85 |
| Emissão indevida de Usuário:, histórico original | {before['gold_history']['user_role_emitted']} | {after['gold_history']['user_role_emitted']} |
| Repetição de trigramas, histórico original | {before['gold_history']['repeated_trigram']} | {after['gold_history']['repeated_trigram']} |

Histórico original usa falas anteriores do arquivo. Histórico próprio usa as mensagens do usuário do arquivo, mas **todas as falas anteriores da TAC vêm das gerações do modelo avaliado**. A resposta esperada só é usada na comparação depois da geração. Só as últimas sete falas entram no modelo, como no chat; nenhuma foi perdida neste material. Igualdade normalizada ignora caixa, acentos e pontuação: não é uma nota semântica.

| Habilidade | Exemplos | Exatas: histórico original (ativo → cópia) | Exatas: histórico próprio (ativo → cópia) |
|---|---:|---:|---:|
{chr(10).join(rows)}

## Evidências e curva

- `passos.json`: 120 registros de atualização, LR, loss do lote, norma do gradiente e tokens. `curva.json`/`curva.csv`: avaliação completa antes e depois.
- `curva_aprendizado.png`: curva real da loss dos lotes e medições do arquivo inteiro. `respostas_exatas.png`: os dois pontos medidos; a linha entre eles só conecta esses pontos, não representa avaliações intermediárias.
- `avaliacao_0000.json` e `avaliacao_0120.json`: entradas, respostas esperadas, gerações e encerramento por exemplo nos dois modos de histórico.
- `comparacao_respostas.json`: 170 comparações (85 exemplos × dois modos). Esses relatórios contêm o material fornecido para o experimento e ficam somente neste computador; não foram enviados a serviços.

{examples}

## Pesos e preservação

- `candidato_experimental.pt` ({(OUT / 'candidato_experimental.pt').stat().st_size:,} bytes), SHA-256 `{result['checkpoint_sha256']}`. Carregamento repetido e hash dos pesos conferidos. Os pesos do candidato diferem dos iniciais: `{result['weights_changed']}`.
- `preservacao_antes.json`, `preservacao_depois.json`, `preservacao_verificada.json`: integridade confirmada de todos os checkpoints originais do diretório de dados, três versões ativas, catálogos, datasets, biblioteca, configurações e arquivos fornecidos. Checkpoint ativo intacto: `{preservation['checks']['active']}`; verificações completas: `{preservation['all_preserved']}`.
- Não houve gravação de conversa no histórico, importação efetiva no SQLite, treino automático, promoção ou alteração dos critérios de aprovação. O aprendizado autorizado ao vivo é verificado antes de atualizar os pesos; testes cobrem desligamento e cancelamento.
- O artefato contém pesos e configuração, **não os estados do otimizador**. Pode ser carregado e receber novo treino, mas não retoma exatamente os momentos AdamW desta execução. A arquitetura/checkpoints de produção continuam intactos.

## Verificação de software

{tests_note}

O candidato salvo foi recarregado com o hash de pesos conferido. O chat experimental e as duas opções do BAT foram abertos em processos separados, enviando apenas a primeira pergunta do próprio material estudado e encerrando; não foram introduzidas perguntas de teste inéditas. Transcrições locais: `teste_chat_experimental.txt`, `teste_bat.txt` e `teste_bat_ativo.txt`. Esse teste de abertura não substitui os 85 exemplos avaliados na execução.

## Teste manual sem substituir o ativo

1. Feche a janela principal da TAC.
2. Na pasta TAC, abra **`TESTAR_APRENDIZADO_EXPERIMENTAL.bat`**.
3. Escolha **1** para conversar com o candidato ou **2** para comparar com o ativo. Digite uma fala por vez; `/novo` limpa o contexto e `/sair` fecha.
4. Para verificar generalização, experimente nomes e frases que não estejam neste arquivo. O teste reservado permanece disponível para você, sem ter sido aberto pelo experimento. Esse chat não importa arquivos nem grava as perguntas.

A janela normal da TAC continua com o modelo ativo anterior. O candidato experimental só é usado neste atalho separado. Não há integração automática com a biblioteca ou com a Wikipédia.

## O que o resultado permite concluir

A preparação, máscara de supervisão, atualização dos pesos e geração direta funcionam para este pequeno conjunto: a rede aprendeu a reproduzir todas as respostas, inclusive conversas com seu próprio histórico. O resultado diferencia a dificuldade de aprender um conjunto pequeno da dificuldade de generalizar e de reter habilidades em misturas grandes.

**Ainda não medido:** perguntas inéditas, nomes inéditos, retenção dos exercícios antigos, robustez a paráfrases e repetição em várias seeds. Não é possível atribuir uma nota de compreensão semântica nem afirmar que o candidato conversa livremente. Sem replay nesta execução, perda de habilidades antigas é possível, mas não foi demonstrada nem descartada. O candidato não passou pelos critérios de produção e não deve ser promovido por estes 85 acertos de treino.

Próximo passo: sua avaliação manual inédita, preservando primeiro os resultados. Depois, se houver aquisição sem generalização ou regressão, desenhar outra rodada separada; este experimento não iniciou nenhuma rodada longa.

## Reproduzir sem sobrescrever

```powershell
.\\.venv\\Scripts\\python.exe scripts\\learn_conversation_file.py --input "C:\\Users\\Joaof\\Desktop\\TAC_Experimento_Aprendizado\\TAC_TREINO_CONVERSAS.jsonl" --output reports\\aprendizado_conversas_repeticao
```

Exige aprendizado autorizado, TAC principal fechada e pasta de saída nova. Usa os mesmos pesos ativos de origem; se a versão ativa mudar, isso será registrado como outro ponto inicial. Não lê o teste reservado nem conjuntos internos de exercícios. O atalho manual acompanha a execução original desta pasta; não seleciona automaticamente repetições.
'''
    (OUT / 'RELATORIO.md').write_text(text, encoding='utf-8')
    charts(read('curva.json'), read('passos.json'))
    print('Relatório, comparações, conferência do importador e gráficos salvos.')


if __name__ == '__main__':
    main()
