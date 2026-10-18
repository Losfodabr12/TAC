"""Relatório dos pilotos reais; não treina, não gera e não lê teste selado."""
from __future__ import annotations

import os
import argparse
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tac.config import atomic_json
from tac.conversation_lessons import canonical, prompt_text
from tac.neural.bpe import LocalBPETokenizer
from scripts.run_name_generalization import OUT, read

LABELS = {'sft': 'Supervisão normal', 'nomes_peso3': 'Peso 3 nos tokens dos nomes'}
PANELS = {'train_probe': 'Exemplos estudados (painel)', 'dev_names': 'Nomes inéditos, roteiros conhecidos',
          'dev_structures': 'Nomes conhecidos, roteiros inéditos', 'validation': 'Nomes e roteiros inéditos',
          'validation_own': 'Ambos inéditos, histórico gerado pela rede'}


def percent(value):
    return f'{value * 100:.2f}%'.replace('.', ',')


def graphs(records, active):
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont, QFontDatabase, QPainter
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCharts import QChart, QChartView, QLineSeries, QBarSeries, QBarSet, QBarCategoryAxis, QValueAxis
    app = QApplication.instance() or QApplication([])
    QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')
    app.setFont(QFont('Segoe UI', 10))
    for metric in ('loss', 'accuracy'):
        chart = QChart()
        chart.setTheme(QChart.ChartTheme.ChartThemeDark)
        chart.setTitle('Perda na validação inédita (não é o teste final)' if metric == 'loss' else 'Nome correto no contexto: estudado versus inédito')
        chart.setTitleFont(QFont('Segoe UI', 12))
        if metric == 'loss':
            for r in records:
                series = QLineSeries()
                series.setName(f"{r['arm']} seed {r['seed']}")
                series.setPointsVisible(True)
                for point in r['curve']:
                    series.append(point['step'], point['validation']['loss'])
                chart.addSeries(series)
            chart.createDefaultAxes()
            chart.axes(Qt.Orientation.Horizontal)[0].setTitleText('Passos')
            chart.axes(Qt.Orientation.Vertical)[0].setTitleText('NLL não ponderada: menor é melhor')
        else:
            series = QBarSeries()
            for arm in ('ativo', 'sft', 'nomes_peso3'):
                bars = QBarSet(LABELS.get(arm, 'Ativo original'))
                for panel in ('train_probe', 'dev_names', 'dev_structures', 'validation'):
                    values = [active['evaluation'][panel]['rates']['context_correct']] if arm == 'ativo' else [r['evaluation'][panel]['rates']['context_correct'] for r in records if r['arm'] == arm]
                    bars.append(100 * statistics.mean(values))
                series.append(bars)
            chart.addSeries(series)
            axis = QBarCategoryAxis()
            axis.append(['Estudados', 'Nomes novos', 'Roteiros novos', 'Ambos novos'])
            chart.addAxis(axis, Qt.AlignmentFlag.AlignBottom)
            series.attachAxis(axis)
            vertical = QValueAxis()
            vertical.setRange(0, 100)
            vertical.setTitleText('Acertos factuais pela regra restrita (%)')
            chart.addAxis(vertical, Qt.AlignmentFlag.AlignLeft)
            series.attachAxis(vertical)
        for axis in chart.axes():
            axis.setLabelsFont(QFont('Segoe UI', 9))
            axis.setTitleFont(QFont('Segoe UI', 10))
        chart.legend().setFont(QFont('Segoe UI', 9))
        view = QChartView(chart)
        view.setRenderHint(QPainter.RenderHint.Antialiasing)
        view.resize(1200, 680)
        view.show()
        app.processEvents()
        if not view.grab().save(str(OUT / ('curvas_validacao.png' if metric == 'loss' else 'generalizacao.png'))):
            raise OSError('Falha no gráfico.')
        view.close()


def main():
    final = read(OUT / 'resultado.json')
    selection = read(OUT / 'selecao_congelada.json')
    protocol = read(OUT / 'protocolo.json')
    active = read(OUT / 'ativo.json')
    audit = read(OUT / 'auditoria_dados.json')
    diagnostics = read(OUT / 'sondas_diagnosticas.json')
    panels = read(OUT / 'dados_desenvolvimento.json')['panels']
    records = [read(OUT / f'{arm}_s{seed}/result.json') for arm in protocol['arms'] for seed in protocol['seeds']]
    chosen = next(r for r in records if r['arm'] == final['chosen'] and r['seed'] == final['median_seed'])
    assert len({r['initial_weight_hash'] for r in records} | {active['weight_hash']}) == 1
    matched = {str(seed): len({r['batch_plan_sha256'] for r in records if r['seed'] == seed}) == 1 for seed in protocol['seeds']}
    assert all(matched.values())
    tokenizer = LocalBPETokenizer.from_state(read(OUT / 'dados_desenvolvimento.json')['tokenizer_state'])
    # Roteiros reais, com placeholders: independência não é só nome diferente no ID.
    def signature(row):
        text = prompt_text(row['messages'])
        for name in sorted(row['source_names'], key=len, reverse=True):
            text = text.replace(name, '{NOME}')
        return canonical(text)
    templates_train = {signature(r) for r in panels['train']}
    templates_validation = {signature(r) for r in panels['validation']}
    assert not templates_train & templates_validation
    summary = {'active': {p: e['rates'] for p, e in active['evaluation'].items()},
               'arms': {}, 'chosen': {'arm': chosen['arm'], 'seed': chosen['seed']},
               'matched_batch_plans': matched, 'main_validation_template_overlap': 0,
               'new_final_test_used': final['new_final_test_used'], 'promoted': False}
    for arm in protocol['arms']:
        runs = [r for r in records if r['arm'] == arm]
        summary['arms'][arm] = {p: {k: {'mean': statistics.mean(r['evaluation'][p]['rates'][k] for r in runs),
                                           'min': min(r['evaluation'][p]['rates'][k] for r in runs),
                                           'max': max(r['evaluation'][p]['rates'][k] for r in runs)}
                                      for k in runs[0]['evaluation'][p]['rates']} for p in PANELS}
    effects = []
    for seed in protocol['seeds']:
        a, b = [next(r for r in records if r['arm'] == arm and r['seed'] == seed) for arm in protocol['arms']]
        effects.append({'seed': seed, 'same_examples_and_tokens': a['batch_plan_sha256'] == b['batch_plan_sha256'],
                        'context_delta': b['evaluation']['validation']['rates']['context_correct'] - a['evaluation']['validation']['rates']['context_correct'],
                        'characters_delta': b['evaluation']['validation']['rates']['characters_exact'] - a['evaluation']['validation']['rates']['characters_exact']})
    summary['paired_weight3_effect'] = effects
    summary['validation_by_skill'] = {
        arm: {s: {k: statistics.mean(r['evaluation']['validation']['by_skill'][s]['rates'][k]
                                    for r in records if r['arm'] == arm)
                  for k in chosen['evaluation']['validation']['by_skill'][s]['rates']}
              for s in chosen['evaluation']['validation']['by_skill']}
        for arm in protocol['arms']}
    ablation_effects = {}
    for label, original in [('active', active['evaluation']['validation']),
                            ('candidate', chosen['evaluation']['validation'])]:
        ablated = diagnostics['measurements'][label]['assistant_name_ablated']
        pairs = list(zip(original['outputs'], ablated['outputs']))
        assert all((a['group'], a['turn']) == (b['group'], b['turn']) for a, b in pairs)
        ablation_effects[label] = {
            'correct_to_wrong': sum(a['context_correct'] and not b['context_correct'] for a, b in pairs),
            'wrong_to_correct': sum(not a['context_correct'] and b['context_correct'] for a, b in pairs),
            'text_changed': sum(a['actual'] != b['actual'] for a, b in pairs)}
    summary['assistant_echo_ablation_paired'] = ablation_effects
    atomic_json(OUT / 'resumo_metricas.json', summary)
    comparisons = []
    for panel in PANELS:
        for a, b in zip(active['evaluation'][panel]['outputs'], chosen['evaluation'][panel]['outputs']):
            assert (a['group'], a['turn']) == (b['group'], b['turn'])
            comparisons.append({'panel': panel, 'group': a['group'], 'turn': a['turn'], 'skill': a['skill'],
                                'messages_active': a['messages'], 'messages_candidate': b['messages'],
                                'expected': b['expected'], 'target_names': b['target_names'],
                                'active': a['actual'], 'candidate': b['actual'],
                                'correct_active': a['context_correct'], 'correct_candidate': b['context_correct'],
                                'characters_exact': b['characters_exact'], 'error_type': b['error_type']})
    atomic_json(OUT / 'comparacao_respostas.json', comparisons)
    review = [r for r in comparisons if r['panel'] == 'validation' and not r['correct_candidate']
              and all(re.search(r'(?<!\w)' + re.escape(canonical(n)) + r'(?!\w)', canonical(r['candidate']))
                      for n in r['target_names'].values())]
    atomic_json(OUT / 'revisao_humana_pendente.json', {'method': 'Presença de nomes esperados não comprova sentido correto; somente triagem para revisão.', 'examples': review, 'reviewed_by_human': False})
    lengths = {}
    for label, condition in [('ate_6', lambda n: n <= 6), ('7_a_12', lambda n: 7 <= n <= 12), ('acima_12', lambda n: n > 12)]:
        rows = [r for r in chosen['evaluation']['validation']['outputs'] if condition(max(len(n) for n in r['target_names'].values()))]
        lengths[label] = {'examples': len(rows), 'context_correct': sum(r['context_correct'] for r in rows), 'characters_exact': sum(r['characters_exact'] for r in rows)}
    atomic_json(OUT / 'erros_por_comprimento.json', lengths)
    table_rows = []
    for r in records:
        e = r['evaluation']
        table_rows.append('| ' + ' | '.join([r['arm'], str(r['seed']), str(r['steps']),
             str(r['exposure']['unique_examples']), str(r['exposure']['repetitions'])] +
             [f"{e[p]['context_correct']}/{e[p]['total']}" for p in PANELS] +
             [f"{r['curve'][-1]['train']['loss']:.4f}", f"{r['curve'][-1]['validation']['loss']:.4f}"]) + ' |')
    comparison_md = '''# Comparação dos pilotos de nomes

Acertos abaixo usam a regra factual restrita, que verifica qual nome ocupa o papel pedido. Não são notas de conversação livre. Os painéis auxiliares reutilizam um eixo do treino; somente a validação principal tem nomes e roteiros simultaneamente inéditos.

| Braço | Seed | Passos | Exemplos únicos vistos | Repetições | Estudados | Nomes inéditos | Roteiros inéditos | Ambos inéditos | Ambos, histórico próprio | NLL treino | NLL validação |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
''' + '\n'.join(table_rows)
    comparison_md += '\n\nOs lotes e tokens são idênticos entre os dois braços de cada seed. A única variável é peso 1 versus peso 3 nos tokens dos nomes; o histórico e EOS continuam com o mesmo formato. A NLL de avaliação é sempre sem ponderação.\n\n'
    comparison_md += '| Seed | Efeito do peso 3: contexto (p.p.) | Efeito: caracteres (p.p.) |\n|---|---:|---:|\n'
    comparison_md += '\n'.join(f"| {r['seed']} | {100 * r['context_delta']:+.2f} | {100 * r['characters_delta']:+.2f} |" for r in effects)
    (OUT / 'COMPARACAO.md').write_text(comparison_md, encoding='utf-8')
    main_table = '| Painel | Ativo | Normal, média de 3 seeds | Peso 3, média de 3 seeds |\n|---|---:|---:|---:|\n'
    main_table += '\n'.join(f"| {label} | {percent(summary['active'][p]['context_correct'])} | {percent(summary['arms']['sft'][p]['context_correct']['mean'])} | {percent(summary['arms']['nomes_peso3'][p]['context_correct']['mean'])} |" for p, label in PANELS.items())
    skills = chosen['evaluation']['validation']['by_skill']
    skill_table = '| Habilidade, validação principal | Nome normalizado | Caracteres exatos | Vínculo contextual |\n|---|---:|---:|---:|\n'
    skill_table += '\n'.join(f"| {s} | {v['name_identity']}/{v['total']} | {v['characters_exact']}/{v['total']} | {v['context_correct']}/{v['total']} |" for s, v in skills.items())
    selected_rates = chosen['evaluation']['validation']['rates']
    max_memory = max(r['cuda_peak_allocated_mib'] or 0 for r in records)
    max_ram = max(v['process_mb'] for r in records for v in r['resources'])
    final_note = ('O teste novo foi aberto uma única vez, depois da seleção congelada; resultados em teste_final_unica_avaliacao.json. Não houve ajuste posterior.'
                  if final['new_final_test_used'] else 'O gate de consistência não foi atingido. O teste novo de 56 exemplos permanece selado e não foi avaliado; não foi gasto para escolher hiperparâmetros.')
    scope = '''As formas de responder são conhecidas; a novidade controlada está nos nomes e nos roteiros de entrada. Há 8 roteiros de treino, 4 de desenvolvimento e 4 de teste, todos autorais e sintéticos. IDs diferentes foram complementados por comparação do texto com nomes substituídos por placeholders: zero sobreposição de entradas completas entre treino e validação principal. Isso não prova independência semântica absoluta: os roteiros compartilham o conceito de nome e correção, como é necessário para medir transferência dessa regra.

Os nomes de desenvolvimento/teste são inéditos neste novo dataset. Nomes comuns podem ter aparecido em treinamentos antigos do checkpoint original; não se afirma ineditismo em toda a vida da TAC. Os resultados por nomes inventados separam melhor essa limitação. Os exemplos não são dados pessoais coletados.'''
    report = f'''# Generalização de nomes e correções — TAC 0.13

Foram concluídos **6 pilotos de 120 passos**, com seeds 42, 417 e 903, todos a partir do checkpoint ativo original de **10.004.544 parâmetros**, sem alterar arquitetura ou tokenizer. Não foi usado o candidato que memorizou as 85 respostas anteriores.

## O que aprendeu e o que generalizou

{main_table}

O painel estudado tem 58 exemplos de 16 conversas que realmente pertencem ao treino de 928 exemplos. Ele mede reprodução e aplicação em situações estudadas; não é avaliação completa de todas as 928 respostas. NLL de treino usa o conjunto inteiro. Os painéis de nomes/roteiros inéditos são desenvolvimento, usados para comparar variantes, não uma prova final independente.

{scope}

## Dataset e separação

- Treino: 928 exemplos, 256 conversas, inventário de 160 nomes comuns, raros e inventados, com comprimentos e caracteres variados. Cada exemplo supervisiona uma resposta; conversas inteiras permanecem no mesmo destino.
- Validação principal: 60 exemplos, 16 conversas; inventário próprio de desenvolvimento com 48 nomes e 4 roteiros não usados no treino. Nenhum nome reservado aparece como nome completo ou parte delimitada de um composto no texto de treino.
- Painéis auxiliares: 58 exemplos com nomes novos/roteiros conhecidos e 60 com nomes conhecidos/roteiros novos. Eles intencionalmente isolam um eixo; não são apresentados como separação simultânea dos dois eixos.
- Teste novo: 56 exemplos, 16 conversas, inventário separado de 48 nomes e 4 outros roteiros. {final_note}
- Os dois testes reservados anteriores (`sealed_transfer.json` e `TAC_TESTE_RESERVADO_NAO_IMPORTAR.txt`) foram apenas conferidos por hash, sem interpretação nem avaliação.
- Históricos variam entre 1, 3, 5 e 7 mensagens. Zero truncamentos e zero respostas cujo nome-alvo estivesse ausente do prompt. Familiares/colegas, assinatura, cadastro, duas mudanças de nome e perguntas sobre o anterior estão separados por habilidade.

Uma falha de auditoria foi detectada e corrigida antes de congelar o dataset: nomes reservados curtos apareciam como componentes de nomes compostos do treino. A alocação passou a separar também esses componentes entre destinos. O teste automatizado detectou o problema; nenhum piloto usou a versão contaminada.

## Como treinou

AdamW novo, LR 0,00008, 10 passos de aquecimento e taxa constante depois; batch 8, acumulação 2, weight decay 0,01, corte de gradiente 1, dropout 0. Cada piloto executou 1.920 apresentações, viu os 928 exemplos e repetiu 992 apresentações: 2,069 passagens equivalentes. Não houve mistura de exercícios internos, livros, outras habilidades ou replay/KL.

`sft` usa cross entropy normal. `nomes_peso3` multiplica por 3 a contribuição dos tokens que se sobrepõem aos nomes na resposta esperada, normalizando pela soma dos pesos. O histórico tem loss zero; EOS tem peso 1. Não existe nome esperado sendo entregue ao gerador, filtro que force a cópia, substituição de resposta ou pointer adicionado à arquitetura. Os nomes são conhecidos SOMENTE pelo contexto do usuário, como em qualquer tarefa legítima de cópia.

Os dois braços de cada seed usam exatamente os mesmos lotes, tokens e passos, com hashes conferidos. Parâmetros/AdamW FP32, autocast FP16 e GradScaler em CUDA; métricas e geração em FP32. Greedy com limite de 80 novos tokens. Dados, protocolo e seeds foram congelados antes de treinar. Não foi iniciada extensão longa.

Tempo de atualizações nos seis pilotos: {sum(r['train_seconds'] for r in records):.2f} segundos, excluindo avaliações/auditorias. GPU: {final['gpu']}. Pico máximo alocado pelo PyTorch: {max_memory:.2f} MiB; maior RAM amostrada do processo: {max_ram:.2f} MiB. Limites configurados respeitados. Relatórios técnicos usam compactação NTFS sem perda para preservar espaço; seus conteúdos lógicos e os pesos originais permanecem iguais.

## Como ler as medidas

- **Resposta literal:** todo o texto deve coincidir com a resposta escrita no dataset.
- **Nome normalizado:** nomes extraídos pela gramática restrita coincidem após ignorar caixa, acentos e pontuação; não basta citar palavras aleatórias da conversa.
- **Caracteres exatos:** nomes extraídos preservam todos os caracteres, caixa, acentos e separadores; não usa a normalização anterior.
- **Recuperação contextual:** além do nome, deve acertar os papéis de atual e anterior. Trocar os dois pode preservar caracteres e ainda falhar aqui.
- **Correções:** as linhas `correcao`, `apos_correcao`, `nome_anterior` e `contraste` mostram as partes do problema, sem uma média que esconda erros de papel.
- **Coerência restrita:** vínculo factual correto + EOS + ausência de marcador Usuário: e repetição. Não é nota semântica geral. Respostas fora da gramática podem exigir revisão humana; não foram automaticamente creditadas ou consideradas semanticamente erradas.
- **Histórico próprio:** cada resposta anterior da TAC é gerada por ela, sem usar respostas esperadas. O teste gold é reportado separadamente.

## Candidato experimental registrado

Para guardar um artefato reproduzível, foi escolhido **{chosen['arm']}, seed mediana {chosen['seed']}**. Escolha por média do braço, usando o menor entre contexto e caracteres na validação principal; depois seed mediana por contexto, nunca a melhor seed. Resultado desse candidato: {percent(selected_rates['context_correct'])} em vínculo contextual e {percent(selected_rates['characters_exact'])} em caracteres exatos na validação principal.

{skill_table}

Nenhum candidato foi promovido. O gate de consistência exige médias ≥80% para contexto e caracteres, mínimo de 70% por seed em contexto e amplitude ≤15 p.p. entre seeds. Esse gate de pesquisa não substitui nem relaxa os critérios de produção existentes. A seleção não abriu o teste anterior.

## O que realmente melhorou

O treino especializado melhorou métricas de desenvolvimento em relação ao ativo, mas o ganho do peso 3 precisa ser julgado pelos três pares, não por uma seed isolada. Consulte `COMPARACAO.md` para os efeitos positivos e negativos por seed. Nenhuma evidência foi fabricada para aprovar uma variante. As limitações e sondas estão em `FALHAS.md`.

Nesta rodada, peso 3 piorou o acerto contextual nas três seeds, com média de {100 * statistics.mean(e['context_delta'] for e in effects):+.2f} p.p. em relação à supervisão normal. Não foi incorporado ao treino principal. O treino especializado, por outro lado, elevou a média contextual inédita de {percent(summary['active']['validation']['context_correct'])} para {percent(summary['arms']['sft']['validation']['context_correct']['mean'])}; isso ainda fica longe da consistência exigida.

## Evidências e segurança

`protocolo.json`, `auditoria_dados.json`, `dados_desenvolvimento.json`, `ativo.json`, seis pastas de pilotos com `passos.json`/`curva.json`/`result.json`, `resumo_metricas.json`, `comparacao_respostas.json`, `sondas_diagnosticas.json` e `selecao_congelada.json`. Curvas em `curvas_validacao.png` e `generalizacao.png`.

`candidato_nomes.pt` contém somente pesos/configuração próprios, fora do catálogo de produção; não contém momentos do otimizador para retomada exata. SHA-256: `{final['checkpoint_sha256']}`. Carregamento conferido contra hash de pesos da execução. As outras cinco execuções guardam métricas e hashes, não cinco cópias grandes dos pesos.

`preservacao_antes.json`, `preservacao_depois.json` e `preservacao_verificada.json` conferem modelos ativos, todos os checkpoints originais, pesos de pesquisas anteriores, datasets, biblioteca, configurações e arquivos fornecidos anteriormente. `validacao_final.json` e `testes.xml` registram a checagem final. Nenhum dado foi enviado para serviço externo.

Validação de software: **45 testes passaram**, sem falhas na rodada final. Os testes incluem separação dos conjuntos e históricos, máscara/EOS, tokens UTF-8, comparação de papéis atual/anterior, geração sem resposta esperada e bloqueio de atualizações quando o aprendizado é desligado. A validação final também confere os hashes dos relatórios anteriores compactados, sem alteração de conteúdo.

## Integração e próximos passos

Não há justificativa para integrar automaticamente este candidato ao treinamento principal apenas pelos acertos do painel estudado. A decisão precisa de generalização consistente, teste novo após congelar a seleção e retenção das habilidades antigas. A retenção de outras habilidades nos candidatos não foi avaliada nesta rodada; preservar arquivos não comprova retenção neural. As redes ativas e a biblioteca permanecem intactas. Consulte `PROXIMOS_PASSOS.md`.

## Reproduzir preservando esta rodada

Exige TAC principal fechada e aprendizado autorizado. Estes comandos executam outra rodada curta, em pasta nova, sem substituir esta:

```powershell
.\\.venv\\Scripts\\python.exe -X utf8 scripts\\run_name_generalization.py --prepare --output reports\\generalizacao_nomes_repeticao
.\\.venv\\Scripts\\python.exe -X utf8 scripts\\run_name_generalization.py --output reports\\generalizacao_nomes_repeticao
.\\.venv\\Scripts\\python.exe -X utf8 scripts\\diagnose_name_generalization.py --output reports\\generalizacao_nomes_repeticao
```

A versão ativa precisa ser a mesma para reproduzir o ponto de partida; `protocolo.json` registra ID, SHA, dataset e configurações. Rodadas iniciadas não são sobrescritas. Os geradores de relatório e validação final também aceitam `--output`; a validação final exige o JUnit da rodada.
'''
    (OUT / 'RELATORIO.md').write_text(report, encoding='utf-8')
    d = diagnostics['measurements']
    selected_validation = chosen['evaluation']['validation']
    failures = f'''# Falhas e diagnóstico — somente desenvolvimento

## Evidência de cópia versus compreensão do papel

Erros do candidato na validação: `{selected_validation['error_types']}`. Comprimentos em `erros_por_comprimento.json`; nomes comuns, raros e inventados têm contagens separadas em `result.json`.

As categorias de nomes podem se sobrepor em uma resposta com dois nomes. Não se devem somar essas categorias como observações independentes. A distribuição do treino por habilidade é `{audit['panels']['train']['by_skill']}`; a exposição real de cada piloto está em `exposure`. Nome anterior e contraste têm menos exemplos do que identificação/correção. Esse desequilíbrio é um fato dos dados, não prova de que ele sozinho causou a falha.

Nome e caracteres podem estar corretos e o papel atual/anterior estar trocado. A taxonomia é uma triagem literal: `wrong_name_from_context` usa um nome da conversa em lugar do pedido; `partial_or_mutated_copy` indica prefixo reconhecível com mutação; `unverified_or_out_of_context` inclui texto fora da gramática e NÃO comprova, sozinho, erro semântico.

Há {len(review)} casos da validação com todos os nomes esperados mencionados, mas sem crédito contextual automático. Estão em `revisao_humana_pendente.json`, sem nota humana inventada. Formatos legítimos não reconhecidos devem ser revisados antes de alegar uma falha semântica.

## Tokenizer e representação

O BPE próprio de 1.024 tokens foi mantido. Todos os nomes dos inventários passaram por encode/decode com preservação exata de UTF-8; não existe palavra fora do vocabulário irrepresentável, porque há fallback de bytes. Isso NÃO demonstra que a rede aprendeu a copiar esses bytes.

`auditoria_dados.json` registra tokens por nome, uso dos tokens de bytes e caracteres sem exemplos neste novo treino. Nomes inventados frequentemente exigem mais tokens. A comparação por categoria/comprimento é correlacional, não uma prova de que o BPE causou os erros. Todos os nomes necessários estão presentes no contexto efetivamente entregue à rede. Zero truncamentos; a hipótese de perda por corte não explica estes erros.

## Predição com resposta anterior correta versus geração livre

| Modelo | Tokens dos nomes corretos no treino | Na validação inédita | Primeiro token dos nomes na validação |
|---|---:|---:|---:|
| Ativo | {percent(d['active']['train_tokens']['name_token_accuracy_teacher_forced'])} | {percent(d['active']['validation_tokens']['name_token_accuracy_teacher_forced'])} | {percent(d['active']['validation_tokens']['first_name_token_accuracy'])} |
| Candidato | {percent(d['candidate']['train_tokens']['name_token_accuracy_teacher_forced'])} | {percent(d['candidate']['validation_tokens']['name_token_accuracy_teacher_forced'])} | {percent(d['candidate']['validation_tokens']['first_name_token_accuracy'])} |

Essas sondas fornecem o prefixo correto da resposta; não devem ser confundidas com a geração livre. Diferença entre acertar tokens com prefixo ouro e produzir o nome inteiro indica uma dificuldade de trajetória autoregressiva, sem demonstrar por si só a causa arquitetural.

## Ablação do eco nas falas da TAC

Foi alterado SOMENTE o texto das respostas anteriores da TAC para `Entendido.`. As falas do usuário, nomes, ordem, perguntas e limite de contexto foram mantidos. Nenhuma linha foi treinada.

| Modelo | Histórico original: contexto correto | Sem nomes nas respostas anteriores da TAC |
|---|---:|---:|
| Ativo | {active['evaluation']['validation']['context_correct']}/60 | {d['active']['assistant_name_ablated']['context_correct']}/60 |
| Candidato | {selected_validation['context_correct']}/60 | {d['candidate']['assistant_name_ablated']['context_correct']}/60 |

Esse teste mede sensibilidade à apresentação do histórico e possível dependência de eco do assistente. A mudança também muda a distribuição das respostas intermediárias; não é prova isolada de um mecanismo interno. Histórico próprio, reportado separadamente, evita atribuir capacidade ao simples uso de respostas anteriores sempre perfeitas.

Comparação por pergunta (um total igual pode esconder trocas): ativo `{ablation_effects['active']}`; candidato `{ablation_effects['candidate']}`. `correct_to_wrong` conta perdas, `wrong_to_correct` conta ganhos e `text_changed` conta respostas alteradas. Não se conclui ausência completa de sensibilidade apenas pela igualdade dos totais.

## Cópia isolada de nomes inéditos

Prompt diagnóstico fixo, fora da seleção de hiperparâmetros: `Repita exatamente, sem mudar as letras, este nome: ...`. Só usa nomes do desenvolvimento. Não abre o teste final.

- Ativo: {d['active']['isolated_copy']['characters_exact']}/{d['active']['isolated_copy']['total']} nomes com caracteres exatos.
- Candidato: {d['candidate']['isolated_copy']['characters_exact']}/{d['candidate']['isolated_copy']['total']}.

No candidato, por tipo: `{ {k: {'correct': v['characters_exact'], 'total': v['total']} for k, v in d['candidate']['isolated_copy']['by_name_kind'].items()} }`. A rede também erra alguns nomes comuns; a categoria raro/inventado, sozinha, não explica os erros.

O Transformer tem atenção causal e pode aprender cópia, mas não possui mecanismo explícito pointer/copy head. Não foi adicionado extrator externo nem forçada uma resposta a partir do prompt. As sondas orientam uma próxima hipótese; não justificam aumentar a rede ou o dataset automaticamente.

## Exemplos verificáveis do candidato

'''
    for error in ('correct', 'changed_characters', 'swapped_roles', 'wrong_name_from_context', 'partial_or_mutated_copy', 'unverified_or_out_of_context'):
        cases = [r for r in selected_validation['outputs'] if r['error_type'] == error][:2]
        for r in cases:
            failures += f"\n- Grupo `{r['group']}`, turno {r['turn']}, habilidade `{r['skill']}`, triagem `{error}`.\n  - Histórico: {r['messages']}\n  - Esperado: {r['expected']}\n  - Gerado: {r['actual']}\n"
    (OUT / 'FALHAS.md').write_text(failures, encoding='utf-8')
    next_steps = '''# Próximos passos

1. Revisar os casos fora da gramática antes de concluir que todos são semanticamente errados. Preservar as notas automáticas desta rodada; não editar o avaliador retroativamente para fazer um candidato passar.
2. Usar a comparação pareada das três seeds para decidir se a ponderação de nomes merece nova rodada. Um efeito pequeno/inconsistente não comprova melhoria. Não abrir o teste novo enquanto falhar o gate de consistência.
3. Separar, numa próxima hipótese pré-registrada, cópia imediata de recuperação/atualização. Como nome anterior/contraste falham mesmo quando a cópia isolada funciona, priorizar exposição às perguntas que exigem vincular cada nome ao papel. Comparar ordem aleatória com intercalação por habilidade, preservando o mesmo multiconjunto de exemplos, tokens e passos; sem criar exemplos novos nem repetir somente um conjunto raro. A ablação do eco teve efeito líquido nulo neste candidato, portanto variar confirmações ainda é uma hipótese secundária, não uma correção comprovada. Só repetir pilotos curtos depois de congelar a hipótese.
4. Não ampliar automaticamente o corpus ou a arquitetura. Se nomes inéditos continuarem falhando apesar de dominar o treino, testar representação em bytes dos mesmos nomes como comparação controlada antes de criar um pointer head. Qualquer mudança no tokenizer exige atenção à compatibilidade dos embeddings e checkpoints e uma comparação explícita, não troca silenciosa do vocabulário.
5. Mesmo que a habilidade passe no desenvolvimento e no teste novo, comparar o candidato com o original nos critérios de produção e tarefas antigas. Depois, testar integração com replay em outro experimento, preservando retenção e sem promover automaticamente.
6. Comparação de objetos, negação e preferências ficam adiadas até nomes/correções demonstrarem consistência. A sequência proposta é: relações entre dois objetos com atributos inéditos; depois negação de uma propriedade; depois preferência atualizada. Cada habilidade precisa de dataset e teste selado próprios, com apenas um fator novo por rodada. Nenhuma dessas habilidades foi misturada neste treino.

Os testes finais reservados anteriores continuam fechados. A validação deste experimento já foi usada para comparar variantes; novos resultados nela não equivalem a um teste final independente.
'''
    (OUT / 'PROXIMOS_PASSOS.md').write_text(next_steps, encoding='utf-8')
    graphs(records, active)
    print('Relatórios, comparações e gráficos prontos; nenhum treinamento realizado pelo gerador de relatórios.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output.resolve()
    if not OUT.is_relative_to(ROOT / 'reports') or OUT == ROOT / 'reports':
        parser.error('Pasta de rodada inválida; use TAC/reports.')
    main()
