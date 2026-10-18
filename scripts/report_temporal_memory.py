"""Relatórios de memória temporal, somente leitura dos resultados e gráficos."""
from __future__ import annotations

import argparse
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tac.config import atomic_json
from tac.conversation_lessons import canonical
from scripts.run_temporal_memory import ARMS, OUT, read

LABELS = {'original': 'Original', 'intercalado': 'Intercalado', 'redistribuido': 'Redistribuído', 'temporal_explicito': 'Temporal explícito'}
LABELS['cobertura_temporal'] = 'Cobertura de anterior intermediário'


def pct(v):
    return f'{100 * v:.2f}%'.replace('.', ',')


def metrics(evaluation):
    temporal = evaluation['temporal']['by_skill']
    return {'current': temporal['atual']['rates']['context_correct'],
            'previous': temporal['anterior']['rates']['context_correct'],
            'transition': temporal['explicar_mudanca']['rates']['context_correct'],
            'copy': evaluation['copy']['rates']['characters_exact'],
            'basic': evaluation['basic']['exact_accuracy'],
            'validation': evaluation['validation']['rates']['context_correct'],
            'validation_own': evaluation['validation_own']['rates']['context_correct'],
            'train_probe': evaluation['train_probe']['rates']['context_correct'],
            'temporal_characters': evaluation['temporal']['rates']['characters_exact'],
            'temporal_coherence_proxy': evaluation['temporal']['rates']['coherence_restricted_proxy']}


def charts(summary, records):
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont, QFontDatabase, QPainter
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCharts import QChart, QChartView, QBarSeries, QBarSet, QBarCategoryAxis, QValueAxis, QLineSeries
    app = QApplication.instance() or QApplication([])
    QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')
    app.setFont(QFont('Segoe UI', 10))
    for mode in ('results', 'loss'):
        chart = QChart()
        chart.setTheme(QChart.ChartTheme.ChartThemeDark)
        chart.setTitle('Memória temporal e preservação: médias de três seeds' if mode == 'results' else 'NLL na mesma validação de nomes (média por configuração)')
        chart.setTitleFont(QFont('Segoe UI', 12))
        if mode == 'results':
            bars = QBarSeries()
            for arm in ('active', *ARMS):
                item = QBarSet(LABELS.get(arm, 'Ativo original'))
                for key in ('current', 'previous', 'transition', 'copy', 'basic'):
                    item.append(100 * (summary['active'][key] if arm == 'active' else summary['arms'][arm][key]['mean']))
                bars.append(item)
            chart.addSeries(bars)
            axis = QBarCategoryAxis()
            axis.append(['Nome atual', 'Nome anterior', 'Transição', 'Cópia exata', 'Básicas'])
            chart.addAxis(axis, Qt.AlignmentFlag.AlignBottom)
            bars.attachAxis(axis)
            vertical = QValueAxis()
            vertical.setRange(0, 100)
            vertical.setTitleText('Acertos (%) — critérios originais')
            chart.addAxis(vertical, Qt.AlignmentFlag.AlignLeft)
            bars.attachAxis(vertical)
        else:
            for arm in ARMS:
                series = QLineSeries()
                series.setName(LABELS[arm])
                for step in (0, 40, 80, 120):
                    values = [point['validation']['loss'] for r in records if r['arm'] == arm for point in r['curves'] if point['step'] == step]
                    series.append(step, statistics.mean(values))
                series.setPointsVisible(True)
                chart.addSeries(series)
            chart.createDefaultAxes()
            chart.axes(Qt.Orientation.Horizontal)[0].setTitleText('Passos de atualização')
            chart.axes(Qt.Orientation.Vertical)[0].setTitleText('NLL não ponderada (menor é melhor)')
        chart.legend().setFont(QFont('Segoe UI', 9))
        for axis in chart.axes():
            axis.setLabelsFont(QFont('Segoe UI', 10))
            axis.setTitleFont(QFont('Segoe UI', 10))
        view = QChartView(chart)
        view.setRenderHint(QPainter.RenderHint.Antialiasing)
        view.resize(1200, 680)
        view.show()
        app.processEvents()
        if not view.grab().save(str(OUT / ('resultados.png' if mode == 'results' else 'curvas_validacao.png'))):
            raise OSError('Falha ao salvar gráfico.')
        view.close()


def main():
    global ARMS
    final = read(OUT / 'resultado.json')
    first_final = final
    protocol = read(OUT / 'protocolo.json')
    baseline = read(OUT / 'ativo.json')['evaluation']
    derived = read(OUT / 'dados_derivados.json')
    records = [read(OUT / f'{arm}_s{seed}/result.json') for arm in ARMS for seed in protocol['seeds']]
    coverage_final_path = OUT / 'cobertura_temporal/resultado.json'
    if coverage_final_path.exists():
        ARMS = (*ARMS, 'cobertura_temporal')
        coverage_final = read(coverage_final_path)
        records += [read(OUT / f'cobertura_temporal/cobertura_temporal_s{seed}/result.json') for seed in protocol['seeds']]
        rankings = {}
        for arm in ARMS:
            runs = [r for r in records if r['arm'] == arm]
            means = {s: statistics.mean(r['evaluation']['temporal']['by_skill'][s]['rates']['context_correct'] for r in runs)
                     for s in ('atual', 'anterior', 'explicar_mudanca')}
            rankings[arm] = {'rank': min(means.values()),
                            'validation_mean': statistics.mean(r['evaluation']['validation']['rates']['context_correct'] for r in runs)}
        winner = max(ARMS, key=lambda arm: (rankings[arm]['rank'], rankings[arm]['validation_mean'], arm == 'original'))
        checkpoint_result = coverage_final if winner == 'cobertura_temporal' else first_final
        assert checkpoint_result['chosen'] == winner
        final = {**checkpoint_result, 'pilots': 15,
                 'optimizer_updates_this_run': first_final['optimizer_updates_this_run'] + coverage_final['optimizer_updates_this_run']}
        atomic_json(OUT / 'selecao_final_comparacao.json', {'chosen_for_diagnosis': winner,
                    'median_seed': final['median_seed'], 'rankings': rankings,
                    'checkpoint_sha256': final['checkpoint_sha256'], 'promoted': False,
                    'stage2_hypothesis_motivated_by_development': True, 'reserved_tests_used': False})
    summary = {'active': metrics(baseline), 'arms': {}, 'selected_for_diagnosis': {'arm': final['chosen'], 'seed': final['median_seed']},
               'reserved_tests_used': False, 'promoted': False}
    protection = []
    for arm in ARMS:
        values = [metrics(r['evaluation']) for r in records if r['arm'] == arm]
        summary['arms'][arm] = {key: {'mean': statistics.mean(v[key] for v in values), 'min': min(v[key] for v in values),
                                     'max': max(v[key] for v in values)} for key in values[0]}
    paired = []
    for arm in ARMS[1:]:
        for seed in protocol['seeds']:
            control_run = next(r for r in records if (r['arm'], r['seed']) == ('original', seed))
            candidate_run = next(r for r in records if (r['arm'], r['seed']) == (arm, seed))
            a, b = metrics(control_run['evaluation']), metrics(candidate_run['evaluation'])
            paired.append({'arm': arm, 'seed': seed, 'deltas': {k: b[k] - a[k] for k in a},
                           'same_supervised_token_budget': control_run['exposure']['supervised_tokens'] == candidate_run['exposure']['supervised_tokens']})
    summary['paired_effects'] = paired
    for r in records:
        basic = r['evaluation']['basic']
        original = baseline['basic']
        unknown = basic['by_skill'].get('unknown', {})
        protection.append({'arm': r['arm'], 'seed': r['seed'],
             'loss_basica': basic['loss'] < original['loss'] * 1.02,
             'respostas_basicas': basic['exact_accuracy'] >= max(.60, original['exact_accuracy'] - .02),
             'desconhecimento': not unknown or unknown['correct'] / max(1, unknown['total']) >= .8,
             'actual_basic_loss': basic['loss'], 'max_basic_loss_exclusive': original['loss'] * 1.02,
             'actual_basic_accuracy': basic['exact_accuracy'], 'min_basic_accuracy': max(.60, original['exact_accuracy'] - .02),
             'full_production_checks_executed': False, 'promoted': False})
    atomic_json(OUT / 'resumo_metricas.json', summary)
    atomic_json(OUT / 'protecoes_parciais.json', protection)
    atomic_json(OUT / 'efeitos_pareados.json', paired)
    chosen = next(r for r in records if (r['arm'], r['seed']) == (final['chosen'], final['median_seed']))
    control = next(r for r in records if r['arm'] == 'original' and r['seed'] == final['median_seed'])
    comparison = []
    meta = {r['group']: r for r in derived['temporal_development']}
    for index, c in enumerate(chosen['evaluation']['temporal']['outputs']):
        a = baseline['temporal']['outputs'][index]
        old = control['evaluation']['temporal']['outputs'][index]
        assert a['group'] == old['group'] == c['group']
        source = meta[c['group']]
        comparison.append({'group': c['group'], 'skill': c['skill'], 'changes': source['changes'],
             'name_chronology': source['source_names'], 'messages': c['messages'], 'expected': c['expected'],
             'active': a['actual'], 'control_same_seed': old['actual'], 'candidate': c['actual'],
             'active_correct': a['context_correct'], 'control_correct': old['context_correct'],
             'candidate_correct': c['context_correct'], 'characters_exact': c['characters_exact'],
             'triage': c['error_type'], 'parsed': c['parsed'], 'target_names': c['target_names']})
    atomic_json(OUT / 'comparacao_temporal_por_pergunta.json', comparison)
    question_sensitivity = []
    for label, e in [('active', baseline), *[(f"{r['arm']}_s{r['seed']}", r['evaluation']) for r in records]]:
        stories = {}
        for r in e['temporal']['outputs']:
            parts = r['group'].split('_')
            stories.setdefault((parts[2], parts[3]), {})[r['skill']] = r
        q = {'model': label, 'stories': len(stories), 'same_current_previous_answer': 0,
             'current_correct_and_same_wrong_previous': 0, 'previous_returns_current_single_name': 0,
             'previous_returns_initial_single_name': 0}
        for tasks in stories.values():
            current, previous, transition = [tasks[s] for s in ('atual', 'anterior', 'explicar_mudanca')]
            assert current['messages'][:-1] == previous['messages'][:-1] == transition['messages'][:-1]
            same = current['actual'] == previous['actual']
            q['same_current_previous_answer'] += same
            q['current_correct_and_same_wrong_previous'] += current['context_correct'] and not previous['context_correct'] and same
            names = meta[previous['group']]['source_names']
            if len(previous['parsed']) == 1:
                value = canonical(next(iter(previous['parsed'].values())))
                q['previous_returns_current_single_name'] += value == canonical(names[-1])
                q['previous_returns_initial_single_name'] += value == canonical(names[0])
        question_sensitivity.append(q)
    atomic_json(OUT / 'sensibilidade_pergunta_temporal.json', {'method': 'Mesmo histórico; só muda a pergunta final. Não treina nem muda as notas.', 'models': question_sensitivity})
    lost_basic = []
    for a, b in zip(baseline['basic']['outputs'], chosen['evaluation']['basic']['outputs']):
        assert a['messages'] == b['messages']
        if a['exact'] and not b['exact']:
            lost_basic.append({'messages': a['messages'], 'skill': a['skill'], 'expected': a['expected'],
                               'active': a['actual'], 'candidate': b['actual']})
    atomic_json(OUT / 'regressoes_basicas.json', {'count': len(lost_basic), 'outputs': lost_basic, 'scope': 'somente avaliação; jamais treinado nesta rodada'})
    ab = summary['active']
    table = '| Configuração | Nome atual | Nome anterior | Explicar transição | Cópia exata | Nomes/roteiros inéditos | Básicas antigas |\n|---|---:|---:|---:|---:|---:|---:|\n'
    keys = ('current', 'previous', 'transition', 'copy', 'validation', 'basic')
    table += '| Ativo original | ' + ' | '.join(pct(ab[k]) for k in keys) + ' |\n'
    for arm in ARMS:
        table += '| ' + LABELS[arm] + ' | ' + ' | '.join(pct(summary['arms'][arm][k]['mean']) for k in keys) + ' |\n'
    per_seed = '| Configuração | Seed | Atual | Anterior | Transição | Cópia | Básicas | Validação anterior |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in records:
        e = r['evaluation']
        cells = [f"{e['temporal']['by_skill'][s]['context_correct']}/16" for s in ('atual', 'anterior', 'explicar_mudanca')]
        cells += [f"{e['copy']['characters_exact']}/32", f"{e['basic']['exact_matches']}/142", f"{e['validation']['context_correct']}/60"]
        per_seed += f"| {LABELS[r['arm']]} | {r['seed']} | " + ' | '.join(cells) + ' |\n'
    exposure_rows = []
    for r in records:
        ex = r['exposure']
        counts = ex['by_skill']
        exposure_rows.append(f"| {LABELS[r['arm']]} | {r['seed']} | {ex['unique_examples']} | {ex['minimum_presentations']}–{ex['maximum_presentations']} | {counts['nome_anterior']} | {counts['contraste']} | {counts['apos_correcao']} | {ex['supervised_tokens']} | {ex['padded_input_tokens']} |")
    (OUT / 'COMPARACAO.md').write_text('# Comparação de memória temporal\n\n' + table + '\nMédias de três seeds. Os critérios são os mesmos; estes são conjuntos de desenvolvimento.\n\n' + per_seed +
       '\n## Exposição real\n\n| Configuração | Seed | Únicos | Usos por exemplo | Nome anterior | Contraste | Após correção | Tokens-alvo | Tokens de entrada com padding |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n' + '\n'.join(exposure_rows) +
       '\n\nTodas têm 1.920 apresentações, 928 únicos e 992 repetições, sem duplicar linhas do dataset. Intercalado preserva o multiconjunto original. Redistribuído preserva o tamanho do alvo por posição e limita usos a 3; deixa alguns exemplos doadores com 1 uso. A apresentação explícita conserva os alvos, mas aumenta entrada e muda a redação; não se deve chamar seu custo de FLOPs de idêntico.\n', encoding='utf-8')
    paired_table = '\n## Efeito contra o controle da mesma seed (pontos percentuais)\n\n| Configuração | Seed | Atual | Anterior | Transição | Cópia | Básicas |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in paired:
        paired_table += f"| {LABELS[r['arm']]} | {r['seed']} | " + ' | '.join(f"{100 * r['deltas'][k]:+.2f}" for k in ('current', 'previous', 'transition', 'copy', 'basic')) + ' |\n'
    with (OUT / 'COMPARACAO.md').open('a', encoding='utf-8') as stream:
        stream.write(paired_table)
    selected = summary['arms'][final['chosen']]
    baseline_guards = baseline['basic']
    temporal_scope = ('Explicar mudança, nesta medição, significa identificar a transição ordenada do nome imediatamente anterior para o atual. '
                      'Não se atribui uma nota de justificativa causal livre, intenção ou semântica geral. O avaliador de contraste original foi mantido.')
    memorization_note = 'Sonda adicional indisponível; nenhuma nota foi estimada.'
    if (OUT / 'sonda_memorizacao_anterior.json').exists():
        memorization = read(OUT / 'sonda_memorizacao_anterior.json')
        memorization_note = '| Checkpoint guardado | Anterior nos 64 exemplos realmente estudados | Mesmo histórico novo, pergunta familiar |\n|---|---:|---:|\n'
        for label, item in memorization['models'].items():
            e, q = item['evaluation'], item['familiar_question_probe']
            memorization_note += f"| {label} | {e['context_correct']}/{e['total']} | {q['context_correct']}/{q['total']} |\n"
        memorization_note += ('\nA segunda coluna usa os históricos efetivamente treinados por cada checkpoint, incluindo os prefixos/antecedentes das variantes. '
                             'A terceira muda SOMENTE a formulação final dos 16 casos novos de anterior para a pergunta usada no treino daquele checkpoint. '
                             'É uma sonda de dependência da formulação, não um novo teste cego e não altera notas principais. São checkpoints e seeds identificados, não médias de três seeds. '
                             'O painel legado `train_probe` tem os 58 textos originais: nas variantes que mudam históricos, ele mede retenção do formato anterior, não memorização dos novos textos.')
    basic_regression = selected['basic']['mean'] < baseline_guards['exact_accuracy'] - .02
    conditions = ('Há regressão nas perguntas básicas em relação ao ativo; a configuração não é recomendada para integração.' if basic_regression else
                  'O painel básico não mostra regressão acima da tolerância original; as demais proteções de produção ainda precisam ser executadas.')
    ranking = read(OUT / 'selecao_final_comparacao.json')['rankings'] if coverage_final_path.exists() else read(OUT / 'selecao_congelada.json')['aggregates']
    tie_note = ('Houve empate no mínimo temporal entre todas as configurações. O desempate veio da validação anterior; isso não demonstra superioridade temporal.'
                if len({v['rank'] for v in ranking.values()}) == 1 else
                'O ranking considera as três partes da tarefa; o melhor valor médio não garante melhora em todas as seeds ou ausência de regressão.')
    report = f'''# TAC — memória temporal e atualização de nomes

Rodada concluída: **{len(records)} condições/seed de 120 passos**, {len(ARMS)} configurações e seeds 42, 417 e 903. Foram executadas **{final['optimizer_updates_this_run']} atualizações novas**, pois o controle da seed 42 reutiliza pesos existentes, com hash conferido. Os controles 417/903 foram reconstituídos porque seus pesos não estavam guardados; os hashes finais coincidem exatamente com a rodada anterior. Nenhum modelo foi promovido.

Uma gravação de checkpoint foi bloqueada por margem de disco insuficiente. As métricas e os originais ficaram preservados. O último artefato foi recuperado reconstruindo apenas a seed mediana a partir do mesmo ativo, em 120 passos; o hash dos pesos coincide exatamente com o registrado antes da falha. Essas 120 atualizações de reconstrução estão incluídas no total acima, sem treinar o candidato por mais épocas nem mudar hiperparâmetros. Evidência em `cobertura_temporal/reconstrucao_checkpoint.json`.

## Antes e depois

{table}

As três primeiras colunas usam 16 situações de desenvolvimento cada: metade com duas mudanças, metade com três. Cópia usa os mesmos 32 nomes inéditos da sonda anterior. A validação de nomes/roteiros inéditos continua com 60 exemplos e foi reutilizada como desenvolvimento, não como teste independente. As 142 básicas são perguntas públicas já expostas na investigação anterior e usadas SOMENTE para verificar retenção, sem entrar no treino.

O painel temporal contém 16 históricos e três perguntas alternativas por histórico; suas 48 linhas não são 48 conversas independentes. Não foi atribuída significância estatística a pequenas diferenças. Os efeitos de cada variante contra o controle da mesma seed, inclusive regressões, estão em `efeitos_pareados.json` e `COMPARACAO.md`.

{temporal_scope}

## Variáveis controladas

- Dataset original de 928 exemplos em 256 conversas. Mesmo checkpoint inicial `{protocol['parent']}` e SHA `{protocol['parent_sha256']}`. Nunca partiu do candidato que memorizou as 85 respostas.
- 10.004.544 parâmetros, BPE próprio de 1.024 tokens, 5 camadas, dimensão 384, 6 cabeças, contexto 512 e no máximo 7 falas. Arquitetura, tokenizer e função de avaliação não foram alterados.
- AdamW novo; LR 0,00008; warmup linear de 10 passos; taxa constante depois; batch 8 e acumulação 2; weight decay 0,01; clipping 1; dropout 0. FP16 autocast + scaler em CUDA e parâmetros/AdamW FP32; geração/avaliação FP32.
- 120 passos lógicos, 1.920 apresentações e mesmo número de tokens de resposta por seed. O objetivo é a mesma cross entropy com máscara no prompt, sem peso extra nos nomes. As respostas são geradas livremente; a referência só é usada depois para medir.
- Sem documentos, RAG, busca, treinamento com perguntas de avaliação, replay de outras habilidades ou APIs externas.

## Configurações comparadas

1. **Original:** amostragem original, sem alteração. Métricas antigas de nomes são reutilizadas somente após conferir o hash dos pesos; as avaliações temporal, cópia e básicas desta rodada são reais e novas.
2. **Intercalado:** mesmos textos, respostas e multiplicidades da original. Só muda a ordem, distribuindo as habilidades proporcionalmente ao longo das atualizações, sem esgotar as raras no início.
3. **Redistribuído:** mesmos exemplos/textos, com mais exposição a nome anterior, contraste e pergunta após correção. Troca exposições repetidas de correção/recuperação por exemplos de alvo do mesmo tamanho. Todos os 928 continuam aparecendo; teto de 3 usos, igual ao original. O orçamento de tokens-alvo é idêntico por posição, não apenas em média.
4. **Temporal explícito:** cópia derivada dos mesmos 928 exemplos e respostas, com pistas sobre informação inicial, correção, anterior e atual. Nas 32 conversas `t5`, inclui uma mudança intermediária adicional, preservando os nomes pedidos no alvo. Não é o mesmo texto original, nem um teste puro de amostragem. Esta comparação testa a apresentação como um conjunto; não separa o efeito das pistas do efeito da terceira mudança.
5. **Cobertura de anterior intermediário (hipótese suplementar):** após auditar o desenvolvimento, foi registrado um novo protocolo antes desses três pilotos. A base original tinha 64 perguntas isoladas de anterior somente nas famílias `t0/t4`, ambas de uma troca. Esta cópia muda só o início desses históricos para que o mesmo nome-alvo seja intermediário após duas ou três trocas (32 casos de cada), mantendo todas as respostas, contagens, sampler e orçamento. Não adiciona pistas nas perguntas. A mudança foi motivada pela lacuna observada, portanto é uma investigação adaptativa de desenvolvimento, não confirmação por teste independente.

Os pilotos de amostragem usam o dataset integralmente intacto. A variante derivada reconcilia a exigência de manter a base com a de ensinar três mudanças: não foi fingido que o dataset anterior já continha essas histórias. A entrada cresce de máximo {protocol['audits']['original']['max_length']} para {protocol['audits']['explicit']['max_length']} tokens; o custo de entrada e os tempos são reportados, sem alegar FLOPs iguais.

## Ordem temporal e novos casos

O painel novo tem 48 perguntas, com nomes de desenvolvimento que não aparecem no treino e quatro formulações por tarefa. Há 2 ou 3 mudanças de nome com valores distintos, mantendo o nome imediatamente anterior diferente do primeiro. As falas intermediárias da TAC não contêm nomes; todos os fatos estão nas falas do usuário. O histórico completo cabe nas sete falas: a segunda e terceira mudanças podem ocorrer em uma única mensagem, em ordem explícita. Máximo de {protocol['audits']['temporal_development']['max_length']} tokens.

Zero truncamentos, zero nomes-alvo ausentes do contexto. Os painéis anteriores também são avaliados com histórico próprio da rede, separando essa condição do histórico fornecido. Os novos casos temporais têm confirmações fixas sem nomes para isolar o acompanhamento dos fatos; não equivalem a um teste irrestrito com nove ou mais falas.

## Melhor configuração e limites

Pela regra pré-registrada (maximizar o mínimo entre atual/anterior/transição, em média nas três seeds), a configuração registrada para diagnóstico foi **{LABELS[final['chosen']]}**, seed mediana **{final['median_seed']}**. Médias: atual {pct(selected['current']['mean'])}, anterior {pct(selected['previous']['mean'])}, transição {pct(selected['transition']['mean'])}. O ranking não é aprovação de produção nem autorização para trocar o modelo ativo.

{tie_note}

{conditions} Não foi usada uma melhora temporal para compensar uma regressão grave. Os critérios originais `loss_basica`, `respostas_basicas` e `desconhecimento` estão em `protecoes_parciais.json`, com os mesmos limiares. As demais proteções completas de produção não foram executadas; nenhuma foi relaxada ou declarada atingida sem medição.

Com apenas 120 passos por piloto, os dados permitem comparar essas hipóteses locais, não afirmar capacidade máxima do Transformer nem domínio de memória temporal. Não se conclui que aumentar rede, contexto ou dataset resolveria a dificuldade.

## Erros e preservação das habilidades

`ERROS.md` mostra exemplos verificáveis e as diferenças entre nome inicial, anterior e atual. `comparacao_temporal_por_pergunta.json` compara ativo, controle da mesma seed e candidato. `regressoes_basicas.json` registra {len(lost_basic)} respostas antes corretas e perdidas no candidato mediano. Acertos de nomes, caracteres exatos, coerência factual restrita, duas/três mudanças, repetição e EOS continuam disponíveis nos resultados completos por habilidade.

`sensibilidade_pergunta_temporal.json` compara histórias idênticas alterando só a pergunta final. Registra quando a rede dá a mesma resposta para atual/anterior, inclusive casos em que acerta o atual e repete esse nome indevidamente ao perguntar pelo anterior. Isso distingue uma dificuldade de condicionamento pela pergunta da simples impossibilidade de escrever o nome. A sondagem descreve comportamento; não identifica sozinha um mecanismo neural interno.

## Memorização e dependência da formulação

{memorization_note}

## Dados e integridade

`protocolo.json`, `dados_derivados.json`, 12 pastas com `plano.json`, `passos.json`, `curvas.json` quando há treino novo e `result.json`. A subpasta `cobertura_temporal` guarda seu protocolo, treino derivado e três pilotos suplementares. `COMPARACAO.md`, `resumo_metricas.json`, gráficos `resultados.png` e `curvas_validacao.png` contêm os resultados agregados. Os hashes permitem relacionar fontes, planos e pesos.

Foi guardado um checkpoint por rodada: `candidato_temporal.pt` na primeira e `cobertura_temporal/candidato_cobertura.pt` na suplementar. O selecionado na comparação final tem SHA `{final['checkpoint_sha256']}`. Pesos/configurações próprios, fora do catálogo ativo; sem momentos de AdamW para retomada exata. Os outros candidatos preservam métricas, planos, seeds e hashes de pesos. Os artefatos salvos foram recarregados e conferidos.

Modelos ativos, checkpoints antigos, biblioteca, datasets e configurações foram preservados. A compactação NTFS dos JSONs foi sem perda: 27 hashes iguais antes/depois, registrada em `compactacao_datasets.json`. Todos os testes finais anteriores permanecem fechados, inclusive os 56 casos selados de nomes, independentemente do resultado desta rodada. Nenhum dado foi enviado a serviço externo.

Os testes de software e a auditoria final estão em `testes.xml` e `validacao_final.json`. O aprendizado autorizado é checado antes de cada atualização, e desligamento/cancelamento são testados.

## Reproduzir sem sobrescrever

```powershell
.\\.venv\\Scripts\\python.exe -X utf8 scripts\\run_temporal_memory.py --prepare --output reports\\memoria_temporal_repeticao
.\\.venv\\Scripts\\python.exe -X utf8 scripts\\run_temporal_memory.py --output reports\\memoria_temporal_repeticao
```

Exige mesma versão ativa para repetir o ponto inicial, aprendizado autorizado, TAC principal fechada e pasta nova. A fonte anterior não é alterada. Geradores de relatório/validação aceitam `--output`; a validação exige o JUnit da rodada.
'''
    (OUT / 'RELATORIO.md').write_text(report, encoding='utf-8')
    errors = '# Erros temporais e regressões\n\nAs categorias automáticas são triagem, não um avaliador semântico geral. As referências abaixo não foram fornecidas ao gerador.\n\n'
    errors += '## Sensibilidade à pergunta final\n\n'
    for r in question_sensitivity:
        errors += f"- `{r['model']}`: {r['current_correct_and_same_wrong_previous']}/{r['stories']} histórias com atual correto e a mesma resposta errada ao pedir anterior; {r['previous_returns_current_single_name']} respostas ao anterior usam um único nome que é o atual.\n"
    errors += '\n## Casos completos do candidato selecionado\n\n'
    for skill in ('atual', 'anterior', 'explicar_mudanca'):
        for success in (True, False):
            part = [r for r in comparison if r['skill'] == skill and r['candidate_correct'] == success][:2]
            for r in part:
                errors += f"- `{r['group']}`; {r['changes']} mudanças; tarefa {skill}; correto pela regra: {success}.\n  - Ordem dos nomes: {r['name_chronology']}\n  - Histórico: {r['messages']}\n  - Esperado: {r['expected']}\n  - Ativo: {r['active']}\n  - Controle mesma seed: {r['control_same_seed']}\n  - Candidato: {r['candidate']}\n\n"
    errors += '## Perguntas básicas antes corretas e perdidas\n\n'
    for r in lost_basic[:8]:
        errors += f"- Pergunta: {r['messages']}\n  - Ativo/esperado: {r['expected']}\n  - Candidato: {r['candidate']}\n\n"
    (OUT / 'ERROS.md').write_text(errors, encoding='utf-8')
    next_steps = '''# Próximos passos

1. Comparar cada variante com o controle da mesma seed. Um total melhor em uma seed não comprova generalização estável; observar também caracteres, perguntas básicas e diferenças entre duas/três mudanças.
2. Se a intercalação ajudar sem alterar contagens, testar a hipótese de ordem/gradientes em outra rodada curta antes de mudar a proporção. Se redistribuição ajudar, verificar se o ganho veio da cobertura temporal ou da perda de outras exposições; não ultrapassar o teto de repetição sem justificativa.
3. Se apresentação explícita ajudar, separar em uma segunda rodada pistas temporais versus terceira mudança intermediária. O experimento atual não isola esses dois componentes e não justifica incorporar ambos automaticamente.
4. Se ainda confundir o nome imediatamente anterior com o primeiro/atual, revisar os erros e testar uma hipótese de representação/ordem antes de ampliar a base ou a rede. Não transformar o sistema em um extrator determinístico de nomes apresentado como geração neural.
5. Se houver regressão nas habilidades básicas, preservar o ativo. Integração com retenção/replay exige OUTRO experimento controlado; as 142 perguntas desta avaliação não podem virar treino para recuperar artificialmente a nota.
6. Manter todos os testes finais selados. Estes painéis são desenvolvimento, já expostos para comparação; um teste final futuro só pode ser usado após congelar a hipótese e a seleção, sem ajustar depois.

Não houve promoção, alteração de critérios ou treino de objetos, negação, preferências e outras habilidades nesta rodada.
'''
    (OUT / 'PROXIMOS_PASSOS.md').write_text(next_steps, encoding='utf-8')
    charts(summary, records)
    print('Relatórios e gráficos salvos sem treinamento ou abertura de testes finais.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output.resolve()
    if not OUT.is_relative_to(ROOT / 'reports') or OUT == ROOT / 'reports':
        parser.error('Use uma pasta dentro de TAC/reports.')
    main()
