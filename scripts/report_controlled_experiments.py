"""Consolida somente medições existentes; nenhum modelo é carregado ou treinado."""
from __future__ import annotations

import hashlib
import argparse
import importlib.util
import json
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tac.config import atomic_json
from tac.neural.closed_facts import novel_fact

OUT=ROOT/'reports/experimentos_0130'
LABELS={'control':'Controle','A_response':'A — resposta da TAC','B_balanced':'B — importações balanceadas',
    'C_coverage':'C — cobertura ampliada','C_representative':'C — cobertura representativa'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def ratio(a,b):return a/max(1,b)


def flat(e):
    g=e['generation'];novel=[s['factual_correct'] for s in e['novel']['by_skill'].values()]
    result={f'{k}_accuracy':e[k]['exact_accuracy'] for k in ('basic','context','natural','imported')}
    result.update(basic_loss=e['basic']['loss'],imported_loss=e['imported_full_loss']['loss'],
        fluency_accuracy=e['fluency']['rule_accuracy'],session_recall=e['sessions']['recall_accuracy'],
        eos_rate=ratio(g['eos'],g['responses']),role_rate=ratio(g['user_role_emitted'],g['responses']),
        repetition_rate=ratio(g['repeated_trigram_3_times'],g['responses']),
        novel_factual_proxy=ratio(sum(s['correct'] for s in novel),sum(s['total'] for s in novel)),
        imported_own_accuracy=ratio(e['imported_sessions']['exact_matches'],e['imported_sessions']['examples']))
    arithmetic=e['context']['by_skill'].get('arithmetic',{})
    result['arithmetic_accuracy']=ratio(arithmetic.get('correct',0),arithmetic.get('total',0))
    facts=[novel_fact(r) for r in e['novel']['results'] if r['expected'] is not None]
    result['closed_factual_verified_fraction']=ratio(sum(r['status']=='correto' for r in facts),len(facts))
    result['closed_factual_unassessed_fraction']=ratio(sum(r['status']=='nao_avaliado' for r in facts),len(facts))
    return result


def row(result,path):
    e=result['evaluation'];r={'arm':result['arm'],'seed':result['seed'],**flat(e),
        'result_file':path.relative_to(OUT).as_posix(),'checks_count':f"{sum(result['protection_checks'].values())}/{len(result['protection_checks'])}",
        'failed_checks':[k for k,v in result['protection_checks'].items() if not v],
        'training_seconds':result['training_seconds'],'probe_seconds':result['probe_seconds'],
        'peak_vram_mb':result['peak_vram_mb'],'passed_original_checks':result['passed_original_checks']}
    for k in ('basic','context','natural','imported'):r[k+'_count']=f"{e[k]['exact_matches']}/{e[k]['examples']}"
    return r


def aggregate(rows):
    out=[]
    fields=[k for k,v in rows[0].items() if type(v) is float and k!='seed']
    for arm in LABELS:
        runs=[r for r in rows if r['arm']==arm]
        if not runs:continue
        metrics={}
        for k in fields:
            values=[r[k] for r in runs if r[k] is not None]
            if values:metrics[k]={'mean':statistics.mean(values),'std':statistics.stdev(values) if len(values)>1 else None,
                'min':min(values),'max':max(values),'n':len(values)}
        out.append({'arm':arm,'seeds':[r['seed'] for r in runs],'metrics':metrics})
    return out


def skills_delta(base,current):
    report={}
    for split in ('basic','context','natural','imported'):
        old=base[split]['outputs'];new=current[split]['outputs'];assert len(old)==len(new)
        groups=defaultdict(lambda:{'total':0,'old_correct':0,'new_correct':0,'lost':0,'gained':0,'changed_answers':0})
        examples=[]
        for a,b in zip(old,new):
            assert (a['messages'],a['expected'],a['skill'])==(b['messages'],b['expected'],b['skill'])
            s=groups[a['skill']];s['total']+=1;s['old_correct']+=a['exact'];s['new_correct']+=b['exact']
            s['lost']+=a['exact'] and not b['exact'];s['gained']+=b['exact'] and not a['exact'];s['changed_answers']+=a['actual']!=b['actual']
            if a['actual']!=b['actual']:
                examples.append({'skill':a['skill'],'messages':a['messages'],'expected':a['expected'],
                    'active':a['actual'],'candidate':b['actual'],'active_exact':a['exact'],'candidate_exact':b['exact'],
                    'semantic_equivalence':None,'off_topic':None,'incomplete':None,'review':'pendente_humano'})
        report[split]={'by_skill':dict(groups),'changed_examples':examples}
    return report


def gradient_report(result):
    rows=result['gradients_and_adam'];records=[]
    for r in rows:
        before=r['probe_losses_before'];after=r['probe_losses_after'];u=r['actual_adam_update']
        records.append({'step':r['step'],'legacy_imported_cosine':r['fixed_probe_pairs']['imported__legacy']['cosine'],
            'adam_vs_clipped_gradient_cosine':u['vs_clipped_gradient']['cosine'],
            'actual_batch_fp32_vs_clipped_cosine':r['clipped_vs_fp32_weighted_gradient']['cosine'],
            'actual_update_norm':u['norm'],
            'tasks':{p:{'gradient_dot_adam_delta':u['vs_fixed_gradients'][p]['dot'],
                'actual_one_update_loss_change':after[p]-before[p],
                'linear_prediction_sign_agrees':(u['vs_fixed_gradients'][p]['dot']>0)==(after[p]-before[p]>0)} for p in before}})
    return {'arm':result['arm'],'seed':result['seed'],'measurements':records,
        'scope':'5 lotes fixos de treino por sonda (2 exemplos); todos os parâmetros; 5 atualizações reais. Não é média populacional nem prova causal de todos os erros.'}


def fairness(results):
    checks=[]
    for seed in sorted({r['seed'] for r in results}):
        rs={r['arm']:r for r in results if r['seed']==seed}
        if 'control' not in rs:continue
        base=rs['control']
        for name,r in rs.items():
            if name=='control':continue
            variable='language' if name=='A_response' else ('imported' if name=='B_balanced' else 'legacy')
            check={'seed':seed,'arm':name,'same_initial_weights':r['initial_weight_hash']==base['initial_weight_hash'],
                'same_steps':r['steps_completed']==base['steps_completed'],
                'other_pools_identical':all(v==r['plan']['xy_hashes'][k] for k,v in base['plan']['xy_hashes'].items() if k!=variable),
                'tokens_by_pool_control':base['plan']['target_tokens'],'tokens_by_pool_variant':r['plan']['target_tokens']}
            if name=='A_response':check['language_inputs_identical']=r['plan']['x_hashes']['language']==base['plan']['x_hashes']['language']
            if name.startswith('C_'):check['all_supervised_token_budgets_equal']=base['plan']['target_tokens']==r['plan']['target_tokens']
            checks.append(check)
    return checks


def paired_results(rows):
    out=[]
    for arm in LABELS:
        if arm=='control':continue
        comparator='C_coverage' if arm=='C_representative' else 'control'
        pairs=[]
        for current in (r for r in rows if r['arm']==arm):
            before=next((r for r in rows if r['arm']==comparator and r['seed']==current['seed']),None)
            if before is None:continue
            metrics=('basic_accuracy','natural_accuracy','arithmetic_accuracy','imported_accuracy',
                'imported_own_accuracy','basic_loss','imported_loss','eos_rate','role_rate','repetition_rate','novel_factual_proxy')
            pairs.append({'seed':current['seed'],'delta':{k:current[k]-before[k] for k in metrics}})
        if pairs:
            out.append({'arm':arm,'comparator':comparator,'pairs':pairs,
                'mean_delta':{k:statistics.mean(r['delta'][k] for r in pairs) for k in pairs[0]['delta']}})
    return out


def scientific_findings(results,base,paired):
    lines=['## Conclusões baseadas nas medições','']
    counts=[r['evaluation']['basic']['exact_matches'] for r in results]
    high=sum(r['evaluation']['basic']['loss']>=base['basic']['loss']*1.02 for r in results)
    basic_range=str(min(counts)) if min(counts)==max(counts) else f'{min(counts)} a {max(counts)}'
    lines.append(f"Básicas: ativo {base['basic']['exact_matches']}/142, pilotos {basic_range}/142. Em {high}/{len(results)} pilotos a loss básica ultrapassou a margem original de 2%. Isso é regressão medida, não apenas falta de progresso.")
    values=[r['evaluation']['imported']['exact_matches'] for r in results]
    lines.append(f"Importadas: ativo {base['imported']['exact_matches']}/60 exatas; pilotos {min(values)}..{max(values)}/60. O painel é pequeno e estratificado; não equivale aos 1.021 exemplos antigos de validação+teste e não quantifica semântica.")
    for arm in ('control','B_balanced'):
        rs=[r for r in results if r['arm']==arm]
        if not rs:continue
        uniques=statistics.mean(r['exposure']['pools']['imported']['unique_examples'] for r in rs)
        groups=statistics.mean(r['exposure']['pools']['imported']['unique_groups'] for r in rs)
        names=statistics.mean(r['exposure']['draws_by_original_skill'].get('imported/memoria_nome',0) for r in rs)
        general=statistics.mean(r['exposure']['draws_by_original_skill'].get('imported/conversa_geral',0) for r in rs)
        lines.append(f"{LABELS[arm]}: média de {uniques:.2f} conteúdos importados únicos e {groups:.2f} grupos, em 528 sorteios; rótulos originais memoria_nome {names:.2f} e conversa_geral {general:.2f}. Maior unicidade de linha não garante maior cobertura de conversas ou conceitos.")
    probes=[p for r in results for p in r['gradients_and_adam']]
    negative=[p for p in probes if p['fixed_probe_pairs']['imported__legacy']['cosine'] is not None and p['fixed_probe_pairs']['imported__legacy']['cosine']<0]
    improve=sum(p['probe_losses_after']['legacy'] < p['probe_losses_before']['legacy']-1e-6 for p in negative)
    worsen=sum(p['probe_losses_after']['legacy'] > p['probe_losses_before']['legacy']+1e-6 for p in negative)
    lines.append(f"D: {len(negative)}/{len(probes)} sondas tiveram cosseno antigo/importado negativo. Dentre elas, a atualização real reduziu a loss antiga em {improve} casos e aumentou em {worsen} casos acima de 0,000001; os demais foram menores que esse corte descritivo. Portanto o sinal negativo sozinho não identifica quais atualizações prejudicam o antigo. Não se demonstrou uma causa global que justifique cirurgia de gradientes.")
    if (OUT/'cobertura_reproduzida.json').exists():
        coverage=read(OUT/'cobertura_reproduzida.json')['results']
        for arm in ('C_coverage','C_representative'):
            rows=[v for k,v in coverage.items() if k.startswith(arm+'_s')]
            if rows:
                n=[r['legacy_from_previously_excluded_unique'] for r in rows]
                reach=[r['full_pool_reachable_by_skill_and_target_length_in_this_schedule'] for r in rows]
                lines.append(f"{LABELS[arm]}: {min(n)}..{max(n)} conteúdos únicos antes excluídos efetivamente vistos por piloto. Com casamento de habilidade e comprimento, o pool alcançável nesta rodada ficou entre {min(reach)} e {max(reach)} linhas; não foram vistos todos os 82.512 exemplos. Tokens de contexto/padding constam no JSON; somente os tokens supervisionados são exatamente igualados.")
    if (OUT/'sondas_eos_ouro.json').exists():
        eos=read(OUT/'sondas_eos_ouro.json')['results'];a=eos['A_response']['context_256'];c=eos['control']['context_256']
        lines.append(f"A, sonda adicional EXPLORATÓRIA somente na seed 42: ao fornecer o texto correto de cada resposta, EOS passou a argmax em {a['eos_argmax_count']}/120 fronteiras, contra {c['eos_argmax_count']}/120 no controle; probabilidade média {a['mean_p_eos']:.6f} versus {c['mean_p_eos']:.6f}. Isso mostra efeito local na fronteira supervisionada. Não demonstra ganho em geração livre ou generalização: a sonda usa trechos de treino e histórias ouro.")
    lines+=['','Nenhuma variante é aprovada por melhora isolada de loss importada, contagem de exemplos ou sonda de treino. As proteções anteriores e a comparação por habilidade continuam obrigatórias. A decisão desta rodada é preservar o ativo e não iniciar treinamento longo automaticamente.']
    return lines


def preservation():
    spec=importlib.util.spec_from_file_location('tac_audit_readonly',ROOT/'reports/investigacao_0130/audit.py')
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    before=read(OUT/'preservacao_antes.json');after=audit.state()
    atomic_json(OUT/'preservacao_depois.json',after)
    fields=('active','runs','sources','checkpoints','config_sha256','dataset_hashes','candidate_sha256','reserved_sha256','reserved_consumed')
    checks={k:before[k]==after[k] for k in fields}
    checks['sealed_test_not_consumed']=not after['reserved_consumed']
    trainfiles=('tac/neural/conversation.py','tac/neural/model.py','tac/neural/research.py','tac/neural/transfer_eval.py',
        'tac/neural/bpe.py','tac/dialogue_imports.py','tac/fluency_lessons.py')
    checks['production_training_tokenizer_generation_criteria_unchanged']=all(before['application_code'][p]==after['application_code'][p] for p in trainfiles)
    atomic_json(OUT/'preservacao_verificada.json',checks)
    return checks


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();OUT=args.output.resolve()
    if not OUT.is_relative_to(ROOT/'reports') or OUT==ROOT/'reports':parser.error('Escolha uma subpasta de TAC/reports.')
    protocol=read(OUT/'protocol.json');base=read(OUT/'active.json')['evaluation']
    saved_config=read(ROOT/'config.json')
    atomic_json(OUT/'recursos_configurados.json',{k:saved_config[k] for k in ('ram_mb','reserve_ram_mb','vram_mb','cpu_threads','device')})
    paths=sorted(OUT.glob('*_s*/result.json'));results=[read(p) for p in paths]
    rows=[row(r,p) for r,p in zip(results,paths) if r['status']=='completed']
    if not rows:raise ValueError('Nenhum piloto completo. Não criar resultados fictícios.')
    aggregates=aggregate(rows);fair=fairness(results);protected=preservation();paired=paired_results(rows)
    summary={'format':'TAC-controlled-summary-0130-v1','protocol':protocol,'active':flat(base),
        'rows':rows,'aggregates':aggregates,'paired_comparisons':paired,'fairness':fair,'preservation':protected,
        'completed_pilots':len(rows),'expected_pilots':15,'promoted':False,'semantic_review':'pendente_humano',
        'final_test_used':False,'information':'Desenvolvimento exposto, não teste final. Sem mudança de critérios ou promoção automática.'}
    atomic_json(OUT/'summary.json',summary)
    pairs={f"{r['arm']}_s{r['seed']}":skills_delta(base,r['evaluation']) for r in results}
    atomic_json(OUT/'comparacao_por_habilidade.json',pairs)
    atomic_json(OUT/'gradientes_adam.json',{'results':[gradient_report(r) for r in results]})
    atomic_json(OUT/'avaliacoes.json',{'active':base,'pilots':{f"{r['arm']}_s{r['seed']}":r['evaluation'] for r in results},
        'final_test_used':False,'semantic_scores':None,
        'metric_notes':{'semantic_equivalence_closed_proxy':'Campo histórico da rubrica literal limitada. Não é equivalência semântica geral.',
            'semantic_equivalence':'null significa não avaliado; correct=0,total=0 não é zero medido.'}})
    atomic_json(OUT/'datasets_estatisticas.json',{'protocol':protocol,
        'actual_exposure':{f"{r['arm']}_s{r['seed']}":r['exposure'] for r in results},
        'caveat':'Categorias operacionais: regex no último usuário, rótulo correcao e contexto de memoria_dados. Não é anotação humana de intenção. Habilidades originais preservadas nos contadores.',
        'unique_definition':'Identidade textual exata: histórico completo + resposta. Prefixos estilísticos diferentes contam separados. Em language, conta janela (índice+posição), não quantidade de textos/fontes. Não mede diversidade semântica.',
        'full_retention_policy':'Mesma habilidade e EXATA contagem de tokens de resposta do controle por linha; cobertura é do pool elegível, não número visto no piloto.'})
    review=[{'pilot':name,'split':split,**ex} for name,report in pairs.items() for split,part in report.items() for ex in part['changed_examples']]
    atomic_json(OUT/'revisao_humana_pendente.json',{'format':'TAC-controlled-human-review-v1','answers':review,
        'instructions':'Leia o contexto. Preencha semantic_equivalence, off_topic e incomplete com true/false/null. null = não revisado. Não muda pesos nem aprovação.'})
    table=['# Comparação dos pilotos v0.13.0','',
        '| Variante | Seed | Básicas | Contexto + aritmética | Diálogos | Importadas (ouro) | Loss básica | Loss importada completa | EOS | Proteções |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    table.append(f"| Ativa | — | {base['basic']['exact_matches']}/142 | {base['context']['exact_matches']}/175 | {base['natural']['exact_matches']}/65 | {base['imported']['exact_matches']}/60 | {base['basic']['loss']:.6f} | {base['imported_full_loss']['loss']:.6f} | {summary['active']['eos_rate']:.1%} | referência |")
    for r in rows:
        table.append(f"| {LABELS[r['arm']]} | {r['seed']} | {r['basic_count']} | {r['context_count']} | {r['natural_count']} | {r['imported_count']} | {r['basic_loss']:.6f} | {r['imported_loss']:.6f} | {r['eos_rate']:.1%} | {r['checks_count']} |")
    table+=['','Médias e desvios amostrais de 3 seeds (quando completas):','',
        '| Variante | Básicas | Aritmética | Importadas ouro | Importadas histórico próprio | Fatos novos (proxy) |',
        '|---|---:|---:|---:|---:|---:|']
    for a in aggregates:
        vals=[f"{a['metrics'][k]['mean']:.2%} ± {(a['metrics'][k]['std'] or 0):.2%}" for k in ('basic_accuracy','arithmetic_accuracy','imported_accuracy','imported_own_accuracy','novel_factual_proxy')]
        table.append('| '+LABELS[a['arm']]+' | '+' | '.join(vals)+' |')
    table+=['','O painel de importações tem 60 linhas fixas escolhidas por habilidade da validação; a loss usa todas as 353 linhas de validação. Os 668 exemplos importados de teste NÃO foram avaliados. O conjunto final selado NÃO foi aberto.',
        'Contexto na tabela inclui 118 contas e 57 tarefas antigas de contexto/lógica. Consulte os JSONs para separar as habilidades.',
        'Média ± desvio de três seeds não é demonstração estatística de generalização. A rubrica de fatos pode aceitar uma palavra correta em frase errada; semântica permanece sem nota.']
    table+=['','Exposição importada real (média das execuções de cada variante; 528 sorteios por piloto):','',
        '| Variante | Conteúdos únicos | Grupos únicos | Habilidade memoria_nome | correcao | conversa_geral |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in LABELS:
        runs=[r for r in results if r['arm']==arm]
        if not runs:continue
        unique=statistics.mean(r['exposure']['pools']['imported']['unique_examples'] for r in runs)
        groups=statistics.mean(r['exposure']['pools']['imported']['unique_groups'] for r in runs)
        skillmeans=[statistics.mean(r['exposure']['draws_by_original_skill'].get('imported/'+s,0) for r in runs)
            for s in ('memoria_nome','correcao','conversa_geral')]
        table.append('| '+LABELS[arm]+f' | {unique:.2f} | {groups:.2f} | '+' | '.join(f'{v:.2f}' for v in skillmeans)+' |')
    table+=['','O balanceamento usa categorias operacionais amplas; “conversa_geral” nesta tabela é o RÓTULO ORIGINAL do dataset. Não são a mesma coisa. As categorias não garantem aumento de exposição a todos os rótulos raros. Nem unicidade de linha nem ausência de repetição provam diversidade semântica.']
    table+=['','Sondas de gradiente e atualizações REAIS (cinco por piloto):','',
        '| Variante | Sondas | Cosseno antigo/importado negativo | Delta Adam prevê aumento antigo | Perda antiga aumentou > 0,000001 | Perda antiga caiu > 0,000001 |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in LABELS:
        probes=[p for r in results if r['arm']==arm for p in r['gradients_and_adam']]
        if not probes:continue
        neg=sum(p['fixed_probe_pairs']['imported__legacy']['cosine'] is not None and p['fixed_probe_pairs']['imported__legacy']['cosine']<0 for p in probes)
        pred=sum(p['actual_adam_update']['vs_fixed_gradients']['legacy']['dot']>0 for p in probes)
        changes=[p['probe_losses_after']['legacy']-p['probe_losses_before']['legacy'] for p in probes]
        table.append(f"| {LABELS[arm]} | {len(probes)} | {neg} | {pred} | {sum(x>1e-6 for x in changes)} | {sum(x< -1e-6 for x in changes)} |")
    table+=['','O limiar descritivo 0,000001 só separa variações muito pequenas nesta tabela; os deltas sem corte continuam no JSON. Não é critério de aprovação. Lotes de sonda têm 2 exemplos e não representam a distribuição inteira. Produtos com o delta real incluem momentos e decay do AdamW. Não se interpreta o sinal de um gradiente isolado como prova de dano global.']
    (OUT/'COMPARACAO.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    lines=['# TAC v0.13.0 — experimentos controlados','',
        f"Foram concluídos {len(rows)}/15 pilotos de 120 passos. Nenhum modelo foi promovido. O chat continua usando {protocol['parent']} (10.004.544 parâmetros).",
        '', *scientific_findings(results,base,paired), '', '## Protocolo executado','',
        f"Seeds 42, 417 e 903; mesmo checkpoint inicial em todos os braços; AdamW novo, lr nominal 0,00002, weight decay 0,01, betas (0,9; 0,999), epsilon 1e-8, batch 8, acumulação 2, corte 1, KL 2 somente em lotes legacy. Scheduler original de 6.000 passos e warmup 300: taxa real no passo 120 = {results[0]['history'][-1]['learning_rate']:.14f}. Consulte history no JSON para cada passo; ambiente.json registra a assinatura do otimizador instalado.",
        'FP16 autocast na GPU; pesos, momentos e sondas FP32. Sem aumento da rede/contexto, sem dados externos, sem treinamento automático ou alteração das configurações do usuário.',
        'RNGs independentes por tarefa e deslocamento de posição zero são comuns a TODOS os braços. Isso remove contaminação dos sorteios entre tarefas, mas não reproduz bit a bit os pilotos anteriores da v0.12.',
        '', '## Intervenções isoladas','',
        '**A:** mesmos tokens de entrada e janelas textuais. Nas 40 cenas autorais, a loss deixa de cobrar usuário/papéis e passa a cobrar só cada resposta e EOS. Os 304 textos narrativos continuam como escrita. Todos os históricos permanecem disponíveis na atenção causal. Não é uma nova arquitetura.',
        '**B:** somente sorteio de importações muda. Ciclos de categorias e grupos, sem repetir o índice de uma linha até esgotar o pool. Categorias raras são retiradas do ciclo quando acabam; não são recicladas para fingir equilíbrio. Linhas de texto idêntico já existentes podem ainda aparecer; contadores de identidade mostram isso.',
        'Limite de B: as seis categorias são operacionais e amplas. A categoria conversa_geral inclui cumprimentos e identidade; ela NÃO é o rótulo raro conversa_geral original (27 linhas). COMPARACAO.md expõe os rótulos originais separadamente. Não se afirma que todas as habilidades raras foram balanceadas. Não se mudou a classificação no meio da rodada para melhorar resultados.',
        f"**C cobertura:** pool antigo passa de {protocol['current_retention']} a {protocol['full_retention_eligible']} exemplos elegíveis, mantendo habilidade e tokens supervisionados por linha selecionada. O casamento por comprimento limita o subconjunto alcançável; não é amostragem uniforme de todos os 82.512 exemplos nem exposição a todos eles. cobertura_reproduzida.json conta o alcance e os exemplos antes excluídos realmente vistos. **C representativo:** compara com C cobertura mudando só seleção por estratos (tamanho do histórico, negação textual, retido/excluído). Em relação ao controle, C representativo muda cobertura E seleção; não atribuir causalidade a um único fator.",
        '**D:** cinco sondas nos passos 1, 12, 20, 60 e 120; gradientes de todos os parâmetros, lotes fixos de treino de 2 exemplos por tarefa; contribuições dos lotes reais, clipping e delta real depois do AdamW, incluindo momentos e decay. Dot positivo com delta prevê piora local; compara-se com a perda real antes/depois da MESMA atualização. Não foi aplicada cirurgia de gradientes.',
        '', '## Resultados agregados','']
    for a in aggregates:
        m=a['metrics'];lines.append(f"- {LABELS[a['arm']]} ({len(a['seeds'])} seeds): básicas {m['basic_accuracy']['mean']:.2%}; importadas ouro {m['imported_accuracy']['mean']:.2%}; loss básica {m['basic_loss']['mean']:.6f}; loss importada {m['imported_loss']['mean']:.6f}; EOS {m['eos_rate']['mean']:.2%}.")
    lines+=['','Os resultados completos por semente, habilidade e pergunta estão em COMPARACAO.md, comparacao_por_habilidade.json e avaliacoes.json. Não se escolheu uma seed para omitir regressões das outras.',
        '', '## Comparações pareadas (variante menos referência da mesma seed)','']
    for p in paired:
        d=p['mean_delta'];lines.append(f"- {LABELS[p['arm']]} versus {LABELS[p['comparator']]}: básicas {d['basic_accuracy']*100:+.2f} pontos percentuais; importadas ouro {d['imported_accuracy']*100:+.2f}; EOS {d['eos_rate']*100:+.2f}; emissão de Usuário: {d['role_rate']*100:+.2f}; loss básica {d['basic_loss']:+.6f}.")
    lines+=['','Cada piloto tem 66 minilotes importados, 66 revisados, 72 de escrita e 36 antigos, totalizando 240 minilotes/120 atualizações. Cada CE é média por minilote, com peso 1/2 na acumulação; mais tokens não significam automaticamente mais peso na função de perda. KL de peso 2 aparece somente nos minilotes antigos.',
        '', '## Aprovação e limites','',
        f"{sum(r['passed_original_checks'] for r in rows)}/{len(rows)} pilotos atingiram todas as proteções originais. Nenhum foi ativado, mesmo se atingir. Os motivos exatos por execução estão em protection_checks e failed_checks.",
        'Correspondência exata usa as respostas aceitas originais. Os 0/60 exatos no painel importado não informam quantas dessas respostas seriam semanticamente corretas; essa quantidade segue indisponível. As métricas de contexto e conversa revisada mantêm suas regras e limitações conhecidas. Não há avaliador confiável de semântica geral: revisao_humana_pendente.json deixa todos esses campos como null. Resposta curta e repetição de trigramas são sinais descritivos, não notas de inteligência.',
        'Auditoria factual adicional EXPLORATÓRIA: auditoria_factual_fechada.json usa cálculo aritmético e gramáticas restritas de atributos (nome, local, código, sim/não). Não confunde uma palavra de nome numa frase sobre cor com memória do nome. Frases não reconhecidas ficam sem avaliação, não como acertos. A fração verificada usa todas as 28 tarefas fechadas como denominador; também se informa a parcela não avaliada. Isto não é nota semântica e não altera aprovação ou a seleção de variantes.',
        'A avaliação de situações novas é o desenvolvimento já exposto da v0.12, reutilizado e rotulado. Ela não é evidência de teste cego. O teste final reservado e os exemplos importados de teste não foram usados para escolher configurações.',
        '', '## Recursos e preservação','',
        f"GPU: {protocol['gpu_name']}. Pico alocado PyTorch entre os pilotos: {max(r['peak_vram_mb'] or 0 for r in rows):.1f} MiB. A leitura do driver pode incluir memória do Windows e outras aplicações.",
        'Duas tentativas iniciais do controle foram interrompidas pela reserva de RAM antes de 120 passos. Permanecem em preflight* e não entram nas tabelas. O runner passou a manter os vetores das sondas na GPU, fazer hashes em blocos e evitar duplicações integrais FP64. Não se reduziu a reserva de RAM.',
        f"A suíte detectou arredondamento no cosseno (1,0000015 para vetores iguais). O helper foi corrigido para usar a mesma redução no produto e nas normas e limitar o resultado ao intervalo matemático [-1,1]. Os {len(rows)} pilotos concluídos foram executados no mesmo processo com a redução FP32 anterior; os JSONs brutos permanecem como medidos. Esta correção muda somente a instrumentação em reproduções futuras, não o treinamento, os pesos ou critérios. Valores muito próximos de +/-1 devem ser lidos com precisão FP32.",
        'Pesos experimentais da seed 42 são mantidos fora do cadastro da aplicação. Nas demais seeds, o arquivo novo temporário foi excluído depois de registrar seu hash e a avaliação. Não é um checkpoint com momentos para retomada exata; a reprodução começa do pai original com a seed/configuração registrada.',
        'Preservação verificada: '+json.dumps(protected,ensure_ascii=False)+'.',
        'As quatro coleções (biblioteca, conversas importadas, memória de interação e pesos) permanecem separadas. Nenhum livro ou material novo do usuário foi injetado nesses pilotos.',
        '', '## Reproduzir e visualizar','',
        'Execute `.venv\\Scripts\\python.exe -X utf8 scripts\\run_controlled_experiments.py`. Execuções concluídas são verificadas e preservadas; não são sobrescritas. Para reproduzir em pasta nova: acrescente `--output reports/experimentos_0130_repeticao`. Passe a mesma opção aos scripts report_controlled_experiments.py, audit_controlled_exposure.py e audit_controlled_facts.py. Não apague os resultados originais.',
        'Na TAC: Ferramentas avançadas → Treinamento e versões → resultados dos experimentos de conversa. A tela apenas lê arquivos locais.',
        'Os JSONs de respostas podem conter trechos das conversas importadas do próprio usuário. Permanecem locais; nenhum envio foi feito.']
    if (OUT/'testes_final.xml').exists():
        suites=ET.parse(OUT/'testes_final.xml').getroot().findall('testsuite')
        total=sum(int(s.attrib['tests']) for s in suites);failed=sum(int(s.attrib['failures'])+int(s.attrib['errors']) for s in suites)
        lines+=['','## Validação do software','',f'Suíte completa final: {total} testes, {failed} falhas/erros, em CPU. Os 15 pilotos usaram CUDA real. O primeiro teste completo teve 215 acertos e uma falha numérica de cosseno; a correção acrescentou uma verificação de limites matemáticos e a suíte foi executada novamente.',
            'O BAT iniciou em uma instalação de teste isolada (smoke UI, código de saída 0). O painel foi capturado também com os dados reais da instalação, com temporizador e automação interrompidos. Capturas: capturas/comparacao.png, todos_os_pilotos.png, detalhes.png e fatos_verificados.png.',
            'Os sorteios e contadores dos 15 pilotos foram reconstituídos sem carregar pesos. Os cinco checkpoints experimentais retidos e o ativo foram carregados e tiveram hashes/10.004.544 parâmetros conferidos. validacao_final.json registra a checagem dos JSONs, critérios, dados e artefatos; manifesto_relatorios.json contém hashes sem referência circular a si mesmo.']
    (OUT/'RELATORIO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    problems=['# Problemas confirmados e limites da evidência','',
        '1. **Objetivos incompatíveis em fronteiras de resposta, demonstrados no dado:** o replay original cobra continuação com a próxima fala do usuário; o SFT cobra EOS. A variante A corrige 160 fronteiras em 40 cenas. Isto confirma conflito de alvos nesses pontos; melhora de conversa só pode ser afirmada a partir dos resultados, não da inspeção estrutural.',
        '2. **Exposição original desigual:** compare draws_by_category, draws_by_original_skill e supervised_tokens_by_category de controle/B para as mesmas seeds. O balanceamento não garante igual número de tokens nem igual qualidade dos exemplos. Registra-se o orçamento real.',
        '3. **Replay filtrado:** os 41.119 exemplos atuais são subconjunto dos 82.512 anteriores. A exclusão histórica era intencional (valores sintéticos/pseudopalavras); ampliá-la pode ajudar cópia de nomes e também prejudicar linguagem natural. C não assume que mais dados sejam melhores.',
        '4. **Loss e respostas não são a mesma medida:** uma probabilidade ligeiramente menor em muitos tokens eleva NLL sem mudar o argmax; perdas/ganhos exatos estão separados por habilidade. Não se interpreta aumento isolado de loss como perda de todo o conhecimento.',
        '5. **Interferência medida localmente:** gradientes negativos não são, por si, causalidade. gradientes_adam.json registra produto com a atualização real e perdas antes/depois. Sondas de 2 exemplos não representam todas as tarefas. Não há justificativa nesta etapa para manipulação complexa de gradientes.',
        '6. **Limites de avaliação mantidos explícitos:** literal pode rejeitar paráfrases válidas; regras de palavras podem aceitar frases erradas. Semântica, mudança de assunto e incompletude precisam de revisão contextual humana. Nenhuma nota foi preenchida automaticamente.',
        '7. **Memória disponível limitou a instrumentação inicial:** a proteção interrompeu duas tentativas, preservando o ativo. A solução foi alocação das sondas, sem aumentar rede ou diminuir reservas.']
    problems+=['','Medições que delimitam estas conclusões:',*scientific_findings(results,base,paired)[2:]]
    (OUT/'PROBLEMAS_CONFIRMADOS.md').write_text('\n\n'.join(problems)+'\n',encoding='utf-8')
    next_steps=['# Próximos passos','',
        '1. Revisar as respostas alteradas na ficha humana. Separar paráfrases corretas, respostas parciais, contradições e mudança de assunto. Não treinar sobre o teste final nem usar a revisão para fabricar uma nota semântica automática.',
        '2. Conferir a consistência entre as três seeds na tabela, inclusive regressões por habilidade. Escolher hipótese para uma segunda rodada somente após essa análise; nenhum piloto é promovido por ganho isolado em importações.',
        '3. A melhora de EOS em respostas ouro na seed 42 justifica manter A como hipótese experimental, mas não demonstra ganho em conversa livre. Não iniciar uma rodada longa com base apenas nisso. Repetir sondas em mais seeds exigiria reproduzir os pesos temporários removidos; não foram inventadas medições ausentes.',
        '4. Medir validação de desenvolvimento nos passos 120, 300 e 600 e manter todas as proteções anteriores. Um ganho importado não compensa regressões graves em nomes, correções ou respostas básicas. Guardar o teste final até congelar a decisão experimental.',
        '5. Próximo B, ainda em 120 passos/3 seeds: sortear por habilidade original além da categoria ampla, sem reposição, assegurando que as 27 linhas conversa_geral sejam tratadas como rótulo próprio e não diluídas em cumprimentos/identidade. Reavaliar rótulos usando o objetivo da resposta e o histórico; preservar exemplos válidos e divisões por grupo. Não combinar essa mudança com A ou C.',
        '6. Próximo D, ainda em 120 passos/3 seeds: comparar a ordem atual em blocos (11 passos importados/revisados, 6 de escrita, 3 antigos) com distribuição dos mesmos lotes ao longo do ciclo. Fixar contagens, tokens, seeds, ponto de partida e scheduler. É uma hipótese de exposição e momentos do AdamW, não uma causa já provada. Evitar cirurgia de gradientes.',
        '7. Somente após ganho consistente no desenvolvimento e preservação das habilidades antigas propor 600 passos controle versus a variante isolada, com proteções originais. Não executar essa rodada longa agora. Manter o teste final reservado até congelar a decisão.',
        '8. Ampliar dados variados com respostas contextualizadas e histórias completas mediante autorização. Identidade textual única não prova diversidade: há variações de prefixo e janelas do mesmo texto. Repetir um conjunto pequeno com maior frequência não cria novos conceitos.']
    (OUT/'PROXIMOS_PASSOS.md').write_text('\n\n'.join(next_steps)+'\n',encoding='utf-8')
    print(json.dumps({'completed':len(rows),'preservation':protected,'all_fair':all(all(v for k,v in r.items() if type(v)is bool) for r in fair)},ensure_ascii=False))


if __name__=='__main__':main()
