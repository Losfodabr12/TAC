"""Assemble final evidence only after actual training, inference and review."""
import html
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tac.config import atomic_json

BASE=ROOT/'data/tac_v3';OUT=ROOT/'reports/tac_v3/rodada_20261018_01'


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def cell(value):return html.escape(str(value)).replace('|','&#124;').replace('\n','<br>')
def number(value):return f'{value:,}'.replace(',','.')


def main():
    training=read(BASE/'training_result.json');integrity=read(OUT/'integridade_v3.json')
    preservation=read(OUT/'v2_preservacao_depois.json');interface=read(OUT/'interface_verificada.json')
    review=read(OUT/'REVISAO_GERACOES.json');deployment=read(BASE/'deployment.json')
    corpus=read(BASE/'language_ready.json');data=read(BASE/'sft_reviewed_ready.json')
    assert integrity['ok'] and preservation['ok'] and interface['status']=='OK_interface_only_not_quality_approval'
    labels=['v2_referencia','v2_conversacao','v3_linguagem','v3_epoca1','v3_final']
    generations={label:read(OUT/('geracoes_'+label+'.json')) for label in labels}
    language_scores={label:read(OUT/('linguagem_'+label+'.json')) for label in labels}
    summaries={}
    for label in labels:
        rows=review['models'][label]['rows'];raw=generations[label]['outputs']
        assert {r['id'] for r in rows}=={r['id'] for r in raw}
        byid={r['id']:r for r in rows};mainrows=[r for r in raw if not r['known_probe']]
        categories={}
        for category in sorted({r['category'] for r in mainrows}):
            selected=[r for r in mainrows if r['category']==category]
            categories[category]={'correct':sum(byid[r['id']]['appropriate'] for r in selected),'total':len(selected)}
        summaries[label]={'categories':categories,'main_appropriate':sum(byid[r['id']]['appropriate'] for r in mainrows),
            'main_total':len(mainrows),'loops':sum(r['decoding']['stop_reason']=='repetition' for r in raw),
            'known_appropriate':sum(byid[r['id']]['appropriate'] for r in raw if r['known_probe']),
            'context_factual_correct':sum(byid[r['id']].get('context_task_factual_correct',byid[r['id']]['appropriate'])
                for r in mainrows if r['category'] in ('temporal_context','context_binding','context_update','name_copy')),
            'length_stops':sum(r['decoding']['stop_reason']=='length' for r in raw),
            'eos_stops':sum(r['decoding']['stop_reason']=='eos' for r in raw),
            'truncated_inputs':sum(r['decoding']['truncated_tokens']>0 for r in raw),
            'bits_per_utf8_byte':language_scores[label]['bits_per_utf8_byte']}
    summary={'training':training,'generation_summaries':summaries,'selected':deployment,'corpus':corpus,
             'reviewed_sft_counts':data['reviewed_sft_counts'],'preservation':preservation,
             'semantic_review_is_not_independent_human_certification':True,
             'language_quality':language_scores}
    atomic_json(OUT/'RESULTADOS.json',summary)
    lines=['# TAC v3 — resultados reais','',review['conclusion'],'',
           f"Nova rede: **{number(training['parameters'])} parâmetros**, inicializados aleatoriamente. "
           f"Foram concluídas **{number(training['global_steps'])} atualizações**, com "
           f"{number(training['language_tokens'])} tokens supervisionados de linguagem (incluindo replay) e "
           f"{number(training['sft_target_tokens'])} tokens de resposta. "
           f"As {number(training['sft_unique'])} instruções elegíveis tiveram exatamente duas apresentações, "
           f"totalizando {number(training['sft_presentations'])} exposições.",'',
           f"Tempo acumulado informado pelo runner: {training['seconds']/3600:.2f} h "
           '(inclui suas validações e salvamentos; exclui preparação e avaliação livre posterior); '
           f"pico registrado no processo final: {training['peak_vram_mib']:.0f} MiB de VRAM. CUDA/BF16 na RTX 3050, "
           'com estados/pesos FP32, controle de RAM/VRAM e checkpoints completos novos.','',
           '## Geração livre e generalização','',
           'São 27 casos principais novos e 5 sondas conhecidas separadas. A revisão qualitativa por inspeção do agente '
           'fica em REVISAO_GERACOES.json; não é uma certificação humana independente. As gerações originais permanecem intactas.','',
           '| Modelo | Adequadas / 27 | Matemática | Contexto e cópia integrais | Sondas conhecidas / 5 | Loops / 32 | Limite / 32 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for label in labels:
        s=summaries[label];cats=s['categories'];math=cats['arithmetic']
        contexts=[v for k,v in cats.items() if k in ('temporal_context','context_binding','context_update','name_copy')]
        lines.append(f"| {label} | {s['main_appropriate']}/27 | {math['correct']}/{math['total']} | "
                     f"{sum(x['correct'] for x in contexts)}/{sum(x['total'] for x in contexts)} | {s['known_appropriate']} | {s['loops']} | {s['length_stops']} |")
    lines+=['','Acerto exige pertinência e correção da resposta, com critérios por tarefa. Fluência e contenção de nomes '
            'não bastam. Os números são de desenvolvimento, em amostra pequena; arquitetura, corpus e tokenizer mudaram juntos. '
            'Não há atribuição causal isolada ao número de parâmetros.','',
            'Crédito parcial: os dois checkpoints supervisionados copiam **Arvélio** com acento, mas acrescentam texto '
            'apesar do pedido de somente o nome. Ambos resumem apenas “A feira foi adiada”, fato correto, '
            'omitindo a chuva e a praça da síntese esperada. Esses casos não contam como cumprimento integral. '
            'A primeira passagem recupera Luzia como primeiro nome; o final perde esse acerto. Assim, contexto/cópia '
            'factual, separado do formato, fica em **2/7** na primeira passagem e **1/7** no final. '
            'O cumprimento “Apa, tudo bom” recebe o mesmo tratamento tolerante de grafia dado ao “Opaa” da v2.','',
            '## Linguagem, loss e checkpoints','',
            '| Modelo | Bits por byte nas mesmas 32 passagens inéditas (menor é melhor) | EOS / 32 | Contextos truncados |',
            '|---|---:|---:|---:|']
    for label in labels:
        s=summaries[label]
        lines.append(f"| {label} | {s['bits_per_utf8_byte']:.4f} | {s['eos_stops']} | {s['truncated_inputs']} |")
    initial=read(OUT/'initial_validation_revised_sft.json')
    language_end=read(BASE/'language_result.json');epoch1=read(OUT/'sft_epoch_1.json')
    lines+=['','A previsão de linguagem melhora frente à v2 nas passagens comparadas, mas piora após o ajuste '
            'de instruções em relação ao fim do pré-treinamento. Isso não equivale a compreensão factual. '
            'A métrica por byte permite comparar os tokenizers; loss por token entre v2 e v3 não é diretamente comparável.','',
            '| Etapa v3 | Loss linguagem (32 janelas) | Loss respostas (96 exemplos) |',
            '|---|---:|---:|']
    # The revised initial baseline uses the same SFT validation selection as later checkpoints.
    for name,metrics in [('Aleatória, antes do treino',initial['validation']),('Fim linguagem',language_end['validation']),('Primeira passagem',epoch1['validation']),('Final',training['validation'])]:
        lines.append(f"| {name} | {metrics['language']['loss']:.4f} | {metrics['instruction']['loss']:.4f} |")
    lines+=['','A avaliação supervisionada usa uma amostra fixa de 96 dos 985 exemplos elegíveis de validação, '
            'e a avaliação de linguagem usa 32 janelas; a amostra separada por byte tem 16 passagens de cada fonte. '
            'Não se alega ter avaliado exaustivamente toda a validação. A seleção revisada de validação SFT '
            'não contém categoria de resumo; o caso aberto 21 é a única sonda de resumo desta comparação.','',
            '| Etapa | Checkpoint |', '|---|---|',
            f"| Fim linguagem | `{language_end['checkpoint']}` |",
            f"| Primeira passagem, selecionada experimentalmente | `{epoch1['checkpoint']}` |",
            f"| Final, preservado para comparação e recuperação | `{training['checkpoint']}` |",'',
            'A primeira passagem foi salva com 21.776 apresentações (seis exemplos já da segunda), '
            'devido à fronteira do lote. O final tem exatamente duas apresentações por exemplo. '
            'Nas sessões com respostas próprias, os candidatos carregam erros de animais para matemática e nome; '
            'não recuperam Safira. A montagem usa apenas mensagens da própria sessão.','',
            '## Exemplos conhecidos, antes e depois','',
            '| Pergunta | v2 após 4.000 passos | v3 selecionada |','|---|---|---|']
    chosen=deployment['selected_label'];old={r['id']:r for r in generations['v2_conversacao']['outputs']}
    for row in generations[chosen]['outputs']:
        if row['known_probe']:
            lines.append('| '+cell(row['messages'][-1][1])+' | '+cell(old[row['id']]['actual'])+' | '+cell(row['actual'])+' |')
    lines+=['','## Corpus e proteção','',
            'Retido aproximadamente 1 GiB de texto novo: FineWeb2 em português e Wikipedia, com 389.459 documentos '
            'de treino e 14.912 de validação. Há 281.440.899 tokens disponíveis para linguagem de treino; '
            'o pré-treinamento processou 120.004.608 tokens e o replay acrescentou 4.947.968, sem repetir janelas. '
            'Os alvos efetivamente processados somam 90.153.054 tokens FineWeb2 e 34.799.522 Wikipedia. '
            'A seleção SFT efetiva tem 21.770 exemplos de treino e 985 de validação; 2.507 exemplos de treino '
            'incluem múltiplos turnos e 1.200 são cálculos determinísticos verificados, usados apenas como dados de estudo. '
            'As contas e perguntas novas congeladas não foram adicionadas ao treino. Fontes, revisões, filtros e condições '
            'estão em [METODOLOGIA.md](METODOLOGIA.md) e [LICENSES.md](../../../data/tac_v3/LICENSES.md).','',
            'Defeitos confirmados nos alvos públicos foram corrigidos **antes da primeira atualização supervisionada**. '
            'A transição da seleção de instruções preservou todos os tokens/tokenizer de linguagem, pesos, otimizador '
            'e RNG no passo 2.048. Originais e manifestos iniciais permanecem disponíveis. Ver '
            '[REVISAO_SFT_ANTES_DO_USO.md](REVISAO_SFT_ANTES_DO_USO.md).','',
            f"Integridade: {len(integrity['checkpoints'])} checkpoints v3 carregados estritamente, com hashes e valores finitos; "
            f"{preservation['original_files']} arquivos originais conferidos, sem alteração ou perda detectada. "
            'Origens e hashes dos documentos registrados na biblioteca foram conferidos; essa verificação não '
            'constitui uma auditoria de todos os arquivos pessoais do disco. Nenhuma exclusão foi feita. '
            'Nenhum teste final reservado foi usado; '
            'nenhum modelo/API externo gerou respostas nesta execução.','',
            '## Interface e limites','',
            'Abra INICIAR_TAC.bat, entre em Conversar e escolha **TAC v3 — experimental, 29,4 milhões**. '
            'A v2 de conversação e sua referência anterior continuam no seletor. A troca abre uma conversa limpa. '
            'A prova nativa confirmou geração do checkpoint v3, 29.414.400 parâmetros, fontes documentais vazias '
            'e restauração da versão anterior. A prova nativa também testa as entradas informais sem acento, '
            'em produção FP32; os scripts usam BF16, por isso suas respostas podem divergir e são registradas separadamente. '
            'Disponibilidade experimental não é promoção por qualidade. **46 testes de software passaram**; '
            'eles verificam integração e proteções, não inteligência.','',
            review['limitations'],'',review['recommendation'],'',
            'Comparação por pergunta: [COMPARACAO_MODELOS.md](COMPARACAO_MODELOS.md). '
            'Falhas e diagnósticos: [FALHAS.md](FALHAS.md). '
            'Integridade, retomada e verificação nativa ficam nos JSONs desta pasta.']
    (OUT/'RELATORIO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    comparison=['# Comparação pergunta por pergunta','',
                'Entradas independentes iguais. Respostas completas e metadados de parada/contexto nos geracoes_*.json.','',
                '| ID | Categoria | Entrada | Esperado | v2 conversação | v3 selecionada |','|---:|---|---|---|---|---|']
    for row in generations[chosen]['outputs']:
        comparison.append('| '+' | '.join(map(cell,[row['id'],row['category'],json.dumps(row['messages'],ensure_ascii=False),
                            row['expected'],old[row['id']]['actual'],row['actual']]))+' |')
    comparison+=['','## Todos os checkpoints, com diagnóstico separado','']
    for original in generations[chosen]['outputs']:
        comparison+=['### Caso '+str(original['id']),'','Entrada: '+cell(json.dumps(original['messages'],ensure_ascii=False)),
                     '', 'Esperado: '+cell(original['expected']),'',
                     '| Modelo | Resposta completa | Cumprimento integral | Diagnóstico |','|---|---|---|---|']
        for label in labels:
            row=next(r for r in generations[label]['outputs'] if r['id']==original['id'])
            grade=next(r for r in review['models'][label]['rows'] if r['id']==original['id'])
            comparison.append('| '+' | '.join(map(cell,[label,row['actual'],'sim' if grade['appropriate'] else 'não',grade['diagnosis']]))+' |')
    (OUT/'COMPARACAO_MODELOS.md').write_text('\n'.join(comparison)+'\n',encoding='utf-8')
    failures=['# Falhas observadas','',review['limitations'],'',
              'Os defeitos confirmados da fonte pública e a revisão dos alvos ainda não utilizados estão em '
              'REVISAO_SFT_ANTES_DO_USO.md. As exclusões conservadoras não significam que todos os alvos excluídos fossem falsos.','',
              '## Respostas inadequadas da v3 selecionada','']
    reviewrows={r['id']:r for r in review['models'][chosen]['rows']}
    for row in generations[chosen]['outputs']:
        grade=reviewrows[row['id']]
        if not grade['appropriate']:
            failures+=['### Caso '+str(row['id']), '', 'Pergunta: '+cell(row['messages'][-1][1]),'',
                       'Saída: '+cell(row['actual']),'','Diagnóstico: '+grade['diagnosis'],'']
    (OUT/'FALHAS.md').write_text('\n'.join(failures),encoding='utf-8')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
