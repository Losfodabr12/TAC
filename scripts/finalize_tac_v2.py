"""Confere o resultado concreto, preservação e escreve o fechamento da TAC v2."""
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
BASE=ROOT/'data/tac_v2';OUT=ROOT/'reports/reconstrucao_zero/execucao_v2'
from tac.neural.v2_data import load_jsonl


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def main():
    result=read(BASE/'training_result.json');meta=read(BASE/'encoded_ready.json');corpus=read(BASE/'corpus_ready.json')
    raw=read(OUT/'geracao_livre.json');before=read(BASE/'initial_validation.json')
    lang=read(BASE/'language_result.json');deleted=read(OUT/'deletion_result.json')
    assert result['generation']=='TAC-v2' and result['status']=='experimental_disponivel'
    assert result['parameters']==10004544 and result['pretraining_steps']==12000
    assert result['instruction_presentations']==meta['sft']['train']['examples']*3
    assert result['instruction_supervised_tokens']==meta['sft']['train']['supervised_tokens']*3
    assert result['initial_weight_hash']!=result['final_weight_hash']
    assert not result['old_weights_loaded'] and not result['external_models_used'] and not result['reserved_tests_used']
    for name,info in meta['files'].items():assert sha(BASE/name)==info['sha256']
    for name,info in corpus['files'].items():assert sha(BASE/name)==info['sha256']
    exposure=list(load_jsonl(BASE/'exposure.jsonl'));counts=Counter(x['kind'] for x in exposure)
    assert len(exposure)==result['steps']==len({r['step'] for r in exposure})
    assert [x['step'] for x in exposure]==list(range(1,result['steps']+1))
    assert sum(x['supervised_tokens'] for x in exposure if x['kind']=='language')==result['language_supervised_tokens']
    assert sum(x['supervised_tokens'] for x in exposure if x['kind']=='instruction')==result['instruction_supervised_tokens']
    ids=Counter(i for r in exposure if r['kind']=='instruction' for i in r['indices'])
    assert len(ids)==meta['sft']['train']['examples'] and set(ids.values())=={3}
    old=read(ROOT/'reports/auditoria_pesos_e_dados/rodada_20261017_01/preservacao_depois.json')
    personal={k:v for k,v in old['sha256'].items() if k.startswith(('data/books/','data/inbox/','data/dialogue_imports/','materials/'))}
    assert all((ROOT/k).is_file() and sha(ROOT/k)==v['sha256'] for k,v in personal.items())
    reserved=old['reserved_metadata_only'];changed_reserved=[]
    for relative,info in reserved.items():
        p=Path(relative)
        if not p.is_absolute():p=ROOT/p
        if info.get('exists') is False:continue
        if not p.is_file() or p.stat().st_size!=info['bytes'] or p.stat().st_mtime_ns!=info['mtime_ns']:changed_reserved.append(str(p))
    assert not changed_reserved
    con=sqlite3.connect((ROOT/'data/tac.sqlite3').as_uri()+'?mode=ro',uri=True);con.row_factory=sqlite3.Row
    active=con.execute("SELECT value FROM meta WHERE key='active_conversation'").fetchone()[0]
    assert active==result['checkpoint']
    sources=[dict(r) for r in con.execute('SELECT id,title,kind,content_hash FROM sources')]
    expected=read(ROOT/'reports/auditoria_pesos_e_dados/rodada_20261017_01/metadados_biblioteca.json')['sources']
    assert {r['id']:r['content_hash'] for r in sources}=={r['id']:r['content_hash'] for r in expected}
    n_chunks=con.execute('SELECT COUNT(*) FROM chunks').fetchone()[0];assert n_chunks==7726
    oldfiles=read(OUT/'deletion_manifest.json')['files']
    assert all(not (ROOT/r['file']).exists() for r in oldfiles)
    current_cps=[dict(r) for r in con.execute('SELECT filename,step,status FROM checkpoints WHERE run_id=?',(result['run_id'],))]
    assert all((ROOT/'data/checkpoints'/r['filename']).is_file() for r in current_cps)
    validation={'status':'OK','generation':'TAC-v2','parameters':result['parameters'],'active_checkpoint':active,
        'checkpoint_sha256':sha(ROOT/'data/checkpoints'/(active+'.pt')),'new_checkpoints_registered':len(current_cps),
        'confirmed_optimizer_updates':len(exposure),'language_updates':counts['language'],'instruction_updates':counts['instruction'],
        'each_instruction_example_presented_exactly_3_times':True,'all_dataset_file_hashes_match':True,
        'old_artifacts_deleted':deleted['files_deleted'],'old_weights_backups_created':0,
        'preserved_personal_and_material_files_verified':len(personal),'source_content_hashes_unchanged':True,
        'library_sources':len(sources),'library_chunks':n_chunks,'reserved_metadata_entries_unchanged':len(reserved),
        'reserved_test_contents_opened':False,'free_generation_examples':len(raw['outputs']),
        'empty_outputs':raw['empty_outputs'],'unicode_replacement_outputs':raw['unicode_replacement_outputs'],
        'external_models_or_generation_apis_used':False}
    with (OUT/'validacao_final.json').open('x',encoding='utf-8') as f:json.dump(validation,f,ensure_ascii=False,indent=2)
    after=result['validation_losses'];pre=lang['validation']
    report=f'''# TAC v2 — fechamento da reconstrução

**Concluído: corpus público preparado, rede própria reinicializada, treinamento real em CUDA e TAC v2 experimental ativada na interface.** A disponibilização atende ao pedido de testar a nova rede; não constitui certificação de conversa geral ou factualidade.

- Parâmetros únicos: **{result['parameters']:,}**. Inicialização nova, seed 20261017; nenhuma herança de pesos antigos.
- Dispositivo: {result['device']}, RTX 3050 6 GiB; precisão {result['precision']}.
- Atualizações confirmadas: **{result['steps']:,}** — 12.000 de linguagem inicial e {result['instruction_phase_updates']:,} na etapa seguinte, incluindo replay textual novo.
- Tokens de linguagem efetivamente supervisionados: **{result['language_supervised_tokens']:,}**. Tokens-alvo de instrução: **{result['instruction_supervised_tokens']:,}**.
- Exemplos SFT: 5.393 distintos, cada um apresentado exatamente três vezes, **{result['instruction_presentations']:,} apresentações**. Não há repetição artificial de exercícios antigos.
- Descartes de precisão: {result['precision_skips']}. Checkpoints registrados desta execução nova: {len(current_cps)}.
- Checkpoint ativo: `{active}`. SHA-256: `{validation['checkpoint_sha256']}`.

## Perdas em desenvolvimento

Mesmas amostras fixas de validação: 32 janelas de linguagem/16.384 tokens e as primeiras 64 respostas SFT/6.827 alvos (Aya, pela ordem do arquivo). Média de entropia cruzada por token, não porcentagem de acerto factual. A avaliação suplementar de todas as 156 respostas, por fonte e número de turnos, está em `geracao_livre.json`; não substitui ou altera a métrica fixa anterior.

| Momento | Linguagem | Instruções |
|---|---:|---:|
| Pesos aleatórios | {before['language']['loss']:.4f} | {before['instruction']['loss']:.4f} |
| Após pré-treinamento | {pre['language']['loss']:.4f} | {pre['instruction']['loss']:.4f} |
| Após instruções/conversa | {after['language']['loss']:.4f} | {after['instruction']['loss']:.4f} |

As perdas demonstram aprendizado de previsão nos conjuntos medidos. Não foram usadas para executar dezenas de pilotos, ajustar os critérios de aprovação antigos ou prometer inteligência geral. A avaliação é de desenvolvimento; testes reservados e testes oficiais das fontes ficaram fechados.

## Geração livre e limites

`geracao_livre.json` contém **{len(raw['outputs'])} saídas integrais**, greedy, sem biblioteca/RAG, correção por regra ou respostas substitutas: exemplos públicos de validação e sondas manuais novas de explicação, criação, instrução e contexto. Saídas vazias: {raw['empty_outputs']}; saídas com caractere de substituição Unicode: {raw['unicode_replacement_outputs']}. Esses contadores não equivalem a qualidade semântica; as respostas devem ser lidas, inclusive os erros.

**A geração livre ainda é fraca.** Houve repetição intensa e erros factuais. Na leitura qualitativa das sondas manuais, a rede recuperou `Iolanda`, mas falhou na mudança da visita de quarta para sábado, na cor verde da mochila e em 8+6. A explicação de secagem de roupa e a história solicitada produziram texto incoerente/repetitivo. Não se apresenta a queda de perda como prova de boa conversação, nem se corrige o texto gerado para melhorar a impressão. Essa leitura é pós-treino e não é uma estimativa de acurácia geral. Não foi iniciada outra rodada para perseguir as sondas.

A rede é pequena e o conjunto português supervisionado tem tamanho limitado. Pode falhar em factualidade, coerência longa, aritmética, intenções e memória. O contexto máximo é 1.024 tokens; a interface mantém a conversa recente. Filtros heurísticos não certificam ausência de toda PII ou de todo erro factual nas fontes. As condições das fontes estão em `data/tac_v2/LICENSES.md`, com licenças/cards originais guardados.

## Preservação e acesso

Foram removidos apenas os **{deleted['files_deleted']} artefatos antigos** do manifesto validado, sem backup desses pesos/datasets. O descarte ocorreu depois do corpus pronto, dos testes do treinador e do checkpoint aleatório novo. Backups mistos antigos foram preservados por também conterem documentos; não participam do treino ou da retomada da TAC v2.

Foram conferidos {len(personal)} arquivos de livros, inbox, importações e materiais: hashes intactos. A biblioteca mantém 43 fontes/7.726 chunks e hashes de conteúdo inalterados. Não houve exclusão de código, interface ou documentos pessoais. As alterações de código adicionam treino/retomada v2 e compatibilidade do novo contexto. As referências antigas foram marcadas como descartadas e o treino automático antigo desligado.

Abra `INICIAR_TAC.bat` e use **Conversar com a TAC v2 — experimental**. A geração neural não consulta seus arquivos nesse modo. A consulta documental continua no seu modo próprio. A retomada, se necessária, usa `scripts/train_tac_v2.py` ou o controle TAC v2 da interface; somente checkpoints da geração nova são aceitos.

Não foi iniciada outra rodada após concluir o orçamento. `validacao_final.json`, `corpus_ready.json`, `encoded_ready.json`, `exposure.jsonl` e os checkpoints registram o que efetivamente foi executado.
'''
    with (OUT/'FECHAMENTO.md').open('x',encoding='utf-8') as f:f.write(report)
    readme=ROOT/'README.md';previous=readme.read_text(encoding='utf-8')
    marker='<!-- TAC-V2-CONCLUIDA -->'
    if marker not in previous:
        introduction=f'''{marker}
# TAC v2 experimental

Reconstrução concluída: **10.004.544 parâmetros novos**, pré-treinamento FineWeb2 em português e ajuste Aya/OASST2. Abra `INICIAR_TAC.bat` e escolha **Conversar com a TAC v2 — experimental**. A conversa neural usa a rede própria; a biblioteca continua preservada no modo documental.

Execução: {result['steps']:,} atualizações confirmadas em CUDA, com checkpoints recuperáveis da geração nova. Relatório e limitações: [fechamento](reports/reconstrucao_zero/execucao_v2/FECHAMENTO.md). Fontes/condições: [licenças](data/tac_v2/LICENSES.md). A geração livre observada ainda é fraca, com repetições e erros factuais; a versão serve para testes e não tem boa conversação certificada.

O treino automático antigo está desligado. O botão de retomada v2 e `scripts/train_tac_v2.py` continuam somente a execução nova. Nenhum exercício antigo é a base desse treino. Foram excluídos apenas os artefatos generativos antigos autorizados; livros, documentos, biblioteca e programa foram preservados.

---

'''
        readme.write_text(introduction+previous,encoding='utf-8')
    print(json.dumps(validation,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
