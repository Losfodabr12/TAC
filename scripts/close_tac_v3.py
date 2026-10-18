"""Close the completed v3 round from evidence. No training or weight writes."""
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tac.config import atomic_json
from tac.core import Core

OUT = ROOT / 'reports/tac_v3/rodada_20261018_01'
BASE = ROOT / 'data/tac_v3'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(2**20):
            h.update(block)
    return h.hexdigest()


def main():
    core = Core(ROOT)
    result = read(BASE/'training_result.json')
    progress = read(BASE/'progress.json')
    integrity = read(OUT/'integridade_v3.json')
    preservation = read(OUT/'v2_preservacao_depois.json')
    interface = read(OUT/'interface_verificada.json')
    review = read(OUT/'REVISAO_GERACOES.json')
    deployment = read(BASE/'deployment.json')
    prepared = read(OUT/'preparacao_verificada.json')
    assert progress['phase'] == 'complete' and progress['checkpoint'] == result['checkpoint']
    assert result['global_steps'] == integrity['committed_updates'] == 20696
    assert result['sft_presentations'] == integrity['instruction_presentations'] == 43540
    assert integrity['ok'] and preservation['ok']
    assert interface['status'] == 'OK_interface_only_not_quality_approval'
    assert interface['v2_restored'] and interface['configuration_bytes_unchanged']
    assert core.active_conversation_id() == preservation['v2_active_before']
    assert not deployment['quality_approved'] and not deployment['automatic_promotion']
    assert sha(OUT/'avaliacao_dev.json') == prepared['development_file_sha256']
    assert sha(ROOT/'tac/neural/v3_training.py') == read(OUT/'codigo_retomada_apos_revisao.json')['v3_training.py']
    selected = next(cp for cp in integrity['checkpoints'] if cp['checkpoint'] == deployment['checkpoint'])
    assert sha(core.conversation.storage.checkpoint_path(selected['checkpoint'])) == selected['file_sha256']
    cases = read(OUT/'avaliacao_dev.json')['cases']
    labels = ['v2_referencia', 'v2_conversacao', 'v3_linguagem', 'v3_epoca1', 'v3_final']
    generations = []
    for label in labels:
        d = read(OUT/('geracoes_'+label+'.json'))
        assert len(d['outputs']) == len(cases) == 32
        for row, frozen in zip(d['outputs'], cases):
            for key in ('id', 'messages', 'expected', 'known_probe'):
                assert row[key] == frozen[key]
            assert row['decoding']['truncated_tokens'] == 0
        history = []
        for turn in d['self_generated_session']:
            assert turn['input_messages'][:-1] == history
            history = turn['input_messages'] + [['assistant', turn['actual']]]
        assert len(d['self_generated_session']) == 5
        generations.append(len(d['outputs'])+len(d['self_generated_session']))
    suites = list(ET.parse(OUT/'testes.xml').iter('testsuite'))
    tests = sum(int(s.get('tests', 0)) for s in suites)
    assert tests == 46 and all(int(s.get('errors', 0)) == int(s.get('failures', 0)) == 0 for s in suites)
    # Preserve the earlier in-progress note as history in place, clearly superseded.
    state = OUT/'ESTADO_EXECUCAO.md'
    old = state.read_text(encoding='utf-8')
    marker = '<!-- TAC-V3-FINAL -->'
    if not old.startswith(marker):
        final = f'''{marker}
# Estado final — TAC v3

**Treinamento, avaliação, relatórios e integração concluídos em 18/10/2026. Qualidade de conversação insuficiente. Não iniciar outra rodada automaticamente.**

Run único `{result['run_id']}`: **{result['global_steps']:,} atualizações confirmadas**, sem passos ausentes ou duplicados. O treino terminou às 06:41:21 UTC (03:41:21 em São Paulo), com {result['seconds']:.2f} segundos acumulados no runner. Preparação e avaliação livre posterior não fazem parte desse tempo.

- Rede própria com **29.414.400 parâmetros novos**, inicialização aleatória e CUDA/BF16.
- Pré-treinamento: **120.004.608 tokens**; replay: **4.947.968**; respostas: **7.539.174**.
- **21.770 instruções × duas apresentações = 43.540**, com nenhuma janela de linguagem repetida.
- **11 checkpoints completos novos** conferidos por hash, carregamento estrito e valores finitos. Retomada real no passo 2.048 comprovada, sem reiniciar os pesos.
- **185 gerações** de comparação entre cinco modelos/etapas, mais nove na prova nativa da interface. São 27 tarefas novas, cinco sondas conhecidas e sessão própria por modelo.
- **46 testes de software passaram**. Integração e testes não aprovam a qualidade semântica.
- **96 arquivos originais conferidos**, sem alteração/perda detectada; origens e hashes dos documentos registrados na biblioteca preservados. Nenhuma exclusão foi feita.

## Resultado e disponibilidade

A primeira passagem v3 atende integralmente **1/27** tarefas novas; o final **0/27**, contra **2/27** da v2 de conversação. Cópia correta de Arvélio e resumo parcial são registrados separadamente. Matemática, explicações e contexto permanecem deficientes. Melhor previsão de texto não demonstrou boa conversação.

Checkpoint para comparação: `{deployment['checkpoint']}` (aproximadamente uma passagem supervisionada). Final preservado para recuperação/comparação: `{result['checkpoint']}`. Nenhum candidato foi aprovado ou promovido automaticamente. A v2 `{preservation['v2_active_before']}` permanece ativa por padrão, e sua referência anterior também continua disponível.

**INICIAR_TAC.bat → Conversar → TAC v3 — experimental, 29,4 milhões**. A troca de versão inicia conversa limpa; a prova nativa confirmou inferência real e restaurou a v2.

## Pendências

**Não há tarefa operacional pendente nesta rodada nem treino em andamento.** O objetivo de conversação coerente continua não atingido; isso é o resultado do experimento, não autorização para prolongar o orçamento. A única próxima hipótese está recomendada no relatório, sem execução automática. O comando de treino retorna o resultado concluído, sem iniciar outra execução.

Evidências: [RELATORIO.md](RELATORIO.md), [COMPARACAO_MODELOS.md](COMPARACAO_MODELOS.md), [FALHAS.md](FALHAS.md), `RESULTADOS.json`, `REVISAO_GERACOES.json`, `integridade_v3.json`, `interface_verificada.json`, `v2_preservacao_depois.json`, `validacao_final.json`.

Resultado autoritativo: `data/tac_v3/training_result.json`; `progress.json` registra fase `complete`. `live.json` conserva a última amostra periódica anterior ao encerramento. As sessões antigas 13557 e 54950 terminaram; não precisam ser retomadas.

Nenhum teste final reservado foi aberto/usado, nenhum modelo/API externo gerou respostas nesta execução, nenhum arquivo pessoal foi excluído e nenhuma rodada seguinte foi executada.

---

<details><summary>Registro histórico anterior, substituído pelo encerramento acima</summary>

{old}

</details>
'''
        state.write_text(final, encoding='utf-8')
    required = ['RELATORIO.md','COMPARACAO_MODELOS.md','FALHAS.md','RESULTADOS.json',
                'REVISAO_GERACOES.json','integridade_v3.json','interface_verificada.json',
                'v2_preservacao_depois.json','ESTADO_EXECUCAO.md','testes.xml',
                'avaliacao_dev.json','interface_v3.png']
    required += ['geracoes_'+label+'.json' for label in labels]
    required += ['linguagem_'+label+'.json' for label in labels]
    inventory = {}
    for name in required:
        p = OUT/name
        assert p.is_file() and p.stat().st_size > 0
        if p.suffix == '.json':
            read(p)
        inventory[name] = {'bytes':p.stat().st_size, 'sha256':sha(p)}
    summary = {
        'status':'OK_integridade_e_entrega_qualidade_nao_aprovada',
        'closed_at':datetime.now(timezone.utc).isoformat(),
        'training_completed':True, 'global_updates':result['global_steps'],
        'checkpoints_checked':len(integrity['checkpoints']),
        'v2_original_files_unchanged':preservation['original_files'],
        'active_checkpoint_preserved':core.active_conversation_id(),
        'experimental_checkpoint':deployment['checkpoint'],
        'quality_approved':False, 'coherent_conversation_objective_achieved':False,
        'main_tasks_integrally_satisfied':{label:review['models'][label]['main_appropriate'] for label in labels},
        'main_tasks_per_model':27, 'known_probes_reported_separately':True,
        'comparison_generations':sum(generations),
        'native_interface_generations':len(interface['neural_outputs'])+1,
        'same_frozen_inputs_all_models':True, 'development_hash_unchanged':True,
        'native_interface_verified':True, 'tests_passed':tests,
        'training_code_matches_recorded_resume_version':True,
        'reserved_tests_opened':False, 'external_model_inference_used':False,
        'automatic_promotion':False, 'next_experiment_executed':False,
        'pending_operational_tasks':[],
        'review_limitation':'Agent qualitative inspection, not independent human certification',
        'files':inventory,
    }
    atomic_json(OUT/'validacao_final.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
