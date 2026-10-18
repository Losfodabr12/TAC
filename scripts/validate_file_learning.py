"""Confere artefatos de aquisição e preservação; não treina nem abre teste final."""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tac.config import atomic_json
from scripts.learn_conversation_file import preservation
from scripts.run_controlled_experiments import sha

OUT = ROOT / 'reports/aprendizado_conversas_0130'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    config = read(OUT / 'configuracao.json')
    result = read(OUT / 'resultado.json')
    before = read(OUT / 'preservacao_antes.json')
    source_folder = Path('C:/Users/Joaof/Desktop/TAC_Experimento_Aprendizado')
    now = preservation(source_folder)
    assert before == now, 'Alteração detectada nos originais.'
    atomic_json(OUT / 'preservacao_final.json', now)
    comparisons = read(OUT / 'comparacao_respostas.json')
    assert len(comparisons) == 170 and all(r['exact_candidate'] for r in comparisons)
    assert sum(r['exact_active'] for r in comparisons) == 2
    assert config['parameter_count'] == 10004544
    assert not any(config[k] for k in ('replay', 'kl', 'rag', 'documents', 'promotion', 'test_reserved_read'))
    assert config['internal_examples'] == 0 and config['examples'] == 85
    assert result['steps'] == 120 and result['weights_changed'] and result['checkpoint_reload_verified']
    assert sha(OUT / 'candidato_experimental.pt') == result['checkpoint_sha256']
    exposure = result['exposure']
    steps = read(OUT / 'passos.json')
    assert len(steps) == result['steps']
    assert exposure['unique_examples'] == 85 and exposure['presentations'] == 1920
    assert sum(exposure['supervised_tokens_by_skill'].values()) == sum(r['tokens'] for r in steps)
    preview = read(OUT / 'conferencia_importador_real.json')
    assert preview['accepted_examples'] == 85 and not preview['rejected'] and not preview['duplicates']
    tests = ET.parse(OUT / 'testes.xml').getroot().find('testsuite')
    assert tests is not None and int(tests.attrib['tests']) == 39
    assert int(tests.attrib['failures']) == int(tests.attrib['errors']) == 0
    for name in ('teste_bat.txt', 'teste_bat_ativo.txt'):
        assert 'Você: TAC:' in (OUT / name).read_text(encoding='utf-8')
    for name in ('curva_aprendizado.png', 'respostas_exatas.png'):
        assert (OUT / name).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    preserved_pilots = []
    for path in (ROOT / 'reports/experimentos_0130').glob('*_s*/result.json'):
        pilot = read(path)
        if pilot['checkpoint_kept']:
            assert sha(path.parent / 'candidate.pt') == pilot['checkpoint_sha256']
            preserved_pilots.append(path.parent.name)
    for path in OUT.glob('*.json'):
        read(path)  # Todos os JSON de relatório são válidos.
    final = {'status': 'ok', 'training_steps': 120, 'unique_examples': 85,
             'exact_active_gold': 1, 'exact_candidate_gold': 85,
             'exact_active_own': 1, 'exact_candidate_own': 85,
             'passed_software_tests': 39, 'original_data_preserved': before == now,
             'old_pilot_checkpoints_preserved': preserved_pilots,
             'candidate_sha256': result['checkpoint_sha256'], 'model_active_changed': False,
             'reserved_test_used': False, 'reserved_access': 'somente hashes de preservação',
             'generalization_tested': False, 'old_skills_retention_tested': False,
             'bat_experimental_and_active_smoke_passed': True}
    atomic_json(OUT / 'validacao_final.json', final)
    files = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(OUT.iterdir())
             if p.is_file() and p.name != 'manifesto.json'}
    atomic_json(OUT / 'manifesto.json', {'files': files, 'scope': 'artefatos diretos; não inclui diretório temporário do pytest'})
    print(json.dumps(final, ensure_ascii=True))


if __name__ == '__main__':
    main()
