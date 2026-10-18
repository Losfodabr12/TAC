"""Conferências finais pontuais do corpus congelado e da disponibilidade."""
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
from tac.neural.v2_data import load_jsonl
from tac.config import atomic_json
BASE=ROOT/'data/tac_v2_conversation_20261018';OUT=ROOT/'reports/conversacao_v2/rodada_20261018_01'


def main():
    sets={s:list(load_jsonl(BASE/('curated_'+s+'.jsonl'))) for s in ('train','validation')}
    treegroups={s:{r['group'] for r in rows if r['source'].startswith('OpenAssistant/')} for s,rows in sets.items()}
    overlap=sorted(treegroups['train']&treegroups['validation'])
    assert not overlap
    suite=ET.parse(OUT/'pytest_final.xml').getroot().find('testsuite')
    assert int(suite.attrib['failures'])==int(suite.attrib['errors'])==0 and int(suite.attrib['tests'])==54
    native=json.loads((OUT/'interface_verificada.json').read_text(encoding='utf-8'))
    assert native['version_selector_works'] and native['regression_warning_visible'] and native['session_cleared_on_switch']
    valid=json.loads((OUT/'validacao_final.json').read_text(encoding='utf-8'))
    valid.update(tests_passed=54,native_windows_interface_verified=True,reference_switch_verified=True,
                 oasst_train_validation_tree_overlap=len(overlap))
    assert hashlib.sha256((ROOT/'data/checkpoints/64f45f52fee54053a8f0c6b16d9df172.pt').read_bytes()).hexdigest()==valid['reference_file_sha256']
    atomic_json(OUT/'validacao_final.json',valid)
    print(json.dumps({'tests':54,'tree_overlap':overlap,'interface':True,'reference_preserved':True,'semantic_goal_achieved':False},ensure_ascii=False))


if __name__=='__main__':main()
