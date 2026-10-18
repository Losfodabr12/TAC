"""Auditoria e consulta real opcional, isolada da biblioteca do usuário."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding='utf-8')

from tac.config import atomic_json
from tac.core import Core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Autoriza consulta pública real à Wikipédia')
    args = parser.parse_args()
    core = Core()
    baseline = json.loads((ROOT/'reports/baseline_070.json').read_text(encoding='utf-8'))
    current = {r['id']:r for r in core.memory.sources()}
    for old in baseline['sources']:
        assert current[old['id']]['content_hash'] == old['content_hash']
        assert current[old['id']]['origin'] == old['origin']
    for old in baseline['models']:
        active = core.db.query('SELECT value FROM meta WHERE key=?', (old['key'],))[0]['value']
        assert active == old['id']
        assert hashlib.sha256((core.config.data/'checkpoints'/old['filename']).read_bytes()).hexdigest() == old['sha256']
    result = {'status':'ok', 'original_sources_preserved':len(baseline['sources']), 'all_active_weight_files_unchanged':True,
              'schema_unchanged':True, 'live':False}
    if args.live:
        isolated = Core(ROOT/'reports/live_web_070_final')
        isolated.gate.set_enabled(True)
        isolated.config.update(research_auto=True, research_pages=1)
        question = 'O que é fotossíntese?'
        answer = isolated.perform('Consulta real de aceitação', lambda:isolated.chat(question,
            progress=lambda n,m:print(m,flush=True)), learning=True)
        assert answer['mode'] == 'consulta web documental' and answer['sources']
        assert answer['research']['status'] == 'fonte_unica'
        assert answer['evidence'][0].startswith('Fotossíntese é ')
        restored = Core(isolated.config.root)
        restored.gate.set_enabled(False)
        offline = restored.chat(question)
        assert offline['evidence'] == answer['evidence'] and offline['web_verification']
        from tac.neural.data import Corpus
        try:
            Corpus.from_memory(restored.db,restored.config)
        except ValueError:
            excluded = True
        else:
            excluded = False
        assert excluded
        result.update(live=True, live_answer=answer, restart_offline_persistence=True,
                      researched_text_excluded_from_neural_training=True)
    atomic_json(ROOT/'reports/verification_research_070.json',result)
    print(json.dumps(result,ensure_ascii=True,indent=2))


if __name__ == '__main__':
    main()
