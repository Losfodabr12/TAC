"""Check the staged export without reading ignored data or reserved tests."""
import ast
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERNS = [
    re.compile(rb'gh[pousr]_[A-Za-z0-9_]{30,}'),
    re.compile(rb'github_pat_[A-Za-z0-9_]{40,}'),
    re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    re.compile(rb'AKIA[A-Z0-9]{16}'),
    re.compile(rb'sk-proj-[A-Za-z0-9_-]{40,}'),
]


def main():
    names = subprocess.check_output(['git', 'ls-files', '--cached', '-z'], cwd=ROOT).decode('utf-8').split('\0')
    names = [name for name in names if name]
    assert names, 'No staged files'
    errors = []
    source_count = total_bytes = 0
    for name in names:
        p = Path(name)
        lower = name.lower()
        forbidden = (p.parts[0].lower() in {'data', 'reports', 'materials', '.venv', '.git'}
                     or '__pycache__' in p.parts or p.name == 'config.json'
                     or p.suffix.lower() in {'.pt', '.pth', '.safetensors', '.sqlite3', '.sqlite', '.db', '.pyc', '.zip'}
                     or '_holdout.json' in lower
                     or any(part in lower for part in ('teste_reservado', 'teste_selado', 'sealed_transfer')))
        if forbidden:
            errors.append({'file': name, 'reason': 'Excluded local/private/reserved artifact'})
            continue  # Never read excluded files, even if mistakenly staged.
        payload = (ROOT/p).read_bytes()
        total_bytes += len(payload)
        if len(payload) > 10*2**20:
            errors.append({'file': name, 'reason': 'Unexpected file over 10 MiB'})
        if any(pattern.search(payload) for pattern in SECRET_PATTERNS):
            errors.append({'file': name, 'reason': 'Potential credential; value suppressed'})
        if p.suffix.lower() == '.py':
            ast.parse(payload.decode('utf-8-sig'), filename=name)
            source_count += 1
    required = ['tac/neural/model.py', 'tac/neural/conversation.py', 'tac/neural/training.py',
                'tac/neural/bpe.py', 'tac/neural/v3_training.py', 'tac/conversation_lessons.py',
                'config.example.json', 'PUBLICACAO_GITHUB.md']
    assert all(name in names for name in required), 'Missing requested code'
    cfg = json.loads((ROOT/'config.example.json').read_text(encoding='utf-8'))
    assert cfg['learning_enabled'] is cfg['auto_train'] is cfg['auto_on_start'] is False
    assert cfg['data_dir'] == 'data'
    result = {'ok': not errors, 'tracked_files': len(names), 'python_files_syntax_checked': source_count,
              'total_bytes': total_bytes, 'excluded_data_reports_and_weights': True,
              'local_config_excluded': True, 'errors': errors,
              'limitation': 'Credential pattern scan is limited; ignored local files are not read.'}
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
