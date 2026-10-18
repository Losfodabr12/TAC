"""Logs técnicos sem mensagens livres, caminhos pessoais ou linhas de código."""
from __future__ import annotations

import re
import ast
from collections import Counter

from .diagnostic_privacy import error_category

EXCEPTIONS=set('ValueError RuntimeError TypeError KeyError IndexError AttributeError OSError IOError PermissionError FileNotFoundError UnicodeError UnicodeDecodeError UnicodeEncodeError MemoryError OutOfMemoryError FloatingPointError AssertionError Cancelled LearningDisabled TimeoutError ConnectionError OperationalError IntegrityError JSONDecodeError UnpicklingError'.split())


def collect_logs(root,data,privacy):
    lines=['TAC — LOGS ANONIMIZADOS',
           'Mensagens livres, linhas de código, arquivos pessoais, URLs e caminhos absolutos foram omitidos.',
           'Preservados: data, nível, categoria literal do erro, classe de exceção e frames do código TAC.',
           'Nenhum texto de documento, conversa ou credencial é copiado para este arquivo.','']
    counts=Counter(); files=[]
    folder=(data/'logs').resolve()
    candidates=[folder/('tac.log'+suffix) for suffix in ('.4','.3','.2','.1','')]+[root/'startup_error.log']
    known_modules={p.relative_to(root).as_posix() for p in (root/'tac').rglob('*.py')}
    known_functions={}
    line_counts={}
    for module in known_modules:
        try:
            source=(root/module).read_text(encoding='utf-8')
            line_counts[module]=len(source.splitlines())
            known_functions[module]={n.name for n in ast.walk(ast.parse(source))
                                     if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}|{'<module>'}
        except (OSError,ValueError,SyntaxError):known_functions[module]=set()
    for path in candidates:
        if not path.is_file():continue
        resolved=path.resolve()
        if resolved!=path.absolute():
            lines.append('Arquivo redirecionado omitido.'); continue
        with path.open('rb') as stream:raw=stream.read(2_000_001)
        truncated=len(raw)>2_000_000
        text=raw[:2_000_000].decode('utf-8',errors='replace')
        # Reúne cada evento antes de classificar; seu texto original nunca sai da memória.
        records=[]; current=[]
        for line in text.splitlines():
            if re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ (?:INFO|WARNING|ERROR|CRITICAL|DEBUG) ',line):
                if current:records.append(current)
                current=[line]
            else:current.append(line)
        if current:records.append(current)
        kept=0
        for record in records:
            whole='\n'.join(record)
            header=re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) (WARNING|ERROR|CRITICAL) ',record[0] if record else '')
            if not header and 'Traceback (most recent call last)' not in whole:continue
            kept+=1
            category=error_category(whole); counts[category]+=1
            lines.append(f"{header[1]+' '+header[2] if header else '[data/nível indisponíveis]'} evento={privacy.alias(whole,'log')} categoria_literal={category}")
            for line in record:
                frame=re.match(r'\s*File "([^"]+)", line (\d+), in ([A-Za-z_][A-Za-z_0-9]*|<module>)',line)
                if frame:
                    # Usa somente caminho relativo de um módulo existente do próprio pacote.
                    normalized=frame[1].replace('\\','/')
                    module=next((m for m in known_modules if normalized.endswith('/'+m)),None)
                    number=frame[2] if module and 1<=int(frame[2])<=line_counts.get(module,0) else '[linha omitida]'
                    lines.append(f'  {module or "[módulo externo]"}:{number}' + (f' função={frame[3]}' if module and frame[3] in known_functions[module] else ''))
                exception=re.match(r'^(?:[a-zA-Z_][a-zA-Z_0-9]*\.)*([A-Za-z_][A-Za-z_0-9]*):',line)
                if exception and exception[1] in EXCEPTIONS:lines.append('  exceção='+exception[1]+'; mensagem livre omitida')
            lines.append('')
        label='startup_error.log' if path.name=='startup_error.log' else path.name
        files.append({'arquivo':label,'bytes_lidos':len(raw[:2_000_000]),'truncado':truncated,'eventos_exportados':kept})
    if not files:lines.append('Logs indisponíveis: nenhum arquivo de log encontrado.')
    elif not counts:lines.append('Nenhuma exceção ou aviso encontrado nos arquivos lidos.')
    return '\n'.join(lines),{'arquivos':files,'categorias_literais':dict(counts),
                           'limite_por_arquivo_bytes':2_000_000,'mensagens_livres_exportadas':False}
