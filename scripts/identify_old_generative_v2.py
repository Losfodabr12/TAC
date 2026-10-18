"""Lista exata de exclusão autorizada; não exclui nem copia pesos."""
import gc
import hashlib
import json
import re
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
import torch
OUT=ROOT/'reports/reconstrucao_zero/execucao_v2'


def main():
    assert (ROOT/'data/tac_v2/encoded_ready.json').exists()
    progress=json.loads((ROOT/'data/tac_v2/progress.json').read_text(encoding='utf-8'))
    assert progress['generation']=='TAC-v2' and progress['global_step']==0
    OUT.mkdir(exist_ok=True)
    prior=json.loads((ROOT/'reports/reconstrucao_zero/preparacao_20261017_01/preparacao.json').read_text(encoding='utf-8'))
    inventory=json.loads((ROOT/'reports/auditoria_pesos_e_dados/rodada_20261017_01/preservacao_depois.json').read_text(encoding='utf-8'))['sha256']
    files=[];skipped=[];catalog={x['filename']:x for x in prior['checkpoint_catalog']}
    for n,item in enumerate(prior['candidate_pt_files_not_yet_deleted_or_classified']):
        relative=item['file'];p=(ROOT/relative).resolve()
        if not p.is_relative_to(ROOT) or p.is_symlink() or not p.is_file():raise ValueError('Alvo inválido: '+relative)
        if relative not in inventory:skipped.append({'file':relative,'reason':'not_in_prior_inventory'});continue
        cp=torch.load(p,map_location='cpu',weights_only=True)
        generative=cp.get('format') in ('TAC-v1','TAC-chat-v2') and isinstance(cp.get('model_config'),dict) and isinstance(cp.get('model'),dict) and 'tokens.weight' in cp['model']
        if generative:
            if cp.get('generation')=='TAC-v2':raise ValueError('Checkpoint novo nunca pode ser alvo')
            files.append({'file':relative,'bytes':p.stat().st_size,'sha256':inventory[relative]['sha256'],
                          'kind':'old_generative_checkpoint','format':cp['format']})
        else:skipped.append({'file':relative,'reason':'classifier_or_non_generative_format','format':cp.get('format')})
        del cp
        if n%200==0:gc.collect();print('Identificados',n,'arquivos de checkpoint',flush=True)
    # Snapshots gerenciados antigos: nomes hashes, fora de originais, código, acervo e arquivos mistos.
    for relative,meta in inventory.items():
        p=ROOT/relative
        if p.suffix!='.json' or 'backups' in p.parts or re.search(r'reservad|reserved|sealed|selado',p.name,re.I):continue
        managed=bool(re.search(r'(?:^|/)data/datasets/(?:conversation_)?[0-9a-f]{64}\.json$',relative))
        experiment=(relative.startswith(('reports/memoria_temporal_0130/','reports/aprendizado_contrastivo_temporal/','reports/compreensao_generalizacao_perguntas/')) and p.name.startswith(('dados','dataset')))
        if not (managed or experiment) or not p.is_file():continue
        prefix=p.open(encoding='utf-8').read(65536)
        m=re.search(r'"train"\s*:\s*\[',prefix)
        if not m:continue
        rest=prefix[m.end():].lstrip()
        # Apenas prova do tipo de primeiro item, sem abrir conjuntos de avaliação.
        if rest.startswith('"'):kind='old_language_training_snapshot'
        elif rest.startswith('{') and '"messages"' in rest[:8000] and '"answer"' in rest[:8000]:kind='old_instruction_training_snapshot'
        else:continue
        files.append({'file':relative,'bytes':p.stat().st_size,'sha256':meta['sha256'],'kind':kind})
    files=sorted({r['file']:r for r in files}.values(),key=lambda r:r['file'])
    assert all(not x['file'].startswith(('tac/','scripts/','data/books/','data/inbox/','data/dialogue_imports/','data/tac_v2/')) for x in files)
    manifest={'authorization':'Solicitação atual do usuário: excluir somente artefatos antigos identificados da rede generativa, sem backups, após corpus e treino novo prontos.',
        'root':str(ROOT),'new_checkpoint_protected':progress['checkpoint'],'files':files,'skipped_checkpoints':skipped,
        'bytes':sum(x['bytes'] for x in files),'backups_created':0,
        'preserved':'código, interface, biblioteca/SQLite, documentos/books/inbox/imports, classificador, relatórios sem datasets, arquivos de backup mistos e testes reservados'}
    with (OUT/'deletion_manifest.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
    print('LISTA EXATA',len(files),'arquivos;',round(manifest['bytes']/2**30,3),'GiB; nenhum ainda excluído.',flush=True)


if __name__=='__main__':main()
