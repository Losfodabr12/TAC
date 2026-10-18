"""Exportação local e somente leitura de diagnóstico. Não executa avaliações/treino."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import sqlite3
import subprocess
import tempfile
import zipfile
from collections import Counter,defaultdict
from contextlib import closing
from datetime import datetime,timezone
from pathlib import Path

from . import __version__
from .diagnostics import resources
from .diagnostic_privacy import Privacy,safe_reason,error_category,FIELDS
from .diagnostic_specs import implementation
from .diagnostic_datasets import analyze_dataset
from .diagnostic_logs import collect_logs

NAMES=('resumo.md','arquitetura.json','treinamento.json','checkpoints.json','avaliacoes.json',
       'exemplos_de_respostas.json','datasets_estatisticas.json','logs.txt')
MAX_JSON=64*2**20
DATASET_NAME=re.compile(r'(?:conversation_|dialogue_)?([0-9a-f]{64})\.json')


def read_json(path):
    with path.open('rb') as stream:raw=stream.read(MAX_JSON+1)
    if len(raw)>MAX_JSON:raise ValueError('limite_de_tamanho')
    value=json.loads(raw)
    if not isinstance(value,dict):raise ValueError('objeto_esperado')
    return value


def utc():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def unavailable(exc):
    # Nunca inclui str(exc), que pode conter nomes de arquivos ou dados privados.
    return {'estado':'indisponivel','classe_erro':type(exc).__name__ if type(exc).__name__ in
        ('ValueError','TypeError','KeyError','FileNotFoundError','PermissionError','RuntimeError','OSError',
         'JSONDecodeError','UnpicklingError','MemoryError','OperationalError','UnicodeDecodeError') else 'erro_na_leitura'}


def hardware():
    import psutil
    import torch
    result={'coletado_em':utc(),'sistema':platform.system(),'versao_sistema':platform.release(),
            'python':platform.python_version(),'torch':torch.__version__,'runtime_cuda':torch.version.cuda,
            'cpu_fisicos':psutil.cpu_count(logical=False),'cpu_logicos':psutil.cpu_count(),
            'ram_total_mb':psutil.virtual_memory().total/2**20,'atual':resources(),
            'cuda_disponivel':torch.cuda.is_available(),'gpu':None,
            'historico_uso_recursos':'Indisponível quando a execução não registrou o uso. Limites configurados não são picos medidos.'}
    result['vram_tensores_neste_processo_mb']=torch.cuda.memory_allocated()/2**20 if torch.cuda.is_initialized() else 0.
    result['atual']['cpu_percent']=psutil.cpu_percent(interval=.1)
    try:
        run=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],
            capture_output=True,text=True,timeout=3,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        if run.returncode==0:
            cards=[]
            for row in run.stdout.strip().splitlines():
                name,use,used,total=(part.strip() for part in row.split(','))
                if not re.fullmatch(r'NVIDIA [A-Za-z0-9 .()+/-]{1,90}',name):name='GPU NVIDIA (nome omitido)'
                cards.append({'nome':name,'uso_percent':float(use),'vram_sistema_usada_mb':float(used),'vram_total_mb':float(total)})
            result['gpu']={'estado':'medido','placas':cards}
        else:result['gpu']={'estado':'indisponivel'}
    except (OSError,ValueError,subprocess.SubprocessError) as exc:result['gpu']=unavailable(exc)
    return result


def checkpoint_metadata(path,privacy):
    import torch
    from .neural.model import ModelConfig,count_parameters
    if path.stat().st_size>300*2**20:raise ValueError('checkpoint_grande')
    # mmap carrega metadados sem copiar os tensores para RAM/GPU. Não instancia rede.
    cp=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if cp.get('format') not in ('TAC-v1','TAC-chat-v2','TAC-dialogue-v1','TAC-RL-v1'):raise ValueError('formato')
    model=cp.get('model',{})
    seen=set(); unique=0; byte_count=0; shapes={}
    for name,tensor in model.items():
        if not isinstance(tensor,torch.Tensor):continue
        identity=(tensor.untyped_storage()._cdata,tensor.storage_offset(),tuple(tensor.shape),tuple(tensor.stride()))
        if identity not in seen:
            unique+=tensor.numel(); byte_count+=tensor.numel()*tensor.element_size(); seen.add(identity)
        safe_name=name if re.fullmatch(r'(?:(?:tokens|position|norm|output)|blocks\.\d+\.(?:norm[12]|qkv|proj|ff\.[02])|net\.[02])\.(?:weight|bias)',name) else privacy.alias(name,'tensor')
        shapes[safe_name]={'shape':list(tensor.shape),'dtype':str(tensor.dtype)}
    cfg=cp.get('model_config')
    serialized_unique=unique
    serialized_bytes=byte_count
    if cfg and cfg.get('tie_embeddings') and 'tokens.weight' in model and 'output.weight' in model:
        a,b=model['tokens.weight'],model['output.weight']
        # A gravação CUDA→CPU pode duplicar o mesmo parâmetro em dois storages.
        # A rede reconstruída compartilha o parâmetro por configuração, não pelo ZIP.
        if a.shape==b.shape and a.untyped_storage()._cdata!=b.untyped_storage()._cdata:
            unique-=b.numel(); byte_count-=b.numel()*b.element_size()
    result={'estado':'lido','formato':cp['format'],'parametros_unicos_observados':unique,
            'bytes_parametros':byte_count,'parametros_unicos_armazenados':serialized_unique,
            'bytes_tensores_armazenados':serialized_bytes,
            'compartilhamento_entrada_saida_por_configuracao':bool(cfg and cfg.get('tie_embeddings')),'tensores':shapes,
            'arquitetura':privacy.sanitize(cfg) if cfg else None,
            'dataset_id':privacy.identifier(cp.get('dataset_hash')),
            'step':cp.get('step') if isinstance(cp.get('step'),int) else None,
            'target_steps':cp.get('target_steps') if isinstance(cp.get('target_steps'),int) else None,
            'config_treino_salva':privacy.sanitize(cp.get('training_config')),
            'otimizador_grupos_salvos':[privacy.sanitize({k:v for k,v in group.items() if k!='params'})
                for group in cp.get('optimizer',{}).get('param_groups',[])],
            'estado_amp_registrado':bool(cp.get('scaler'))}
    if cfg:
        result['tipo_rede']='Transformer decoder causal de diálogo' if cp.get('task') in ('TAC-conversation-sft-v1','TAC-conversation-sft-v2') else 'Transformer decoder causal de leitura'
        formula=count_parameters(ModelConfig(**cfg))
        result['parametros_formula']=formula; result['contagem_confere']=formula==unique
        result['feedforward_dim_efetiva']=cfg.get('ff_width') or 4*cfg['width']
        state=cp.get('tokenizer_state')
        result['tokenizador']={'tipo':'BPE local' if state else 'bytes UTF-8',
            'versao':privacy.sanitize(state.get('version'),'version') if state else 'utf8-bytes-v1',
            'vocab_size':259+len(state['merges']) if state else 259,
            'merges_textos_exportados':False}
    elif cp['format']=='TAC-dialogue-v1':
        result['tipo_rede']='MLP classificador de intenções; respostas preparadas'
        result['arquitetura']={'entrada':cp.get('dimension'),'oculta':128,'saida':len(cp.get('labels',[])),
                              'tipo':'MLP classificador de intenções','atencao':False}
        result['limiar_salvo']=privacy.sanitize(cp.get('threshold'))
    elif cp['format']=='TAC-RL-v1':
        result['tipo_rede']='MLP de política; experimento separado de reforço'
        result['arquitetura']={'entrada':8,'oculta':48,'saida':7,'tipo':'MLP política de somas; Tanh','atencao':False}
    del cp
    return result


def separate_examples(value,record_id,privacy,examples,path='',private=False):
    if isinstance(value,dict):
        result={}
        for key,item in value.items():
            if key in ('password','senha','token','api_key','secret','authorization','credentials','data_dir','origin','filename','path'):
                privacy.redactions['campos_excluidos']+=1
                continue
            child=path+'/'+privacy.key(key)
            dynamic=path.rsplit('/',1)[-1] in ('by_skill','dataset_examples','mixture_counts','splits','reserved')
            if key not in FIELDS and key not in privacy.skills and not dynamic and key!='exploratory_samples':
                result[privacy.key(key)]=privacy.alias(item)
                privacy.redactions['campos_desconhecidos_omitidos']+=1
                continue
            if key in ('outputs','sessions','test_results','exploratory_samples') and isinstance(item,list):
                examples.append({'registro_id':record_id,'caminho':child,'quantidade':len(item),
                    'conteudo_privado_omitido':private,'exemplos':privacy.sanitize(item,key,private)})
                result[key]={'arquivo':'exemplos_de_respostas.json','registro_id':record_id,'caminho':child,'quantidade':len(item)}
            elif key in ('detail','error'):
                result[key]=safe_reason(item,privacy)
            else:result[privacy.key(key)]=separate_examples(item,record_id,privacy,examples,child,private or key=='imported_evaluation')
        return result
    return privacy.sanitize(value,path.rsplit('/',1)[-1],private)


def metric_states(value,path='',privacy=None):
    """Não preenche uma avaliação ausente com nota zero."""
    found=[]
    if not isinstance(value,dict):return found
    for key,item in value.items():
        known=key in FIELDS or (privacy and key in privacy.skills) or path.rsplit('/',1)[-1]=='by_skill'
        if isinstance(item,dict) and known:
            found+=metric_states(item,path+'/'+(privacy.key(key) if privacy else str(key)),privacy)
    for metric,denominators in (('exact_accuracy',('examples',)),('rule_accuracy',('turns',)),
        ('immediate_accuracy',('examples',)),('recall_accuracy',('examples',)),('accuracy',('total','examples'))):
        if metric in value:
            number=value[metric] if isinstance(value[metric],(int,float)) else None
            if isinstance(number,float) and not math.isfinite(number):number=None
            n=next((value[k] for k in denominators if isinstance(value.get(k),(int,float))),None)
            state='indisponivel' if number is None else 'zero_sem_amostras' if n==0 else 'zero_medido' if number==0 and n and n>0 else 'zero_origem_indeterminada' if number==0 else 'registrado'
            found.append({'caminho':path+'/'+metric,'valor':number,'amostras':n,'estado':state})
    if 'loss' in value and ('token_accuracy' in value or 'targets' in value) and 'exact_accuracy' not in value:
        found.append({'caminho':path+'/exact_accuracy','valor':None,'amostras':value.get('examples'),
                      'estado':'nao_registrado','explicacao':'Só há avaliação de tokens/perda neste registro; não há nota de respostas completas.'})
    return found


def collect_external(root,privacy,examples):
    records=[]
    folder=root/'reports'
    if not folder.is_dir():return records
    patterns=(r'evaluation_conversation_[0-9a-f]{32}_(?:all|test|validation|test_context|validation_context)\.json',
              r'(?:natural_(?:before|after|intermediate)|fluency_before)_[0-9a-f]{32}_(?:validation_natural|test_natural|validation_fluency|test_fluency)\.json',
              r'conversation_[0-9a-f]{32}\.json')
    for path in sorted(folder.glob('*.json')):
        if not any(re.fullmatch(pattern,path.name) for pattern in patterns):continue
        if path.resolve().parent!=folder.resolve():continue
        alias=path.name # Padrões acima não permitem nomes livres.
        try:
            value=read_json(path)
            records.append({'registro_id':alias,'fonte':'reports/'+alias,
                'dados':separate_examples(value,alias,privacy,examples),
                'estados_metricas':metric_states(value,privacy=privacy)})
        except Exception as exc:records.append({'registro_id':alias,**unavailable(exc)})
    return records


def software_tests(root):
    import ast
    import xml.etree.ElementTree as ET
    # Nomes só são publicados se existirem no código de testes do pacote.
    known=set()
    for path in (root/'tests').glob('test_*.py'):
        try:known.update(n.name for n in ast.walk(ast.parse(path.read_text(encoding='utf-8'))) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)))
        except (OSError,SyntaxError,UnicodeError):pass
    paths=sorted((root/'reports').glob('*.xml'))+sorted((root/'reports').glob('*/testes.xml'))
    reports=[]
    for index,path in enumerate(paths):
        if path.is_symlink() or path.stat().st_size>10*2**20:continue
        try:
            tree=ET.parse(path); cases=[]
            for item in tree.iter('testcase'):
                name=item.get('name','').split('[',1)[0]
                status='erro' if item.find('error') is not None else 'falhou' if item.find('failure') is not None else 'ignorado' if item.find('skipped') is not None else 'passou'
                cases.append({'teste':name if name in known else '[nome não verificado]',
                    'estado':status,'segundos':float(item.get('time')) if item.get('time') else None,
                    'mensagens_livres_omitidas':True})
            suites=[{k:float(n.attrib[k]) for k in ('tests','failures','errors','skipped','time') if k in n.attrib} for n in tree.iter('testsuite')]
            reports.append({'relatorio':'testes_'+str(index+1),'suites_registradas':suites,'casos':cases})
        except (OSError,ValueError,ET.ParseError) as exc:reports.append({'relatorio':'testes_'+str(index+1),**unavailable(exc)})
    return {'estado':'registrado' if reports else 'indisponivel','relatorios':reports,
            'nota':'Resultados de arquivos históricos. Nenhum teste foi executado durante a exportação; não certificam o código atual.'}


def observations(run,detail,privacy):
    out=[]
    before=detail.get('baseline'); after=detail.get('validation')
    if isinstance(before,dict) and isinstance(after,dict):
        if isinstance(before.get('loss'),(int,float)) and isinstance(after.get('loss'),(int,float)) and after['loss']<before['loss']:
            if isinstance(before.get('exact_accuracy'),(int,float)) and isinstance(after.get('exact_accuracy'),(int,float)) and after['exact_accuracy']<=before['exact_accuracy']:
                out.append({'tipo':'loss_caiu_sem_mais_respostas_exatas','fato':{'loss_antes':before['loss'],'loss_depois':after['loss'],
                    'exatas_antes':before['exact_accuracy'],'exatas_depois':after['exact_accuracy']},
                    'interpretacao':'Evidência de métricas diferentes; não demonstra a causa.'})
        if 'exact_accuracy' not in before:
            out.append({'tipo':'geracao_baseline_nao_registrada','interpretacao':'Uma tela que usa get(exact_accuracy,0) pode mostrar zero; esse zero não comprova teste executado.'})
    checks=detail.get('promotion_checks')
    if isinstance(checks,dict):out.append({'tipo':'criterios_registrados_nao_atingidos','criterios':[privacy.key(k) for k,v in checks.items() if v is False]})
    if run['status']=='erro':out.append({'tipo':'execucao_com_erro','categoria_literal':error_category(run['detail']),
        'causa_raiz':'Não determinada automaticamente; consulte os logs técnicos anonimizados.'})
    return out


def collect(core,progress=lambda *_:None):
    privacy=Privacy()
    progress(0,'Preparando diagnóstico: lendo os exercícios públicos para anonimizar exemplos…')
    privacy.load_public_exercises()
    config=dict(core.config.values)
    # Backup SOMENTE em memória: visão transacional consistente, sem SQLite no ZIP.
    uri=core.db.path.resolve().as_uri()+'?mode=ro'
    with closing(sqlite3.connect(uri,uri=True)) as original,closing(sqlite3.connect(':memory:')) as snapshot:
        original.backup(snapshot)
        snapshot.row_factory=sqlite3.Row
        def query(sql,args=()):return [dict(r) for r in snapshot.execute(sql,args)]
        runs=query('SELECT * FROM runs ORDER BY started,id')
        checkpoints=query('SELECT * FROM checkpoints ORDER BY created,id')
        metrics=query('SELECT * FROM metrics ORDER BY run_id,step,id')
        active={r['key']:r['value'] for r in query("SELECT key,value FROM meta WHERE key IN ('active_checkpoint','active_conversation','active_dialogue')")}
        counts={name:snapshot.execute('SELECT count(*) FROM '+name).fetchone()[0]
                for name in ('sources','chunks','interactions','corrections','dialogue_imports','dialogue_examples','dialogue_groups')}
        counts['fontes_por_tipo']=query('SELECT kind,count(*) quantidade FROM sources GROUP BY kind')
        imports=query('SELECT g.split,count(*) examples,count(DISTINCT e.group_id) groups FROM dialogue_examples e JOIN dialogue_groups g ON g.id=e.group_id GROUP BY g.split')
        imported_conflicts=query('SELECT count(*) n FROM (SELECT e.prompt_hash FROM dialogue_examples e JOIN dialogue_groups g ON g.id=e.group_id GROUP BY e.prompt_hash HAVING count(DISTINCT g.split)>1)')[0]['n']
    examples=[]; evaluation_runs=[]; training_runs=[]; notes=[]
    for run in runs:
        rid=privacy.identifier(run['id'])
        try:params=json.loads(run['config'])
        except (ValueError,TypeError):params=None
        try:detail=json.loads(run['detail'])
        except (ValueError,TypeError):detail={}
        if not isinstance(detail,dict):detail={}
        record={'run_id':rid,'tipo':privacy.sanitize(run['kind'],'kind'),'estado':privacy.sanitize(run['status'],'status'),
            'iniciado':privacy.sanitize(run['started'],'started'),'finalizado':privacy.sanitize(run['finished'],'finished'),
            'dataset_id':privacy.identifier(run['dataset_hash']),'config_registrada':privacy.sanitize(params),
            'baseline_coluna_runs':privacy.sanitize(run['baseline']),'final_coluna_runs':privacy.sanitize(run['final']),
            'semantica_colunas':'Accuracy no classificador/reforço; loss nos Transformers. Conferir tipo da execução.',
            'motivo_registrado':safe_reason(detail.get('detail',run['detail']),privacy)}
        if params and isinstance(params.get('batch_size'),int):
            record['exemplos_por_passo_configurado']=params['batch_size']*params.get('accumulation',1)
        training_runs.append(record)
        evaluation_runs.append({'run_id':rid,'estado_execucao':record['estado'],
            'resultado_registrado':separate_examples(detail,rid,privacy,examples),
            'estados_metricas':metric_states(detail,privacy=privacy),
            'criterios_registrados':privacy.sanitize(detail.get('promotion_checks')),
            'motivo_registrado':record['motivo_registrado']})
        notes.extend({'run_id':rid,**note} for note in observations(run,detail,privacy))
    metrics_clean=[privacy.sanitize(row) for row in metrics]
    cp_rows=[]; contexts=defaultdict(set); architectures=[]
    folder=(core.config.data/'checkpoints').resolve()
    # O experimento de reforço guarda pesos fora da tabela checkpoints.
    known={row['filename'] for row in checkpoints}
    for path in folder.glob('rl_*.pt'):
        if re.fullmatch(r'rl_[0-9a-f]{32}\.pt',path.name) and path.name not in known:
            rid=path.stem[3:]
            checkpoints.append({'id':'rl_'+rid,'run_id':rid,'filename':path.name,'step':None,'val_loss':None,
                'status':'arquivo_fora_da_tabela','created':None})
    for index,row in enumerate(checkpoints):
        if index%20==0:progress(index,f'Lendo metadados dos modelos: {index}/{len(checkpoints)}. Os pesos não serão exportados…')
        item={k:privacy.sanitize(v,k) for k,v in row.items() if k!='filename'}
        item['ativo_em']=[key for key,value in active.items() if value==row['id']]
        path=(folder/row['filename']).resolve()
        if path.parent!=folder or path.suffix!='.pt':item['metadados']={'estado':'caminho_recusado'}
        elif not path.is_file():item['metadados']={'estado':'arquivo_ausente'}
        else:
            item['bytes_arquivo']=path.stat().st_size
            try:item['metadados']=checkpoint_metadata(path,privacy)
            except Exception as exc:item['metadados']=unavailable(exc)
            meta=item['metadados']; dataset_id=meta.get('dataset_id'); cfg=meta.get('arquitetura') or {}
            if dataset_id and isinstance(cfg.get('context'),int):contexts[dataset_id].add(cfg['context'])
            if item['ativo_em']:architectures.append({'checkpoint_id':item['id'],'ativo_em':item['ativo_em'],**meta})
        cp_rows.append(item)
    for key in ('active_checkpoint','active_conversation','active_dialogue'):
        if not any(key in a['ativo_em'] for a in architectures):
            architectures.append({'ativo_em':[key],'checkpoint_id':privacy.identifier(active.get(key)),
                                  'estado':'sem_modelo_ativo_legivel'})
    stats=[]; ds_folder=(core.config.data/'datasets').resolve()
    files=sorted(ds_folder.glob('*.json'))
    for index,path in enumerate(files):
        match=DATASET_NAME.fullmatch(path.name)
        if not match:continue
        dsid=match[1]
        progress(index,f'Conferindo estatísticas e separações dos conjuntos: {index+1}/{len(files)}…')
        if path.resolve().parent!=ds_folder:
            stats.append({'dataset_id':dsid,'estado':'caminho_recusado'}); continue
        try:stats.append(analyze_dataset(read_json(path),dsid,privacy,contexts.get(dsid,())))
        except Exception as exc:stats.append({'dataset_id':dsid,**unavailable(exc)})
    existing={row['dataset_id'] for row in stats}
    for row in training_runs:
        if row['dataset_id'] not in existing and row['dataset_id']!='addition-0-3-v1':
            stats.append({'dataset_id':row['dataset_id'],'estado':'snapshot_ausente'})
            existing.add(row['dataset_id'])
    progress(0,'Organizando avaliações, resultados anteriores e logs anonimizados…')
    external=collect_external(core.config.root,privacy,examples)
    logs,log_stats=collect_logs(core.config.root,core.config.data,privacy)
    spec=implementation()
    captured=utc()
    manifest={'formato':'TAC-diagnostico-v1','versao_tac':__version__,'gerado_em':captured,
        'politica_privacidade':{'pesos_exportados':False,'textos_completos_exportados':False,
            'conversas_importadas_exportadas':False,'historico_chat_exportado':False,
            'origens_titulos_caminhos_exportados':False,'mensagens_livres_logs_exportadas':False,
            'exemplos':'Apenas strings iguais aos exercícios autorais do código são legíveis; demais textos viram marcadores HMAC efêmeros.',
            'anonimizacao':dict(privacy.redactions)}}
    report={
        'arquitetura.json':{**manifest,'modelos_ativos':architectures,'implementacao_atual':spec},
        'treinamento.json':{**manifest,'config_atual':privacy.sanitize(config),'recursos':hardware(),
            'execucoes':training_runs,'metricas_por_etapa':metrics_clean,
            'dados_indisponiveis':'Picos históricos de RAM/VRAM/CPU/GPU, se não registrados; batch/precisão antigos sem evidência não são inferidos dos defaults atuais.'},
        'checkpoints.json':{**manifest,'ativos':{k:privacy.identifier(v) for k,v in active.items()},'checkpoints':cp_rows,
            'nota':'Estados de checkpoints são os registros originais; epoca/candidato não significa aprovação. Use run_id e resultado da execução.'},
        'avaliacoes.json':{**manifest,'execucoes':evaluation_runs,'avaliacoes_externas_registradas':external,
            'testes_software':software_tests(core.config.root),'observacoes':notes,
            'logs_resumo':log_stats,'criterios_codigo_atual':spec['regras_aprovacao_codigo_atual'],
            'interpretacao':'Não há medição geral de inteligência. Métricas ausentes são null/indisponíveis, nunca zeros inventados. Nenhuma avaliação nova foi executada.'},
        'exemplos_de_respostas.json':{**manifest,'avaliacoes':examples,
            'nota':'Todos os exemplos registrados nos relatórios selecionados, com textos privados ou não verificáveis omitidos. Contagens/exact são as notas originais, não recalculadas após anonimização.'},
        'datasets_estatisticas.json':{**manifest,'conjuntos':stats,'acervo_apenas_contagens':{
            **counts,'fontes_por_tipo':[{'tipo':privacy.sanitize(r['kind'],'kind'),'quantidade':r['quantidade']} for r in counts['fontes_por_tipo']]},
            'importados_por_destino':privacy.sanitize(imports),'entradas_importadas_em_destinos_diferentes':imported_conflicts,
            'restricao':'Igualdade não comprova ausência de vazamento semântico. Estudos do usuário e nomes dos materiais não são exportados.'},
        'logs.txt':logs}
    for content in report.values():
        if isinstance(content,dict):content['politica_privacidade']['anonimizacao']=dict(privacy.redactions)
    report['resumo.md']=summary(report,manifest,runs,cp_rows,stats,notes)
    return report


def summary(report,manifest,runs,checkpoints,stats,notes):
    status=Counter(row['status'] for row in runs)
    # Usa apenas estados conhecidos para que texto livre não alcance o Markdown.
    status={k:status[k] for k in ('aprovado','rejeitado','interrompido','erro','treinando','concluido') if status[k]}
    lines=['# TAC — Diagnóstico técnico completo','',f"Gerado em {manifest['gerado_em']}. TAC {__version__}.",'',
        'Este pacote contém registros existentes. Não treinou, gerou respostas, executou testes ou modificou pesos.',
        'Não contém banco SQLite, modelos, livros, documentos completos, histórico privado, conversas importadas, caminhos pessoais ou credenciais.',
        'Textos não verificáveis como exercícios autorais foram substituídos por marcadores. A chave de anonimização não é exportada.','',
        f"Execuções: {len(runs)}. Checkpoints registrados/arquivos de reforço: {len(checkpoints)}. Conjuntos encontrados ou referenciados: {len(stats)}.",
        'Estados das execuções: '+json.dumps(status,ensure_ascii=False)+'.','', '## Como analisar','',
        '1. Consulte arquitetura.json para modelos ativos, camadas, tokenização, geração e regras atuais do código.',
        '2. Cruze run_id, checkpoint e dataset_id nos arquivos treinamento.json, checkpoints.json e avaliacoes.json.',
        '3. Compare promotion_checks e motivo_registrado. Critérios atuais do código são separados das decisões históricas.',
        '4. Os exemplos completos registrados estão em exemplos_de_respostas.json, sujeitos à anonimização. Suas notas originais não foram recalculadas.',
        '5. Consulte datasets_estatisticas.json para duplicatas, conflitos, tamanhos e separações; logs.txt traz exceções/avisos anonimizados.','',
        '## Como interpretar notas zero','',
        'zero_medido: houve amostras e a nota registrada foi zero. zero_sem_amostras: divisão com conjunto vazio; não mede desempenho.',
        'nao_registrado/null: o registro não contém essa medição. Uma interface que usa valor padrão zero pode exibi-la como zero sem ter avaliado respostas.',
        'zero_origem_indeterminada: há um zero salvo, mas faltam amostras/evidências para explicar sua origem. Não se presume a causa.','',
        '## Limites da auditoria','',
        'A loss de tokens e a correção de respostas completas são medidas diferentes. A mistura de tarefas e a retenção podem alterar a loss de treino sem melhorar o diálogo.',
        'As regras de conversa revisada são lexicais; não constituem um julgador semântico. Acertos importados são informativos. Novas execuções podem usar loss de validação importada na seleção protegida; teste não seleciona.',
        'Contexto e correções são tarefas fechadas. Há avaliação sintética limitada de negação em reports/research_0120; não equivale a avaliar inteligência geral. Pilotos isolados dessa pasta não fazem parte do histórico de produção deste ZIP.',
        'Igualdade normalizada detecta parte do vazamento; paráfrases e exposição histórica fora dos snapshots disponíveis não podem ser descartadas.',
        'Testes usados repetidamente no desenvolvimento não são testes cegos. Testes de software históricos verificam o programa, não medem inteligência.',
        'Picos históricos de recursos e parâmetros que não foram salvos permanecem indisponíveis. Nenhum valor padrão atual substitui silenciosamente um valor histórico.',
        'Os logs omitem mensagens livres e linhas de código para evitar exposição de conteúdo privado; mantêm classes, frames relativos e categorias literais.','',
        '## Observações registradas','']
    for note in notes:
        if note['tipo']=='criterios_registrados_nao_atingidos' and note['criterios']:
            lines.append(f"- Execução {note['run_id']}: critérios não atingidos: {', '.join(note['criterios'])}.")
        elif note['tipo']=='loss_caiu_sem_mais_respostas_exatas':lines.append(f"- Execução {note['run_id']}: loss caiu sem aumento das respostas exatas. É uma divergência observada, não uma causa identificada.")
        elif note['tipo']=='execucao_com_erro':lines.append(f"- Execução {note['run_id']}: erro registrado, categoria literal {note['categoria_literal']}; causa raiz não determinada.")
    if not notes:lines.append('Sem observações calculáveis nos registros disponíveis.')
    lines+=['','Todos os relatórios JSON usam UTF-8 e null para dados ausentes. Arquivos e SHA-256 constam em manifesto.json.']
    return '\n'.join(lines)+'\n'


def export_diagnostic(core,target,progress=lambda *_:None,overwrite=False):
    target=Path(target).expanduser().resolve()
    if target.suffix.lower()!='.zip':raise ValueError('Escolha um nome terminado em .zip para o diagnóstico.')
    if target.exists() and not overwrite:raise FileExistsError('Já existe um arquivo com esse nome. Escolha outro destino.')
    # Escolha explícita do usuário permite exportar fora da pasta TAC.
    if not target.parent.is_dir():raise ValueError('A pasta de destino não existe. Escolha uma pasta disponível.')
    report=collect(core,progress)
    def finite(value):
        if isinstance(value,float) and not math.isfinite(value):return None
        if isinstance(value,list):return [finite(v) for v in value]
        if isinstance(value,dict):return {k:finite(v) for k,v in value.items()}
        return value
    blobs={name:content.encode('utf-8') if isinstance(content,str) else
           json.dumps(finite(content),ensure_ascii=False,indent=2,allow_nan=False).encode('utf-8') for name,content in report.items()}
    manifest={'formato':'TAC-diagnostico-v1','versao_tac':__version__,'gerado_em':utc(),
              'arquivos':{name:{'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)} for name,data in blobs.items()}}
    temp=None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent,prefix='.tac_diagnostico_',suffix='.tmp',delete=False) as file:
            temp=Path(file.name)
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for name,data in blobs.items():archive.writestr(name,data)
            archive.writestr('manifesto.json',json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False).encode('utf-8'))
        with zipfile.ZipFile(temp) as archive:
            if archive.testzip() is not None or set(archive.namelist())!=set(NAMES)|{'manifesto.json'}:
                raise ValueError('Falha na verificação do ZIP. Nenhum diagnóstico foi entregue.')
            for name,meta in manifest['arquivos'].items():
                data=archive.read(name)
                if hashlib.sha256(data).hexdigest()!=meta['sha256']:raise ValueError('Falha de integridade no diagnóstico.')
                if name.endswith('.json'):json.loads(data)
        with temp.open('r+b') as file:file.flush(); os.fsync(file.fileno())
        if not overwrite and target.exists():raise FileExistsError('O destino passou a existir. Escolha outro nome.')
        os.replace(temp,target)
    finally:
        if temp:temp.unlink(missing_ok=True)
    return {'path':str(target),'bytes':target.stat().st_size,'files':len(blobs)+1,
            'runs':len(report['treinamento.json']['execucoes']),'status':'exportado'}
