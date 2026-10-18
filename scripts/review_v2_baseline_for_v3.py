"""Separate qualitative inspection of already generated v2 baseline answers."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tac.config import atomic_json
OUT=ROOT/'reports/tac_v3/rodada_20261018_01'
models={}
for label,accepted in [('v2_referencia',{7}),('v2_conversacao',{9,12,28,32})]:
    original=json.loads((OUT/('geracoes_'+label+'.json')).read_text(encoding='utf-8'))
    rows=[]
    for row in original['outputs']:
        i=row['id'];category=row['category'];ok=i in accepted
        if ok:
            diagnosis={7:'Informa Talvo, o nome atual correto; a redação é ruim, mas o valor solicitado está inequívoco.',
                9:'Recupera Luzia como primeiro nome.',12:'Recupera a última escolha, verde.',
                28:'Cumprimento pertinente apesar da grafia informal Opaa.',32:'Recupera Carlos corretamente.'}[i]
        elif category=='arithmetic':diagnosis='Não fornece o resultado correto do cálculo solicitado.'
        elif category=='temporal_context':
            diagnosis=('Responde com Luzia, confundindo o primeiro nome com o papel solicitado.' if label=='v2_conversacao' and i in (7,8)
                       else 'Não identifica de maneira inequívoca o papel temporal ou a sequência solicitada.')
        elif i==11:diagnosis='Não associa Cora à preferência por ameixa; produz fragmentos ou entidades incorretas.'
        elif i==12:diagnosis='Responde branca, embora o contexto termine em verde.'
        elif i==13:diagnosis=('Menciona arvélio, mas viola o pedido de somente copiar o nome e acrescenta afirmações incoerentes.'
                                  if label=='v2_referencia' else 'Corrompe Arvélio em Av; não copia o nome.')
        elif category=='general_explanation':diagnosis='Não explica o fato solicitado; há tautologia ou afirmações desconexas/incorretas.'
        elif category=='instruction':diagnosis='Não executa a transformação, lista, restrição ou resumo solicitado; inclui conteúdo sem suporte.'
        elif category=='writing':diagnosis='Não produz o tipo de texto solicitado; não há história, bilhete ou diálogo adequado.'
        elif category=='natural_conversation':diagnosis='Resposta desconectada do cumprimento, agradecimento ou pedido de esclarecimento.'
        elif i==28:diagnosis='Reconhece parte do cumprimento, mas termina em fragmento sem sentido.'
        elif i==29:diagnosis='Descrição de lobo factualmente inadequada; lobo de areia ou local/região não responde ao conceito.'
        elif i==30:diagnosis='Não responde 4 de forma correta; número errado ou cadeia de igualdades falsas.'
        elif i==31:diagnosis='Não cria uma história; inventa entidade ou afirmação sem relação com a instrução.'
        else:diagnosis='Não recupera Carlos, limitando-se a repetir meu nome.'
        rows.append({'id':i,'appropriate':ok,'diagnosis':diagnosis,
                     'expected_value_literal_present':row['expected'] in row['actual']})
    models[label]={'rows':rows,'main_appropriate':sum(r['appropriate'] for r in rows if r['id']<=27)}
atomic_json(OUT/'REVISAO_V2_BASELINES.json',{'reviewer':'Inspeção qualitativa do agente Codex; não certificação humana independente',
             'original_outputs_modified':False,'earlier_reports_or_grades_changed':False,'models':models})
print({label:model['main_appropriate'] for label,model in models.items()})
