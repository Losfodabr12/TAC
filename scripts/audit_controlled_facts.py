"""Verificadores fechados adicionais: não substituem as notas anteriores."""
import json
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tac.config import atomic_json
from tac.neural.closed_facts import arithmetic,novel_fact

OUT=ROOT/'reports/experimentos_0130'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def measure(e):
    rows=[]
    for row in e['context']['outputs']:
        if row['skill']=='arithmetic':
            rows.append({'skill':'aritmetica','question':row['messages'][-1][1],'actual':row['actual'],
                'expected':row['expected'],'literal_exact':row['exact'],**arithmetic(row['messages'][-1][1],row['actual'])})
    for row in e['novel']['results']:
        rows.append({'skill':row['skill'],'id':row['id'],'actual':row['actual'],'expected':row['expected'],
            'literal_exact':row['exact'],**novel_fact(row)})
    groups=defaultdict(Counter)
    for row in rows:groups[row['skill']][row['status']]+=1
    return {'by_skill':{s:{'total':sum(v.values()),'correct':v['correto'],'incorrect':v['incorreto'],
        'unassessed':v['nao_avaliado']} for s,v in groups.items()},'rows':rows,
        'literal_rejected_but_closed_fact_correct':[r for r in rows if r['literal_exact'] is False and r['status']=='correto']}


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();OUT=args.output.resolve()
    if not OUT.is_relative_to(ROOT/'reports') or OUT==ROOT/'reports':parser.error('Escolha uma subpasta de TAC/reports.')
    evaluations={'active':read(OUT/'active.json')['evaluation']}
    for path in sorted(OUT.glob('*_s*/result.json')):evaluations[path.parent.name]=read(path)['evaluation']
    atomic_json(OUT/'auditoria_factual_fechada.json',{'format':'TAC-closed-facts-exploratory-v1',
        'scope':'Auditoria adicional exploratória. Gramáticas restritas de números, nomes, código, local e sim/não. Não mede semântica geral nem altera seleção/aprovação.',
        'unrecognized_policy':'null/sem avaliação, nunca converter ausência de gramática em acerto. Contagens totais incluem não avaliados.',
        'final_test_used':False,'results':{k:measure(e) for k,e in evaluations.items()}})
    print('Auditoria factual fechada:',len(evaluations),'modelos registrados; nenhum modelo carregado.')


if __name__=='__main__':main()
