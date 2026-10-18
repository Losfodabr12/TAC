"""Reconstitui sorteios congelados e confere contadores, sem carregar pesos."""
import json
import argparse
import sys
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tac.config import atomic_json
from tac.fluency_lessons import lessons
from tac.neural.controlled import ResponseObjective, Exposure, row_key,category
from tac.neural.conversation import ConversationData
from tac.neural.conversation_corpus import TextReplay
from scripts.run_controlled_experiments import LiveSettings, DATASET, OLD, CORPUS, OUT, full_retention, arm_plan


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();OUT=args.output.resolve()
    if not OUT.is_relative_to(ROOT/'reports') or OUT==ROOT/'reports':parser.error('Escolha uma subpasta de TAC/reports.')
    config=LiveSettings(ROOT)
    data=ConversationData.load(config.data/'datasets'/f'conversation_{DATASET}.json')
    old=ConversationData.load(config.data/'datasets'/f'conversation_{OLD}.json')
    full=full_retention(data,old);replay=TextReplay(config.data/'datasets'/(CORPUS+'.json'),data.tokenizer)
    objective=ResponseObjective(replay,lessons()['train'],data.tokenizer)
    retained={row_key(r) for r in data.sets['train'] if r.get('mixture','legacy')=='legacy'}
    full_buckets=Counter((r['skill'],len(data.tokenizer.encode(' '+r['answer'],special=False))+1) for r in full)
    total_imported=sum(r.get('mixture')=='imported' for r in data.sets['train'])
    total_groups=len({r['group'] for r in data.sets['train'] if r.get('mixture')=='imported'})
    records={}
    for path in sorted(OUT.glob('*_s*/result.json')):
        result=json.loads(path.read_text(encoding='utf-8'))
        batches,plan=arm_plan(data,objective,full,result['seed'],120,result['arm'])
        if plan!=result['plan']:raise AssertionError(f'Sorteio não reproduzido: {path.parent.name}')
        exposure=Exposure();legacy=set();import_indices=set();eos=Counter();draws=Counter();matched_keys=set()
        input_tokens=Counter();padded_inputs=Counter();category_seen=defaultdict(Counter);category_groups=defaultdict(set)
        for pool,x,y,rows in batches:
            exposure.add(pool,rows,y)
            input_tokens[pool]+=int((x!=0).sum());padded_inputs[pool]+=x.numel()
            for i,row in enumerate(rows):
                eos[pool+'/'+row['skill']]+=int((y[i]==data.tokenizer.eos).sum())
                if pool=='legacy':
                    key=row_key(row);legacy.add(key);draws['retained' if key in retained else 'previously_excluded']+=1
                    matched_keys.add((row['skill'],int((y[i]!=0).sum())))
                if pool=='imported':
                    import_indices.add(id(row));cat=category(row)
                    category_seen[cat][row_key(row)]+=1;category_groups[cat].add(row['group'])
        if exposure.report()!=result['exposure']:raise AssertionError('Contadores divergentes: '+path.parent.name)
        e=exposure.report()['pools']['imported']
        records[path.parent.name]={'schedule_and_exposure_reproduced':True,
            'imported_unique_row_indices':len(import_indices),'imported_pool_rows':total_imported,
            'imported_row_coverage':len(import_indices)/total_imported,
            'imported_unique_content':e['unique_examples'],'imported_unique_groups':e['unique_groups'],
            'imported_pool_groups':total_groups,'imported_group_coverage':e['unique_groups']/total_groups,
            'legacy_unique_content':len(legacy),'legacy_from_previously_excluded_unique':sum(k not in retained for k in legacy),
            'full_pool_reachable_by_skill_and_target_length_in_this_schedule':sum(full_buckets[k] for k in matched_keys),
            'legacy_draws_origin':dict(draws),'supervised_eos_by_original_skill':dict(eos),
            'input_nonpad_tokens_by_pool':dict(input_tokens),'input_positions_including_padding_by_pool':dict(padded_inputs),
            'imported_categories':{k:{'unique_content':len(v),'draws':sum(v.values()),'repeats':sum(v.values())-len(v),
                'unique_groups':len(category_groups[k])} for k,v in category_seen.items()},
            'note':'Índices de linha distintos podem ter o mesmo texto completo e resposta. Não confundir esses dois contadores.'}
        print(path.parent.name,'auditado',flush=True)
    atomic_json(OUT/'cobertura_reproduzida.json',{'results':records,'weights_loaded':False,'optimizer_created':False})


if __name__=='__main__':main()
