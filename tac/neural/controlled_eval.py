"""Medições de desenvolvimento; não ativam modelos nem abrem o teste final."""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import torch

from ..conversation_lessons import prompt_text
from ..fluency_lessons import score_reply, fluency_ready
from .conversation import ConversationEngine, context_ready, natural_ready
from .research import stratified
from .transfer_eval import evaluate_suite


@torch.inference_mode()
def trace_reply(model, messages, device, max_tokens=128):
    tokenizer = model.text_tokenizer
    ids = tokenizer.encode(prompt_text(messages), special=False)
    x = torch.tensor([[tokenizer.bos, *ids[-model.cfg.context+1:]]], device=device)
    generated = model.generate(x, max_new_tokens=max_tokens, greedy=True)
    new = generated[0, x.shape[1]:].tolist()
    text = tokenizer.decode(new).strip()
    words = re.findall(r'\w+', text.casefold())
    triples = Counter(tuple(words[i:i+3]) for i in range(max(0, len(words)-2)))
    return text, {'eos': bool(new and new[-1] == tokenizer.eos), 'generated_tokens': len(new),
        'hit_limit_without_eos': len(new) == max_tokens and new[-1] != tokenizer.eos,
        'user_role_emitted': bool(re.search(r'usu[aá]rio\s*:', text, re.I)),
        'assistant_role_emitted': bool(re.search(r'\bTAC\s*:', text, re.I)),
        'repeated_trigram_3_times': max(triples.values(), default=0) >= 3, 'empty': not bool(text)}


class DevelopmentEvaluator(ConversationEngine):
    """Mesmas respostas exatas e critérios internos; instrumentação adicional separada."""
    def measured(self, model, data, device, split):
        result = super().evaluate(model, data, device, split=split)
        outputs = []
        for row in data.sets[split]:
            self.gate.check()
            text, trace = trace_reply(model, row['messages'], device)
            accepted = row.get('acceptable_answers', [row['answer']])
            outputs.append({'messages': row['messages'], 'skill': row['skill'], 'expected': row['answer'],
                'acceptable_answers': accepted, 'actual': text, 'exact': text in accepted, **trace,
                'strict_prefix_of_reference': bool(text and any(a.startswith(text) and a != text for a in accepted)),
                'semantic_equivalence': None})
        result.update(outputs=outputs, exact_matches=sum(r['exact'] for r in outputs),
            exact_accuracy=sum(r['exact'] for r in outputs)/max(1,len(outputs)),
            by_skill={s:{'correct':sum(r['exact'] for r in outputs if r['skill']==s),
                'total':sum(r['skill']==s for r in outputs)} for s in sorted({r['skill'] for r in outputs})})
        return result

    def sessions(self, model, data, device):
        sessions=[]; traces=[]
        rows=[r for r in data.sets['validation_natural'] if r['skill'].endswith('_recall') and len(r['messages'])==5]
        for row in rows:
            history=[]
            for index in (0,2,4):
                self.gate.check()
                user=row['messages'][index]
                text,trace=trace_reply(model,[*history[-6:],user],device)
                traces.append(trace);history.extend([user,['assistant',text]])
            sessions.append({'skill':row['skill'],'history':history,'expected_immediate':row['messages'][3][1],
                'expected_recall':row['answer'],'immediate_correct':history[3][1]==row['messages'][3][1],
                'recall_correct':history[5][1] in row.get('acceptable_answers',[row['answer']])})
        n=max(1,len(sessions))
        return {'examples':len(sessions),'immediate_correct':sum(r['immediate_correct'] for r in sessions),
            'recall_correct':sum(r['recall_correct'] for r in sessions),
            'immediate_accuracy':sum(r['immediate_correct'] for r in sessions)/n,
            'recall_accuracy':sum(r['recall_correct'] for r in sessions)/n,'sessions':sessions,'traces':traces}

    def fluency(self, model, data, device):
        episodes=defaultdict(list)
        for row in data.sets['validation_fluency']:episodes[row['episode']].append(row)
        sessions=[];scores=[];traces=[]
        for episode,rows in episodes.items():
            history=[];turns=[]
            for row in sorted(rows,key=lambda r:r['turn']):
                self.gate.check();user=row['messages'][-1]
                actual,trace=trace_reply(model,[*history[-6:],user],device)
                score=score_reply(actual,row['relevance_rules']);scores.append(score);traces.append(trace)
                turns.append({'user':user[1],'actual':actual,'reference':row['answer'],
                    'rules':row['relevance_rules'],**score,**trace})
                history.extend([user,['assistant',actual]])
            sessions.append({'episode':episode,'turns':turns})
        n=max(1,len(scores))
        return {'episodes':len(sessions),'turns':len(scores),'rule_matches':sum(s['passes_simple_rules'] for s in scores),
            'rule_accuracy':sum(s['passes_simple_rules'] for s in scores)/n,
            'slot_drift_rate':sum(s['slot_drift'] for s in scores)/n,'sessions':sessions,'traces':traces,
            'method':'Regras lexicais originais, com falsos positivos conhecidos; não são nota semântica.'}

    def all(self, model, data, device, novel):
        result={}
        for label,split in (('basic','validation'),('context','validation_context'),('natural','validation_natural')):
            result[label]=self.measured(model,data,device,split)
        result['sessions']=self.sessions(model,data,device)
        result['fluency']=self.fluency(model,data,device)
        # Toda a validação importada entra na loss; geração em painel estratificado fixo.
        result['imported_full_loss']=super().evaluate(model,data,device,split='validation_imported')
        from .conversation import ConversationData
        panel=ConversationData({'train':[], 'validation_imported':stratified(data.sets['validation_imported'],60)},data.tokenizer)
        result['imported']=self.measured(model,panel,device,'validation_imported')
        result['imported']['history_policy']='histórico ouro do arquivo; painel fixo de desenvolvimento, não o teste'
        result['imported_sessions']=self.evaluate_imported(model,panel,device,'validation_imported')
        result['novel']=evaluate_suite(model,novel,device,self.gate)
        result['novel']['scope']='Desenvolvimento já exposto na v0.12; reutilizado explicitamente. Não é teste cego.'
        traces=[r for key in ('basic','context','natural','imported') for r in result[key]['outputs']]
        traces+=result['sessions']['traces']+result['fluency']['traces']
        keys=('eos','hit_limit_without_eos','user_role_emitted','assistant_role_emitted','repeated_trigram_3_times','empty')
        result['generation']={'responses':len(traces), **{key:sum(t[key] for t in traces) for key in keys},
            'strict_prefix_of_reference':sum(r['strict_prefix_of_reference'] for key in ('basic','context','natural','imported') for r in result[key]['outputs']),
            'scope':'Básicas, contexto, naturais, painel importado ouro, sessões próprias e conversas revisadas. Prefixo curto é indício, não diagnóstico semântico.'}
        result['test_final_evaluated']=False
        return result


def protection_checks(current, baseline, completed=True):
    """Reprodução literal dos limiares existentes, sem promoção ou critérios novos."""
    basic=current['basic'];old=baseline['basic'];unknown=basic['by_skill'].get('unknown',{})
    return {'treino_concluido':completed, 'loss_basica':basic['loss'] < old['loss']*1.02,
        'respostas_basicas':basic['exact_accuracy'] >= max(.60,old['exact_accuracy']-.02),
        'desconhecimento':not unknown or unknown['correct']/max(1,unknown['total'])>=.8,
        'contexto_anterior':context_ready(current['context']),
        'dialogos_e_correcoes':natural_ready(current['natural'],baseline['natural']),
        'sessoes_com_respostas_proprias':current['sessions']['immediate_accuracy']>=.8 and current['sessions']['recall_accuracy']>=.8,
        'conversas_revisadas':fluency_ready(current['fluency'],baseline['fluency'])}
