"""Direct free generation, same development prompts, no final reserved tests."""
import argparse
import gc
import json
import re
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
import torch
from tac.core import Core
from tac.config import atomic_json
from tac.neural.decoding import decode_conversation,conversation_options

OUT=ROOT/'reports/tac_v3/rodada_20261018_01'


def cases():
    result=[]
    def add(category,text,expected,history=None,verification=None,known=False):
        result.append({'id':len(result)+1,'category':category,'messages':(history or [])+[['user',text]],
                       'expected':expected,'verification':verification,'known_probe':known})
    for q,a in [('Quanto dá sete mais oito?',15),('Calcule 18 menos 7.',11),('Qual é o produto de 6 por 4?',24),
                ('Divida 20 por 5.',4),('Some 5 e 9.',14),('Quanto sobra de 8 depois de tirar 3?',5)]:
        add('arithmetic',q,str(a),verification={'number':a})
    history=[['user','Meu nome é Luzia.'],['assistant','Entendi.'],['user','Passei a me chamar Néris.'],
             ['assistant','Entendi a mudança.'],['user','Agora meu nome é Talvo.']]
    for q,a in [('Como eu me chamo agora?','Talvo'),('Qual nome eu usava imediatamente antes do atual?','Néris'),
                ('Diga o primeiro nome que informei.','Luzia'),('Liste os três nomes na ordem em que foram informados.','Luzia, Néris, Talvo')]:
        add('temporal_context',q,a,history,verification={'names':a.replace(',','').split()})
    add('context_binding','Quem prefere ameixa?', 'Cora', [['user','Cora prefere ameixa e Ivo prefere pêssego.']],{'names':['Cora']})
    add('context_update','Qual cor escolhi por último?', 'verde', [['user','Escolhi a cor lilás.'],['assistant','Certo.'],['user','Troquei minha escolha para verde.']],{'names':['verde']})
    add('name_copy','Escreva somente o nome que acabei de informar.','Arvélio',[['user','Pode me chamar de Arvélio.']],{'names':['Arvélio']})
    for q,a in [('Por que existem o dia e a noite?','A rotação da Terra alterna a região iluminada pelo Sol.'),
                ('Explique o que uma planta precisa para fazer fotossíntese.','Luz, água e dióxido de carbono; produz matéria orgânica e libera oxigênio.'),
                ('Descreva um lobo para uma criança.','Mamífero da família dos cães, animal selvagem que pode viver em grupos.'),
                ('Por que o gelo derrete quando fica fora do congelador?','Recebe calor do ambiente e passa do estado sólido para o líquido.')]:
        add('general_explanation',q,a)
    for q,a in [('Diga três nomes de frutas.','Três frutas reais, sem itens de outras categorias.'),
                ('Passe para o plural: A menina leu um livro.','As meninas leram uns livros.'),
                ('Responda com duas frases sobre cuidar das árvores.','Duas frases coerentes sobre o tema.'),
                ('Resuma em uma frase: A chuva começou de manhã. A praça ficou vazia. A feira foi adiada.','A chuva esvaziou a praça e adiou a feira.')]:
        add('instruction',q,a)
    for q in ('Conte uma história de três frases sobre um barco de papel.',
              'Escreva um bilhete curto agradecendo a ajuda de um vizinho.',
              'Crie um pequeno diálogo entre duas amigas planejando um passeio.'):
        add('writing',q,'Texto coerente, relacionado à instrução, sem loops; revisar semanticamente.')
    for q in ('E aí, como você está?', 'Muito obrigado pela ajuda!', 'Não entendi a explicação. Pode dizer de outro jeito?'):
        add('natural_conversation',q,'Resposta pertinente à fala; revisar semanticamente.')
    for q,a in [('Opa, tudo bom?','Cumprimento natural.'),('O que é um lobo?','Mamífero canídeo selvagem.'),
                ('Quanto é 2 + 2?','4'),('Crie uma história curta.','História coerente.'),
                ('Meu nome é Carlos. Qual é meu nome?','Carlos')]:
        add('known_user_probe',q,a,known=True)
    return result


def automatic(row,text):
    verification=row['verification']
    if not verification:return {'automatic_factual_correct':None}
    if 'number' in verification:
        values=re.findall(r'(?<!\w)-?\d+(?:[.,]\d+)?(?!\w)',text)
        # Multiple contradictory numbers or prose are left for manual review,
        # rather than accepting a correct numeral hidden inside nonsense.
        correct=len(values)==1 and float(values[0].replace(',','.'))==verification['number']
        return {'automatic_factual_correct':correct,'rule':'one numeric value only; semantic review separate'}
    names=verification['names'];normalized=text.casefold()
    positions=[normalized.find(n.casefold()) for n in names]
    return {'automatic_factual_correct':None,'expected_names_present_case_insensitive':all(i>=0 for i in positions),
            'exact_expected_names_present':all(n in text for n in names),
            'expected_name_order_present':all(i>=0 for i in positions) and positions==sorted(positions),
            'rule':'name containment is diagnostic only, not factual approval'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--label',required=True)
    args=p.parse_args();core=Core(ROOT);torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.60)
    target=OUT/('geracoes_'+args.label+'.json')
    if target.exists():print('Existing evaluation preserved:',target);return
    model,cp=core.conversation.load(args.checkpoint,'cuda');model.eval()
    outputs=[]
    dev_path=OUT/'avaliacao_dev.json'
    evaluation_cases=json.loads(dev_path.read_text(encoding='utf-8'))['cases'] if dev_path.exists() else cases()
    for row in evaluation_cases:
        with torch.autocast('cuda',dtype=torch.bfloat16):
            result=decode_conversation(model,row['messages'],'cuda',max_tokens=128,
                                       generation_options=conversation_options('controlled',seed=42))
        outputs.append({**row,'actual':result['text'],'decoding':result,**automatic(row,result['text'])})
        print(args.label,row['id'],repr(result['text'])[:140],flush=True)
    sessions=[];messages=[]
    for question in ('Olá, tudo bem?', 'Quero conversar sobre animais. O que é um lobo?',
                     'Agora vamos mudar de assunto: quanto é 2 + 2?', 'Meu nome é Safira.', 'Qual é meu nome?'):
        messages.append(['user',question])
        with torch.autocast('cuda',dtype=torch.bfloat16):
            result=decode_conversation(model,messages,'cuda',max_tokens=128,generation_options=conversation_options('controlled',seed=42))
        sessions.append({'input_messages':[m[:] for m in messages],'actual':result['text'],'decoding':result})
        messages.append(['assistant',result['text']])
    atomic_json(target,{'checkpoint':args.checkpoint,'generation':cp.get('generation'),'parameters':model.parameter_count,
                'config':cp['model_config'],'generation_options':conversation_options('controlled',seed=42),
                'max_new_tokens':128,'heldout_development':True,'reserved_tests_opened':False,
                'precision':'bfloat16_autocast',
                'outputs':outputs,'self_generated_session':sessions,
                'limitations':['Different tokenizers mean 128 tokens allow different amounts of text.',
                    'The five known user probes are reported separately from new development cases.',
                    'Lexical overlap and teacher-forced loss do not certify semantic correctness.']})
    print('SAVED',target,flush=True)


if __name__=='__main__':main()
