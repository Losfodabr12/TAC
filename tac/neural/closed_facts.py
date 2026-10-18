"""Auditoria exploratória de respostas FECHADAS; não pontua semântica geral."""
import re
import unicodedata
from fractions import Fraction


def plain(text):
    return ' '.join(''.join(c for c in unicodedata.normalize('NFKD',text.casefold()) if not unicodedata.combining(c)).split())


def answer_number(text):
    t=plain(text).rstrip('.! ')
    match=re.fullmatch(r'(?:o resultado e |resultado: ?)?(-?\d+(?:[.,]\d+)?)',t)
    return Fraction(match[1].replace(',','.')) if match else None


def arithmetic(question,actual):
    query=re.fullmatch(r'(?:quanto e |calcule:? ?)?(-?\d+)\s*([+*/x×÷-])\s*(-?\d+)',plain(question).rstrip(' ?.!'))
    if not query:return {'status':'nao_avaliado','reason':'pergunta fora da gramática numérica'}
    a,op,b=query.groups();a=Fraction(a);b=Fraction(b);op={'x':'*','×':'*','÷':'/'}.get(op,op)
    if op=='/' and b==0:return {'status':'nao_avaliado','reason':'divisão por zero'}
    expected={'+':lambda:a+b,'-':lambda:a-b,'*':lambda:a*b,'/':lambda:a/b}[op]()
    t=plain(actual).rstrip('.! ')
    equation=re.fullmatch(r'(-?\d+)\s*([+*/x×÷-])\s*(-?\d+)\s*=\s*(-?\d+(?:[.,]\d+)?)',t)
    if equation:
        left,symbol,right,value=equation.groups();symbol={'x':'*','×':'*','÷':'/'}.get(symbol,symbol)
        left=Fraction(left);right=Fraction(right);value=Fraction(value.replace(',','.'))
        if symbol=='/' and right==0:return {'status':'nao_avaliado','reason':'equação divide por zero'}
        stated={'+':lambda:left+right,'-':lambda:left-right,'*':lambda:left*right,'/':lambda:left/right}[symbol]()
        same=symbol==op and ((left==a and right==b) or (op in ('+','*') and left==b and right==a))
        if stated==value and not same:return {'status':'nao_avaliado','reason':'equação verdadeira com outros operandos; relação contextual exige revisão'}
        correct=same and value==expected and stated==value
    else:
        value=answer_number(actual)
        if value is None:return {'status':'nao_avaliado','reason':'resposta não reconhecida; exige revisão humana'}
        correct=value==expected
    return {'status':'correto' if correct else 'incorreto','reason':'aritmética calculada, gramática fechada; sem comparação literal obrigatória'}


def novel_fact(result):
    skill=result['skill'];expected=result['expected'];actual=plain(result['actual']).rstrip('.! ')
    if expected is None:return {'status':'nao_avaliado','reason':'situação aberta: revisão humana'}
    target=plain(expected).rstrip('.! ')
    if skill in ('nome_inedito','correcao_negada','memoria_curta','memoria_apos_janela'):
        target=target.removeprefix('seu nome e ')
        name=re.fullmatch(r'(?:certo, )?(?:voce me disse que )?(?:seu nome e |voce se chama |vou te chamar de |vou chamar voce de )([\w-]+)',actual)
        if name:value=name[1]
        elif actual==target:value=actual
        elif re.fullmatch(r'(?:sua cor favorita e |voce mora em |seu hobby e )[\w-]+',actual):
            return {'status':'incorreto','reason':'resposta usa outro atributo, não o nome'}
        else:return {'status':'nao_avaliado','reason':'frase fora da gramática fechada de nomes'}
    elif skill=='negacao':
        if actual not in ('sim','nao'):return {'status':'nao_avaliado','reason':'negação em frase exige revisão humana'}
        value=actual
    elif skill=='local_atual':
        target=target.removeprefix('em ')
        location=re.fullmatch(r'(?:em |o estojo esta em )([\w-]+)',actual)
        if location:value=location[1]
        else:return {'status':'nao_avaliado','reason':'local fora da gramática fechada'}
    elif skill=='pergunta_indireta':
        code=re.fullmatch(r'(?:o codigo (?:da caixa )?e )?([\w-]+)',actual)
        if code:value=code[1]
        else:return {'status':'nao_avaliado','reason':'código fora da gramática fechada'}
    else:return {'status':'nao_avaliado','reason':'habilidade sem verificador fechado'}
    return {'status':'correto' if value==target else 'incorreto',
        'reason':'atributo e valor verificados em gramática restrita; frases não reconhecidas ficam sem nota'}
