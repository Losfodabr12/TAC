from tac.neural.closed_facts import arithmetic,novel_fact


def test_arithmetic_uses_calculation_not_exact_text_and_does_not_extract_random_numbers():
    assert arithmetic('Quanto é 1 - 2?','O resultado é -1.')['status']=='correto'
    assert arithmetic('Quanto é 1 - 2?','1 - 2 = -9.')['status']=='incorreto'
    assert arithmetic('Quanto é 3 + 2?','2 + 3 = 5.')['status']=='correto'
    assert arithmetic('Quanto é 3 + 2?','1 + 4 = 5.')['status']=='nao_avaliado'
    assert arithmetic('Quanto é 1.5 + 2.5?','4.')['status']=='nao_avaliado'
    assert arithmetic('Quanto é 1 - 2?','Você tem -1 opções e deve pensar nisso.')['status']=='nao_avaliado'
    assert arithmetic('Quanto é 1 / 0?','1.')['status']=='nao_avaliado'


def test_name_verifier_rejects_known_wrong_slot_false_positive_without_claiming_general_semantics():
    row={'skill':'nome_inedito','expected':'Seu nome é Lunavel.','actual':'Sua cor favorita é Lunavel.'}
    assert novel_fact(row)['status']=='incorreto'
    assert novel_fact({**row,'actual':'Vou te chamar de Lunavel.'})['status']=='correto'
    assert novel_fact({**row,'actual':'Você me disse que seu nome é Lunavel.'})['status']=='correto'
    assert novel_fact({**row,'actual':'Você é incrível, Lunavel, e há muitas possibilidades.'})['status']=='nao_avaliado'
    assert novel_fact({**row,'actual':'Seu nome é Outro.'})['status']=='incorreto'


def test_open_answers_and_unrecognized_negation_require_review():
    assert novel_fact({'skill':'situacao_aberta','expected':None,'actual':'Converse.'})['status']=='nao_avaliado'
    row={'skill':'negacao','expected':'Não.','actual':'Não.'}
    assert novel_fact(row)['status']=='correto'
    assert novel_fact({**row,'actual':'Sim.'})['status']=='incorreto'
    assert novel_fact({**row,'actual':'Não tenho certeza.'})['status']=='nao_avaliado'
