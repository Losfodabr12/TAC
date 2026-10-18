import pytest


def put(core, title, text, origin=None):
    core.gate.set_enabled(True)
    return core.memory.ingest(title, origin or title, "file", text)[0]


@pytest.mark.parametrize("question", ["qual e o nome do brasil?", "Qual é o nome oficial do Brasil?", "como se chama oficialmente o Brasil?", "QUAL O NOME COMPLETO DO BRASIL?"])
def test_specific_name_uses_exact_evidence_and_ignores_other_names(core, question):
    put(core, "Brasil", "Brasil, oficialmente República Federativa do Brasil, é um país da América do Sul.\nUm parágrafo longo de história que não deve acompanhar a resposta.\n" + "O Brasil tem diversos estados. " * 80)
    put(core, "Conversas", "Usuário: Meu nome é Lucas.\nAssistente: Muito prazer, Lucas.")
    answer = core.chat(question)
    assert answer["text"].startswith("Nome oficial informado pela fonte: República Federativa do Brasil.")
    assert len(answer["text"]) < 500
    assert "Lucas" not in answer["text"]
    assert "diversos estados" not in answer["text"]
    assert len(answer["sources"]) == 1
    assert answer["mode"] == "documental"
    assert answer["method"] == "extração por regras"
    for proof in answer["evidence"]:
        assert any(proof in chunk["text"] for chunk in core.memory.chunks(answer["sources"][0]["source_id"]))


def test_official_name_is_read_from_data_not_a_country_constant(core):
    put(core, "Lusônia", "Lusônia, oficialmente União das Ilhas de Lusônia, possui duas ilhas.")
    answer = core.chat("Qual é o nome oficial de Lusônia?")
    assert "Nome oficial informado pela fonte: União das Ilhas de Lusônia." in answer["text"]
    assert "Brasil" not in answer["text"]


def test_name_relation_with_explicit_subject_after_field(core):
    put(core, "Caderno", "O nome oficial de Aurória é Federação de Aurória.")
    answer = core.chat("Qual o nome de Aurória?")
    assert "Federação de Aurória" in answer["text"]
    assert answer["sources"]


def test_unknown_entity_is_not_answered_with_another_entity(core):
    put(core, "Brasil", "Brasil, oficialmente República Federativa do Brasil, possui vários estados.")
    answer = core.chat("Qual é o nome de Atlântida?")
    assert not answer["sources"]
    assert "Não encontrei o nome" in answer["text"]
    assert "República" not in answer["text"]


def test_missing_relationship_does_not_return_an_unrelated_passage(core):
    put(core, "Brasil", "O Brasil possui muitos rios. O nome do pesquisador é Carlos. Há informações geográficas neste documento.")
    answer = core.chat("Qual é o nome oficial do Brasil?")
    assert not answer["sources"]
    assert "Não encontrei" in answer["text"]


def test_conflicting_explicit_names_are_not_silently_resolved(core):
    put(core, "Aurória A", "Aurória, oficialmente União de Aurória, foi descrita neste relatório.")
    put(core, "Aurória B", "Aurória, oficialmente Federação de Aurória, foi descrita neste relatório.")
    answer = core.chat("Qual é o nome oficial de Aurória?")
    assert "valores diferentes" in answer["text"]
    assert len(answer["sources"]) == 2


def test_capital_is_bound_to_the_requested_entity(core):
    put(core, "Notas", "A capital de Aurória é Solária. A capital de Boreália é Nivária.")
    answer = core.chat("Qual é a capital de Boreália?")
    assert answer["text"].startswith("Capital informada pela fonte: Nivária.")
    assert "Solária" not in answer["text"]


def test_dialogue_questions_are_not_extracted_as_capital_facts(core):
    put(core, "Aurória", "Capital Solária\nTexto geográfico da fonte.")
    put(core, "Roteiro", "Usuário: Qual é a capital de Aurória?\nTAC: A capital de Aurória é Solária.\nQuem é a capital de Aurória\nOutra pergunta sem resposta.")
    answer = core.chat("Qual é a capital de Aurória?")
    assert answer["text"].startswith("Capital informada pela fonte: Solária.")
    assert "valores diferentes" not in answer["text"]


def test_short_infobox_capital_ignores_late_reference_titles(core):
    text = "Capital Solária\n" + "\n".join(f"Título de seção {i}\nTexto da seção {i}." for i in range(80)) + "\nCapital Americana da Cultura"
    put(core, "Aurória", text)
    answer = core.chat("Qual é a capital de Aurória?")
    assert answer["text"].startswith("Capital informada pela fonte: Solária.")
    assert "Americana da Cultura" not in answer["text"]


def test_generic_answer_is_one_grounded_sentence(core):
    put(core, "Python", "Python é uma linguagem de programação. O SQLite armazena dados. Outros conteúdos não devem ser despejados no chat.")
    answer = core.chat("O que é Python?")
    assert "Python é uma linguagem de programação." in answer["text"]
    assert "SQLite" not in answer["text"]
    assert len(answer["sources"]) == 1


def test_raw_search_remains_available_when_explicitly_requested(core):
    put(core, "Python", "Python é uma linguagem de programação. O SQLite armazena dados.")
    answer = core.chat("Mostre os trechos sobre Python")
    assert "Trechos recuperados" in answer["text"]
    assert "SQLite" in answer["text"]


def test_selected_document_limits_the_answer_sources(core):
    source_id = put(core, "Aurória", "Aurória, oficialmente União de Aurória, tem informações no relatório.")
    put(core, "Outro relatório", "Aurória, oficialmente Federação de Aurória, tem outra descrição.")
    answer = core.chat("Qual o nome oficial de Aurória?", source_id=source_id)
    assert "União de Aurória" in answer["text"]
    assert "Federação" not in answer["text"]


def test_missing_attribute_is_not_replaced_by_general_topic_match(core):
    put(core, "Aurória", "Aurória é uma ilha que tem diversas praias. O turismo é uma atividade descrita neste relatório.")
    answer = core.chat("Qual é a moeda de Aurória?")
    assert not answer["sources"]
    assert "Não encontrei" in answer["text"]
