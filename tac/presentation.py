"""Estado da aplicação e mensagens em linguagem de uso, sem carregar a rede."""
from __future__ import annotations


QUEUE_LABELS = {"pendente": "Aguardando início", "processando": "Estudando agora",
                "concluido": "Disponível para consulta", "parcial": "Concluído com avisos",
                "erro": "Não foi possível estudar"}
RESULT_LABELS = {"estudado": "Adicionado à biblioteca", "atualizado": "Fonte atualizada",
                 "duplicado": "Conteúdo já conhecido", "inalterado": "Sem alterações",
                 "erro": "Falha", "erro_pagina": "Página não estudada"}


def answer_origin(answer):
    import json
    mode = answer.get('mode', '')
    sources = answer.get('sources', [])
    if isinstance(sources, str):
        try:
            sources = json.loads(sources)
        except ValueError:
            sources = []
    if mode == 'consulta web documental':
        return 'Resposta da Wikipédia' if sources else 'Consulta à Wikipédia sem resposta'
    if mode == 'conversa guiada':
        return 'Resposta preparada do programa'
    if mode == 'diálogo neural experimental':
        return 'Resposta da rede da TAC'
    if mode == 'neural experimental':
        return 'Continuação criada pela rede de escrita'
    return 'Resposta dos seus documentos' if sources else 'Consulta aos seus documentos sem resposta'


def training_summary(result, kind, active):
    """Traduz resultados existentes. Não calcula aprovação nem altera métricas."""
    status = result.get('status', '')
    title = 'conversa' if kind == 'conversation' else 'escrita'
    good, bad = [], []

    def compare(before, after, description, lower=False):
        if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
            return
        delta = (before - after) if lower else (after - before)
        if abs(delta) > 1e-9:
            (good if delta > 0 else bad).append(description)

    if kind == 'conversation':
        before, after = result.get('baseline') or {}, result.get('validation') or {}
        compare(before.get('exact_accuracy'), after.get('exact_accuracy'), 'Acertos nas respostas básicas.')
        compare(before.get('loss'), after.get('loss'), 'Previsão das respostas dos exercícios.', lower=True)
        for key, metric, description in (
            ('natural', 'exact_accuracy', 'Acertos nos exercícios de mudanças e correções.'),
            ('session', 'immediate_accuracy', 'Acertos ao responder às mudanças na conversa.'),
            ('session', 'recall_accuracy', 'Acertos ao lembrar a informação corrigida.'),
            ('fluency', 'rule_accuracy', 'Respostas que passaram na verificação simples do assunto.'),
        ):
            compare((result.get(key + '_baseline') or {}).get(metric),
                    (result.get(key + '_validation') or {}).get(metric), description)
    else:
        compare(result.get('baseline'), result.get('validation'), 'Previsão dos textos separados para avaliação.', lower=True)
        compare(result.get('fixed_baseline'), result.get('fixed_final'), 'Desempenho nos exercícios anteriores.', lower=True)
    if result.get('text_baseline') and result.get('text_final'):
        compare(result['text_baseline'].get('loss'), result['text_final'].get('loss'),
                'Previsão de palavras nos textos de avaliação.', lower=True)
    if status == 'interrompido':
        outcome = 'Interrompido. A avaliação final está pendente.'
        reason = 'A execução parou antes de concluir todas as verificações.'
    elif status == 'aprovado':
        outcome = 'Aprovado. A nova versão passou nas verificações desta execução.'
        reason = 'Os critérios do treino foram atendidos. Isso não garante conversa livre.'
    elif status == 'rejeitado':
        outcome = 'Rejeitado. A nova versão não foi ativada.'
        checks = result.get('promotion_checks') or {}
        failed = {k for k,v in checks.items() if not v}
        conversation_failed = bool(failed & {'conversas_revisadas', 'contexto_anterior', 'dialogos_e_correcoes', 'sessoes_com_respostas_proprias'})
        if bad and conversation_failed:
            reason = 'A nova versão perdeu desempenho em verificações e ainda não respondeu bem aos exercícios de conversa.'
        elif conversation_failed:
            reason = 'As respostas ainda não atingiram o mínimo exigido nos exercícios de conversa.'
        elif bad:
            reason = 'A nova versão perdeu desempenho em verificações que protegem o aprendizado anterior.'
        else:
            reason = 'Nem todas as verificações exigidas para ativar a nova versão foram atendidas.'
    else:
        outcome, reason = 'Não concluído.', 'Não há resultado final disponível.'
    lines = [f'TREINO DE {title.upper()}', outcome, '', 'Motivo', reason, '', 'O que melhorou']
    lines += ['• ' + s for s in good] or ['Não houve melhora registrada nas comparações disponíveis.']
    lines += ['', 'O que piorou']
    lines += ['• ' + s for s in bad] or ['Não houve piora registrada nas comparações disponíveis.']
    if not good and not bad:
        lines.append('Os resultados ficaram iguais ou não há comparação suficiente.')
    lines += ['', 'O que está valendo agora', active]
    if kind == 'conversation':
        lines.append('A verificação do assunto é simples; não mede compreensão geral.')
    return '\n'.join(lines)


def imported_results_text(comparison):
    if not comparison:
        return ''
    lines = ['', 'Suas conversas: comparação por habilidade',
             'Comparação informativa com a versão que estava ativa antes deste treino. Não muda a aprovação.']
    if comparison.get('status') != 'concluido':
        lines.append('Comparação não concluída. Isso não foi usado como uma nota para aprovar ou rejeitar.')
        return '\n'.join(lines)
    for split, title in (('validation','Validação'),('test','Teste')):
        value = comparison['splits'].get(split,{})
        previous, candidate = value.get('active',{}), value.get('candidate',{})
        lines.append(title)
        if not candidate.get('examples'):
            lines.append('Nenhuma conversa reservada neste conjunto. Não há comparação disponível.')
            continue
        for skill, score in sorted(candidate['by_skill'].items()):
            baseline = previous['by_skill'].get(skill,{'correct':0,'total':0})
            before, after = baseline['correct'], score['correct']
            change = 'Mais respostas iguais às referências.' if after > before else 'Menos respostas iguais às referências.' if after < before else 'Mesma quantidade de acertos.'
            lines.append(f"• {skill.replace('_',' ')}: anterior {before}/{baseline['total']}; nova {after}/{score['total']}. {change}")
    lines.append('A comparação exige a mesma redação da resposta de referência. Outras formas corretas de responder podem não contar como acerto. Não mede compreensão geral.')
    return '\n'.join(lines)


def friendly_progress(message, job_name, phase='prepare'):
    """A apresentação acompanha as etapas informadas pelo processo existente."""
    plain = message.casefold()
    if job_name == 'Estudar fila':
        return message.replace('Indexando e salvando:', 'Guardando:').replace('Fonte #', 'Material '), phase
    if 'versão inicial' in plain or 'preparando' in plain:
        phase = 'prepare'
    elif 'avaliando' in plain or 'verificando' in plain:
        phase = 'evaluate'
    elif any(word in plain for word in ('diálogo ', 'exercícios ', 'loss inicial', 'loss resposta', 'parâmetros;')):
        phase = 'train'
    text = {'prepare':'Etapa 1 de 3: preparando os exemplos e verificando a versão em uso.',
            'train':'Etapa 2 de 3: praticando os exemplos. Ao terminar, a TAC verificará o resultado.',
            'evaluate':'Etapa 3 de 3: verificando as respostas antes de decidir se a nova versão será usada.'}[phase]
    if 'interromp' in plain:
        text = 'Interrompendo com segurança. Aguarde o fim da etapa atual.'
    return text, phase


def user_error(message):
    """Mensagem curta; o erro completo continua disponível no diagnóstico."""
    raw = str(message)
    lower = raw.casefold()
    if any(word in lower for word in ('vram', 'cuda out of memory')):
        return 'A memória da placa de vídeo não foi suficiente. Feche outros programas ou reduza o uso de memória nas Ferramentas avançadas.'
    if 'cuda indisponível' in lower:
        return 'A placa de vídeo não está disponível para este treino. Escolha processamento automático nos ajustes avançados.'
    if 'ram' in lower and ('limite' in lower or 'pouca' in lower):
        return 'Faltou memória livre no computador. Feche outros programas e tente novamente.'
    if any(word in lower for word in ('gradiente', 'state_dict', 'checkpoint', 'tensor', 'loss', 'tokenizador')):
        return 'Não foi possível concluir esta atividade. Veja a explicação completa nas Ferramentas avançadas ou nos registros de erros.'
    text = explain_error(raw).split('\nDetalhe:', 1)[0]
    return text.replace('Central de Aprendizado', 'Meus materiais').replace('Na Central,', 'Em Meus materiais,').replace('Ativar e estudar fila', 'Permitir e guardar materiais').replace('Consulta documental', 'Consultar meus materiais').replace('robots.txt', 'as regras do site')


def explain_error(message):
    """Retorna uma explicação e um próximo passo, mantendo os detalhes no log."""
    text = str(message)
    lower = text.lower()
    if "dados insuficientes" in lower:
        return "Ainda não há texto suficiente para treinar. Primeiro estude um documento ou site na Central de Aprendizado."
    if "nenhum modelo treinado" in lower:
        return "A geração neural ainda não está disponível. Use Consulta documental; ela funciona assim que um material for estudado."
    if "aprendizado desativado" in lower or "ative o aprendizado" in lower:
        return "O aprendizado está desligado. Na Central, clique em Ativar e estudar fila para autorizar e iniciar o estudo."
    if "robots.txt" in lower:
        return "O site não permitiu o acesso automático ou não foi possível verificar robots.txt. A TAC não estudou a página. Você pode importar uma cópia textual que tenha autorização para usar.\nDetalhe: " + text
    if any(x in lower for x in ("http 403", "http 429", "http 401")):
        return "O site bloqueou ou limitou o acesso automático. A TAC não estudou essa página e não tentará contornar a proteção. Tente mais tarde ou importe um arquivo autorizado.\nDetalhe: " + text
    if any(x in lower for x in ("name resolution", "getaddrinfo", "falha dns", "gaierror")):
        return "Não foi possível localizar o site na rede. Confira o endereço e sua conexão, depois selecione o item e clique em Tentar novamente."
    if any(x in lower for x in ("timed out", "timeout", "tempo máximo")):
        return "O site demorou demais para responder. O material não foi concluído. Você pode tentar novamente; nenhum limite de acesso foi contornado."
    if "página maior" in lower or "texto do site excede" in lower:
        return "A página ultrapassou o limite de tamanho configurado. Escolha uma página menor ou ajuste o limite nas Configurações."
    if "vram" in lower or "cuda out of memory" in lower:
        return "A memória da GPU não foi suficiente. Reduza o lote/contexto ou escolha CPU nas Configurações. O progresso foi salvo quando possível."
    if "ram" in lower and ("limite" in lower or "pouca" in lower):
        return "O estudo ou treino parou para preservar memória do computador. Feche outros programas e tente novamente com configurações menores.\nDetalhe: " + text
    return text.split(": ", 1)[1] if text.startswith(("ValueError: ", "RuntimeError: ", "LearningDisabled: ", "Cancelled: ")) else text


def snapshot(core):
    counts = {r["status"]: r["n"] for r in core.db.query("SELECT status,count(*) n FROM queue GROUP BY status")}
    sources = core.db.query("SELECT count(*) n FROM sources")[0]["n"]
    chunks = core.db.query("SELECT count(*) n FROM chunks")[0]["n"]
    text_ready = bool(core.db.query("SELECT 1 FROM chunks WHERE length(text)>=40 LIMIT 1") or core.db.query("SELECT 1 FROM corrections WHERE validated=1 AND length(correction)>=40 LIMIT 1"))
    active = core.db.query("SELECT value FROM meta WHERE key='active_checkpoint'")
    pending, failed = counts.get("pendente", 0), counts.get("erro", 0)
    enabled = core.gate.enabled
    operation = core.operation
    if operation == "Estudar fila":
        state, title = "working", "ESTUDANDO AGORA"
        detail = "Acompanhe a etapa e o material na Central. Cada fonte só aparece na Biblioteca após ser salva."
    elif operation in ("Treinamento neural", "Retomar treinamento", "Experimento de reforço", "Praticar conversas"):
        if core.gate.paused.is_set():
            state, title, detail = "paused", "TREINAMENTO PAUSADO", "Clique em Continuar treinamento para retomar, ou Interromper e salvar."
        else:
            state, title, detail = "working", "TREINANDO A REDE NEURAL", "Os pesos estão sendo treinados. Consulte os passos e as métricas na aba Treinamento."
    elif operation != "Ociosa":
        state, title, detail = "working", "EM ANDAMENTO", operation
    elif pending:
        state, title = "waiting", f"{pending} MATERIAL(IS) AGUARDANDO INÍCIO"
        detail = "Clique em Estudar fila na Central." if enabled else "Clique em Ativar e estudar fila na Central para autorizar e começar."
    elif failed:
        state, title = "error", f"{failed} MATERIAL(IS) COM ERRO"
        detail = "O estudo desses itens não foi concluído. Veja a explicação na Central e escolha Tentar novamente."
    elif sources:
        state, title = "ready", "PRONTA PARA CONSULTAR"
        detail = f"{sources} fonte(s) disponível(is). Você já pode fazer perguntas no Chat documental, mesmo com aprendizado desligado."
    else:
        state, title = "empty", "NENHUM MATERIAL ESTUDADO AINDA"
        detail = "Adicione um documento ou site na Central e clique em Ativar e estudar fila."
    return {"counts": counts, "pending": pending, "failed": failed, "sources": sources,
            "chunks": chunks, "can_train": enabled and text_ready,
            "text_ready": text_ready, "active_checkpoint": active[0]["value"] if active else None,
            "state": state, "title": title, "detail": detail, "enabled": enabled}
