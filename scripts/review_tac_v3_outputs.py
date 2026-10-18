"""Record the agent's completed qualitative reading; never alter raw outputs."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tac.config import atomic_json

OUT = ROOT / 'reports/tac_v3/rodada_20261018_01'


def main():
    baseline = json.loads((OUT / 'REVISAO_V2_BASELINES.json').read_text(encoding='utf-8'))
    models = baseline['models']
    # These annotations follow inspection of every complete output and session.
    # They are evidence records, not an inference-time response mechanism.
    common = {
        3: 'Interpreta produto como mercadoria; não calcula 6 vezes 4 = 24.',
        4: 'Responde 5; 20 dividido por 5 é 4.',
        7: 'Não recupera Talvo como nome atual; inventa análise e entidades sem suporte.',
        8: 'Não recupera Néris como nome anterior; a menção lexical não resolve o papel temporal.',
        10: 'Não fornece Luzia, Néris e Talvo na sequência; produz análise desconexa.',
        11: 'Fala sobre ameixa, sem associar a preferência à pessoa Cora.',
        12: 'Retorna à cor lilás e ignora que a última escolha foi verde.',
        13: 'Copia Arvélio com acento, mas acrescenta texto apesar do pedido de somente o nome. Acerto factual parcial separado do seguimento integral.',
        14: 'Não explica a rotação da Terra; associa dia/noite a verão ou produz fragmento.',
        15: 'Não explica luz, água e dióxido de carbono; tautologia ou afirmações sem suporte.',
        16: 'Responde sobre a criança em vez de descrever o lobo.',
        17: 'Não explica transferência de calor/fusão; inclui afirmações contraditórias.',
        18: 'Não lista três frutas; muda para dieta ou análise de nomes próprios.',
        19: 'Repete a oração no singular; não executa a transformação para o plural.',
        20: 'Não escreve duas frases sobre cuidar das árvores; muda para família/animais.',
        21: 'A feira foi adiada é uma afirmação correta do texto, mas omite chuva e praça e não preserva a relação causal central do resumo esperado. Crédito parcial factual, sem cumprimento integral.',
        22: 'Não conta uma história de três frases sobre um barco de papel; definição tautológica ou fragmento.',
        23: 'Não escreve bilhete de agradecimento pela ajuda do vizinho; há desvio ou texto desconexo.',
        24: 'Não cria diálogo entre duas amigas; produz instruções ou análise de palavras.',
        25: 'Trata cumprimento como tarefa de análise/tradução e inventa contexto; não conversa.',
        26: 'Não responde ao agradecimento; muda para componentes de frases.',
        27: 'Não pede a explicação ausente nem esclarece; produz análise sem referente.',
        28: 'Cumprimento curto e pertinente; Apa é erro de grafia, sem impedir o sentido. Mesmo tratamento tolerante dado ao Opaa da v2.',
        29: 'Não define lobo como animal canídeo; tautologia ou descrição falsa de alimento.',
        30: 'Não calcula 2 + 2 = 4; apresenta outra conta ou texto sem relação.',
        31: 'Não entrega narrativa coerente; repete a instrução e descreve genericamente uma história.',
        32: 'Recupera Carlos corretamente; construção gramatical ruim não torna o valor ambíguo.',
    }
    for label in ('v3_linguagem', 'v3_epoca1', 'v3_final'):
        raw = json.loads((OUT / ('geracoes_' + label + '.json')).read_text(encoding='utf-8'))
        accepted = {9, 28, 32} if label == 'v3_epoca1' else ({28, 32} if label == 'v3_final' else set())
        rows = []
        for row in raw['outputs']:
            i = row['id']
            if label == 'v3_linguagem':
                diagnosis = ('Continua o texto como página web, com títulos/propaganda ou conteúdo inventado, sem responder à solicitação.'
                             if i not in (7, 8, 9, 10, 19, 29) else
                             'Repete Entendi, a pergunta ou a frase original sem recuperar a informação nem executar a tarefa.')
                if i == 11:
                    diagnosis = 'Menciona Cora, mas a continuação é uma página comercial incoerente; não é resposta factual inequívoca à preferência.'
                if i == 21:
                    diagnosis = 'Inventa eventos e contradições meteorológicas; não resume o texto em uma frase.'
            elif i == 1:
                diagnosis = ('Responde 47; sete mais oito é 15.' if label == 'v3_epoca1' else
                             'Não calcula 15; inventa análise de revisor, calendário e casa.')
            elif i == 2:
                diagnosis = ('Responde 47; 18 menos 7 é 11.' if label == 'v3_epoca1' else
                             'Repete a operação em uma análise, sem dar 11.')
            elif i == 5:
                diagnosis = ('Descreve produto/favoritos; não calcula 14.' if label == 'v3_epoca1' else
                             'Produz um fragmento Comece com o uso, sem calcular 14.')
            elif i == 6:
                diagnosis = ('Responde 47; 8 menos 3 é 5.' if label == 'v3_epoca1' else
                             'Produz instruções metalinguísticas, sem calcular 5.')
            elif i == 9:
                diagnosis = ('Recupera Luzia como primeiro nome de forma inequívoca.' if label == 'v3_epoca1' else
                             'Menciona Luzia em análise desconexa, sem responder inequivocamente qual foi o primeiro nome; regressão em relação à primeira passagem.')
            else:
                diagnosis = common[i]
            annotated = {'id': i, 'appropriate': i in accepted, 'diagnosis': diagnosis}
            if label != 'v3_linguagem' and i in (13, 21):
                annotated.update(partial_factual_credit=True,
                                 task_factual_value_correct=(i == 13),
                                 exact_name_characters=(i == 13))
            if i in (7, 8, 9, 10, 11, 12, 13):
                annotated['context_task_factual_correct'] = (i == 9 and label == 'v3_epoca1') or (i == 13 and label != 'v3_linguagem')
            rows.append(annotated)
        models[label] = {
            'rows': rows,
            'main_appropriate': sum(r['appropriate'] for r in rows if r['id'] <= 27),
            'self_generated_session_diagnosis': 'Após mudar de animais para matemática/nome, as respostas próprias anteriores contaminam a continuação; não recupera Safira. Falha observada na rede com o histórico próprio fornecido, sem evidência de mistura de sessões da interface.'
        }
    review = {
        'reviewer': 'Inspeção qualitativa do agente Codex, sem certificação humana independente e sem chamada a avaliador/modelo externo',
        'original_outputs_modified': False,
        'earlier_reports_or_grades_changed': False,
        'criteria': [
            'Adequada significa resposta factual inequívoca e cumprimento da tarefa solicitada, sem afirmações centrais contraditórias.',
            'Redação diferente e erros menores de gramática/grafia são aceitos quando não mudam o significado.',
            'Cópia correta de Arvélio é registrada como acerto factual parcial, embora texto adicional viole somente o nome.',
            'Resumo que menciona apenas o adiamento da feira recebe crédito factual parcial e falha por omitir a relação central esperada; nenhuma pergunta ou resposta esperada foi alterada.',
            'Mencionar um nome dentro de uma análise incoerente não basta para recuperar seu papel temporal.'
        ],
        'models': models,
        'selected_label': 'v3_epoca1',
        'selection_reason': 'Disponibilidade apenas experimental: a primeira passagem preserva o acerto do primeiro nome (Luzia), perdido no final. Não há ganho consistente sobre a v2 e nenhum candidato é aprovado. A menor loss final não justifica preferi-lo à regressão observada na geração livre.',
        'conclusion': 'O treinamento real da TAC v3 foi concluído dentro do orçamento, mas o objetivo de conversação coerente não foi atingido. A linguagem tem menor surpresa por byte nas passagens inéditas que a v2, porém as instruções continuam inadequadas: 1/27 tarefas integralmente atendidas após a primeira passagem e 0/27 no final, contra 2/27 da v2 de conversação. Há acertos factuais parciais de cópia e resumo, registrados separadamente. A v3 permanece experimental, sem substituir a v2.',
        'limitations': 'Amostra pequena de desenvolvimento, uma seed de treino e uma configuração de geração; revisão pelo agente, sem avaliação humana independente. O limite comum de 128 tokens permite extensões diferentes entre tokenizers e corta algumas respostas. Nenhuma entrada desses casos sofreu truncamento. Os testes de software e a queda da loss não certificam compreensão. Arquitetura, tokenizer e corpus mudaram juntos; os resultados não isolam a causa. Dados públicos filtrados ainda contêm ruído e estilo de análise prolixa. A presença de texto comercial nas gerações e de respostas que imitam análise sem resolver a pergunta sustenta investigar a adequação dos alvos, mas não prova uma causa única. Sessões com respostas próprias também mantêm erros anteriores. Licenças são as declaradas pelas fontes e não validam todos os direitos individuais dos textos.',
        'recommendation': 'Único próximo experimento recomendado, sem execução automática: comparar a partir do mesmo checkpoint de linguagem dois ajustes supervisionados com orçamento equivalente de tokens — corpus atual versus subconjunto revisado de respostas diretas e explicativas, com famílias novas reservadas para avaliação. A hipótese é que a distribuição e a qualidade dos alvos favorecem imitar análises genéricas em vez de executar a solicitação. Preservar ambos os controles, o contexto, a arquitetura e o orçamento; avaliar geração livre e retenção antes de qualquer promoção.'
    }
    target = OUT / 'REVISAO_GERACOES.json'
    if target.exists():
        raise RuntimeError('A revisão existente deve ser preservada')
    atomic_json(target, review)
    print({label: model['main_appropriate'] for label, model in models.items()})


if __name__ == '__main__':
    main()
