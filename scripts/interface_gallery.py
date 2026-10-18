"""Gera uma galeria local das capturas reais da interface, sem serviços externos."""
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / 'reports/interface_090'
entries = [
    ('inicio','Início','Onde cada coisa vai e os caminhos para começar.'),
    ('conversar','Conversar','Resposta real da rede já existente para “eae”. Nenhum novo treino.'),
    ('consultar_materiais','Consultar meus materiais','Consulta real ao artigo do Brasil já guardado. Sem nova pesquisa na internet.'),
    ('meus_materiais','Adicionar site','Importação, permissão e fila de materiais.'),
    ('adicionar_arquivo','Adicionar arquivo','Formatos aceitos e explicação do efeito de guardar materiais.'),
    ('materiais_guardados','Materiais guardados','A biblioteca existente preservada.'),
    ('melhorar_tac','Melhorar a TAC','Treinos separados e o cartão desativado de conversas de exemplo.'),
    ('resultado_treino','Resultado do treino','Apresentação do último treino de conversa registrado: rejeitado, mantendo a versão anterior.'),
    ('preferencias','Preferências','A mesma configuração da Wikipédia, aparência e cópias de segurança.'),
    ('como_comeco','Como começo?','Assistente com quatro passos e ações diretas.'),
    ('inicio_janela_menor','Janela menor','A tela Início em 880 × 640, com rolagem.'),
    ('respostas_guiadas','Respostas guiadas','Treino do classificador preservado nas ferramentas avançadas.'),
    ('treinamento_versoes','Treinamento e versões','Métricas, gráficos, retomada, restauração e experimento preservados.'),
    ('uso_computador','Uso do computador','Monitoramento e diagnóstico continuam disponíveis.'),
    ('ajustes','Ajustes e manutenção','Parâmetros e controles avançados preservados.'),
]
cards = []
for name,title,description in entries:
    image = folder/'capturas'/(name+'.png')
    if not image.is_file():
        raise FileNotFoundError(image)
    cards.append(f'<section id="{name}"><h2>{escape(title)}</h2><p>{escape(description)}</p>'
                 f'<a href="capturas/{name}.png"><img loading="lazy" src="capturas/{name}.png" alt="{escape(title)}"></a></section>')
nav = ' · '.join(f'<a href="#{name}">{escape(title)}</a>' for name,title,_ in entries)
html = ('<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>TAC — interface reorganizada</title><style>body{margin:0;background:#111827;color:#e5eaf4;font:16px Segoe UI,sans-serif}'
        'main{max-width:1240px;margin:auto;padding:24px}a{color:#82e4d3}nav{line-height:2}section{margin:40px 0}img{width:100%;height:auto;border:1px solid #344b65;border-radius:8px}'
        'p{color:#b7c7db}</style><main><h1>TAC — interface reorganizada</h1>'
        '<p>Capturas nativas do programa no Windows. Algoritmos, critérios de aprovação e modelos ativos preservados.</p>'
        f'<nav>{nav}</nav>' + ''.join(cards) + '</main></html>')
(folder/'galeria.html').write_text(html,encoding='utf-8')
print('Galeria criada com',len(entries),'capturas reais.')
