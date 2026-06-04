"""
Gera o documento PDF com Metodologia, Resultados Obtidos e Conclusão do TCC.
Formatação ABNT: fonte Times 12pt, margens 3cm sup, 2cm inf, 3cm esq, 2cm dir.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable
)
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

# ─── Configuração ABNT ────────────────────────────────────────────────────────
# ABNT NBR 14724: margens 3cm esq/sup, 2cm dir/inf
PAGE_WIDTH, PAGE_HEIGHT = A4
LEFT_MARGIN   = 3 * cm
RIGHT_MARGIN  = 2 * cm
TOP_MARGIN    = 3 * cm
BOTTOM_MARGIN = 2 * cm

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "Metodologia_Resultados_Conclusao_TCC.pdf")

# ─── Estilos ──────────────────────────────────────────────────────────────────
def build_styles():
    styles = {}

    # Corpo do texto – Times 12pt, espaçamento 1,5 (≈ 18pt), justificado
    styles["body"] = ParagraphStyle(
        "body",
        fontName="Times-Roman",
        fontSize=12,
        leading=18,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
        firstLineIndent=1.25 * cm,
    )

    # Corpo sem recuo (primeiro parágrafo após título)
    styles["body_no_indent"] = ParagraphStyle(
        "body_no_indent",
        fontName="Times-Roman",
        fontSize=12,
        leading=18,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )

    # Título da seção (1, 2, 3) – maiúsculas, negrito, 12pt
    styles["section"] = ParagraphStyle(
        "section",
        fontName="Times-Bold",
        fontSize=12,
        leading=18,
        alignment=TA_LEFT,
        spaceBefore=18,
        spaceAfter=6,
        textTransform="uppercase",
    )

    # Subseção (1.1) – negrito, 12pt
    styles["subsection"] = ParagraphStyle(
        "subsection",
        fontName="Times-Bold",
        fontSize=12,
        leading=18,
        alignment=TA_LEFT,
        spaceBefore=12,
        spaceAfter=4,
    )

    # Subsubseção (1.1.1) – itálico, 12pt
    styles["subsubsection"] = ParagraphStyle(
        "subsubsection",
        fontName="Times-Italic",
        fontSize=12,
        leading=18,
        alignment=TA_LEFT,
        spaceBefore=8,
        spaceAfter=4,
    )

    # Capa – título do trabalho
    styles["title"] = ParagraphStyle(
        "title",
        fontName="Times-Bold",
        fontSize=14,
        leading=21,
        alignment=TA_CENTER,
        spaceBefore=0,
        spaceAfter=12,
    )

    # Capa – subtítulo/info
    styles["subtitle"] = ParagraphStyle(
        "subtitle",
        fontName="Times-Roman",
        fontSize=12,
        leading=18,
        alignment=TA_CENTER,
        spaceBefore=6,
        spaceAfter=6,
    )

    # Legenda de tabela – 10pt, centralizado
    styles["caption"] = ParagraphStyle(
        "caption",
        fontName="Times-Italic",
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        spaceBefore=4,
        spaceAfter=8,
    )

    # Referências
    styles["ref"] = ParagraphStyle(
        "ref",
        fontName="Times-Roman",
        fontSize=12,
        leading=18,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
        leftIndent=0,
        firstLineIndent=0,
    )

    return styles


def H(n, title, styles):
    """Atalho para parágrafos de seção numerados."""
    level_map = {1: "section", 2: "subsection", 3: "subsubsection"}
    return Paragraph(f"{n} {title}", styles[level_map[len(n.split('.'))]])


def P(text, styles, indent=True):
    key = "body" if indent else "body_no_indent"
    return Paragraph(text, styles[key])


# ─── Conteúdo ─────────────────────────────────────────────────────────────────

def build_content(styles):
    story = []
    s = styles

    # ── CAPA ──────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph(
        "UNIVERSIDADE TECNOLÓGICA FEDERAL DO PARANÁ",
        s["subtitle"]
    ))
    story.append(Paragraph(
        "CURSO DE CIÊNCIA DA COMPUTAÇÃO",
        s["subtitle"]
    ))
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph(
        "UMA ARQUITETURA MLOPS PARA O CICLO DE VIDA DE MODELOS DE BIOACÚSTICA: "
        "UM ESTUDO DE CASO COM DADOS DO WIKIAVES",
        s["title"]
    ))
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph(
        "Seções: Metodologia · Resultados Obtidos · Conclusão",
        s["subtitle"]
    ))
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph(
        "Trabalho de Conclusão de Curso – versão final das seções revisadas",
        s["subtitle"]
    ))
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph("Curitiba, 2026", s["subtitle"]))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # 3. METODOLOGIA
    # ══════════════════════════════════════════════════════════════════════════
    story.append(H("3", "METODOLOGIA", s))
    story.append(P(
        "A metodologia deste trabalho foi estruturada em três eixos principais: "
        "(i) a criação da base de dados bioacústica a partir da plataforma WikiAves; "
        "(ii) o desenvolvimento e treinamento de modelos de Deep Learning para "
        "classificação de vocalização de aves; e (iii) a implementação de uma "
        "infraestrutura de MLOps para rastreamento, versionamento e experimentação "
        "dos modelos. As etapas foram conduzidas de forma incremental e interligada, "
        "refletindo o ciclo de vida de um sistema de aprendizado de máquina conforme "
        "descrito por Kreuzberger, Kühl e Hirschl (2022).",
        s, indent=False
    ))

    # 3.1 Base de dados
    story.append(H("3.1", "BASE DE DADOS", s))
    story.append(P(
        "A base de dados deste trabalho foi construída por meio de Web Scraping "
        "aplicado à plataforma WikiAves (<i>wikiaves.com.br</i>), repositório "
        "colaborativo que concentra registros sonoros de aves brasileiras com alto "
        "grau de confiabilidade, dado seu rigoroso processo de curadoria comunitária. "
        "O WikiAves disponibiliza, para cada espécie catalogada, metadados acessíveis "
        "via endpoints JSON, o que viabilizou a automação do processo de extração.",
        s
    ))
    story.append(P(
        "A seleção de espécies foi restrita à família Columbidae — que engloba "
        "pombas, juritis, avoante e espécies correlatas — com o critério de que "
        "cada espécie possuísse ao menos 200 gravações disponíveis na plataforma, "
        "garantindo um volume mínimo por classe para o treinamento supervisionado. "
        "Esse recorte taxonômico resultou em 27 espécies selecionadas.",
        s
    ))

    # 3.1.1
    story.append(H("3.1.1", "Extração dos dados", s))
    story.append(P(
        "O processo de extração foi implementado no módulo <i>scrapping/spiderSTFT.ipynb</i>, "
        "desenvolvido em Python com as bibliotecas <i>requests</i>, "
        "<i>BeautifulSoup</i> e <i>librosa</i>. O algoritmo consiste em dois estágios: "
        "(1) listagem das espécies elegíveis a partir da tabela de espécies da plataforma, "
        "que é analisada via parsing HTML e filtrada pelos critérios definidos; "
        "(2) download iterativo das gravações em formato MP3 para cada espécie "
        "selecionada, com tratamento de erros de acesso e de formatos não reconhecidos.",
        s
    ))
    story.append(P(
        "Após o download, cada arquivo de áudio foi submetido a uma etapa de "
        "pré-processamento: reamostrado para uma taxa de amostragem fixa de 44.100 Hz "
        "(TARGET_SR), convertido para mono e segmentado em janelas de 5 segundos "
        "centradas na região de maior energia sonora, identificada por meio da "
        "Energia Quadrática Média (<i>Root Mean Square Energy</i>, RMS). Esse "
        "critério garante que o trecho com maior atividade acústica seja preservado, "
        "eliminando regiões de silêncio ou ruído de fundo.",
        s
    ))

    # 3.1.2
    story.append(H("3.1.2", "Análise e preparação dos dados", s))
    story.append(P(
        "Cada segmento de 5 segundos foi transformado em espectrograma por meio da "
        "Transformada de Fourier de Curto Prazo (<i>Short-Time Fourier Transform</i>, "
        "STFT), com parâmetros fixos: n_fft = 2.048 e hop_length = 512, sem "
        "preenchimento implícito (center=False). O resultado foi convertido para "
        "escala logarítmica em decibéis (dB) e salvo como array NumPy (.npy), "
        "produzindo tensores de dimensão (1.025 × N_frames), em que N_frames "
        "corresponde ao número de quadros temporais do segmento.",
        s
    ))
    story.append(P(
        "Para cada espécie foram obtidas até 200 amostras, resultando em um "
        "conjunto de 10.794 amostras distribuídas entre as 27 classes. "
        "A base foi versionada com o DVC (<i>Data Version Control</i>), "
        "ferramenta que registra o hash MD5 do diretório de dados (identificado "
        "como <i>4ad1c1b</i> na versão utilizada nos experimentos), assegurando "
        "a reprodutibilidade dos treinamentos.",
        s
    ))

    # 3.2
    story.append(H("3.2", "ARQUITETURAS DOS MODELOS", s))
    story.append(P(
        "Com base na revisão de literatura realizada no Capítulo 2, que aponta "
        "as Redes Neurais Convolucionais (CNN) como a abordagem dominante em "
        "tarefas de classificação bioacústica e as Redes Convolucionais Recorrentes "
        "(CRNN) como alternativa superior para a captura de dependências temporais "
        "(Stowell, 2021), dois paradigmas arquiteturais foram investigados neste "
        "trabalho: arquiteturas puramente convolucionais e arquiteturas híbridas CRNN.",
        s
    ))
    story.append(P(
        "Todos os modelos utilizam como backbone o ConvNeXt-Tiny pré-treinado "
        "(He et al., 2015; Liu et al., 2022), com adaptação da primeira camada de "
        "convolução para aceitar entradas monocanal (1 canal), em vez dos 3 canais "
        "RGB originais. A adaptação é feita pela média dos pesos dos 3 canais, "
        "preservando os pesos pré-treinados de ImageNet e acelerando a convergência. "
        "A entrada padronizada para todos os modelos é um tensor de dimensão "
        "[Batch, 1, Freq, Tempo].",
        s
    ))

    # 3.2.1
    story.append(H("3.2.1", "CRNN_ConvNeXt_Tiny_BiLSTM", s))
    story.append(P(
        "O primeiro modelo investigado combina o backbone ConvNeXt-Tiny com uma "
        "camada recorrente bidirecional (BiLSTM). Após a extração de características "
        "pelo backbone (saída: 768 canais), um pooling global é aplicado na dimensão "
        "de frequência, produzindo uma sequência temporal [Batch, T_feat, 768] que "
        "serve de entrada para a BiLSTM com 2 camadas e 256 unidades ocultas em cada "
        "direção. O vetor de contexto é obtido por média temporal e classificado por "
        "uma camada linear com Dropout de 0,3.",
        s
    ))

    # 3.2.2
    story.append(H("3.2.2", "CRNN_ConvNeXt_Tiny_DeepConv_BiLSTM", s))
    story.append(P(
        "O segundo modelo estende o anterior com a inserção de um <i>neck</i> "
        "convolucional profundo entre o backbone e a BiLSTM. Esse neck é composto "
        "por três blocos residuais 2D (ConvResBlock) que reduzem progressivamente "
        "os canais de 768 para 256 (768 → 512 → 384 → 256), refinando as "
        "representações espaciais aprendidas pelo backbone antes da etapa recorrente. "
        "Cada bloco residual segue a estrutura Conv2d → BatchNorm → GELU → Conv2d "
        "→ BatchNorm com conexão residual projetada quando necessário.",
        s
    ))

    # 3.3
    story.append(H("3.3", "TREINAMENTO E CONFIGURAÇÕES EXPERIMENTAIS", s))
    story.append(P(
        "O treinamento dos modelos seguiu um protocolo experimental padronizado, "
        "descrito a seguir. O conjunto de dados foi dividido de forma estratificada "
        "em três subconjuntos: 80% para treinamento, 10% para validação e 10% para "
        "teste, com semente aleatória fixada em 42. A estratificação garante que a "
        "proporção de cada espécie seja mantida em todos os subconjuntos, evitando "
        "viés de seleção.",
        s
    ))

    # tabela de hiperparâmetros
    story.append(Spacer(1, 0.3 * cm))
    header = ["Hiperparâmetro", "Valor"]
    data = [
        header,
        ["Épocas máximas", "50"],
        ["Batch size", "32"],
        ["Otimizador", "Adam"],
        ["Taxa de aprendizagem", "0,0001"],
        ["Paciência (early stopping)", "10 épocas"],
        ["Delta mínimo (early stopping)", "0,0001"],
        ["Função de perda", "Cross-Entropy com Label Smoothing (α = 0,1)"],
        ["Augmentation (prob. por técnica)", "0,5"],
    ]
    table = Table(data, colWidths=[7 * cm, 8 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Times-Bold"),
        ("FONTNAME",     (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE",     (0, 0), (-1, -1), 11),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f3f4")]),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.grey),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    story.append(Paragraph(
        "Quadro 1 – Hiperparâmetros de treinamento utilizados nos experimentos. "
        "Fonte: autoria própria.",
        s["caption"]
    ))

    story.append(P(
        "A função de perda Cross-Entropy com Label Smoothing (α = 0,1) foi adotada "
        "para reduzir o overfitting em classes acusticamente similares, como as da "
        "família Columbidae, forçando o modelo a manter um grau de incerteza na "
        "distribuição de probabilidade de saída (Szegedy et al., 2016). "
        "O <i>early stopping</i> monitora a perda de validação e interrompe o "
        "treinamento quando não há melhora superior ao delta mínimo por 10 épocas "
        "consecutivas, prevenindo o sobreajuste.",
        s
    ))

    # 3.3.1 – Augmentation
    story.append(H("3.3.1", "Augmentação de Dados", s))
    story.append(P(
        "Para aumentar a diversidade das amostras de treinamento e melhorar a "
        "capacidade de generalização dos modelos, foi implementado um pipeline de "
        "augmentação aplicado exclusivamente ao subconjunto de treinamento. "
        "O pipeline aplica seis técnicas em sequência, cada uma com probabilidade "
        "independente de 0,5 de ser executada: (1) adição de ruído gaussiano; "
        "(2) deslocamento temporal (<i>time shift</i>); "
        "(3) máscara de frequência (<i>SpecAugment – frequency masking</i>); "
        "(4) máscara temporal (<i>SpecAugment – time masking</i>); "
        "(5) ajuste de ganho; e (6) simulação de <i>pitch shift</i> por deslocamento "
        "vertical das bandas de frequência. Essa abordagem simula variações naturais "
        "presentes em gravações de campo, reduzindo a sensibilidade do modelo a "
        "condições específicas de captura.",
        s
    ))

    # 3.4 – MLOps
    story.append(H("3.4", "INFRAESTRUTURA DE MLOPS", s))
    story.append(P(
        "A infraestrutura de MLOps foi implementada com foco no rastreamento de "
        "experimentos e no versionamento de dados e modelos, correspondendo às "
        "práticas do Nível 1 de Maturidade definido pelo Google Cloud (2024). "
        "Duas ferramentas principais foram integradas: o MLflow, para rastreamento "
        "de experimentos, e o DVC, para versionamento do dataset.",
        s
    ))
    story.append(P(
        "O módulo <i>mlops/mlflow_config.py</i> configura o servidor de rastreamento "
        "MLflow apontando para o diretório <i>mlruns/</i> local. A cada execução de "
        "treinamento, o sistema registra automaticamente: os hiperparâmetros do "
        "experimento, as métricas por época (perda e acurácia de treinamento e "
        "validação), as métricas finais no conjunto de teste (acurácia, precisão, "
        "recall e F1-Score), a versão do dataset (hash MD5 extraído do arquivo "
        "<i>dataset.dvc</i>), o commit Git correspondente ao código, e os artefatos "
        "do modelo treinado. Ao final de cada execução, o modelo é registrado "
        "automaticamente no MLflow Model Registry.",
        s
    ))
    story.append(P(
        "O versionamento do dataset foi realizado com o DVC, que rastreia alterações "
        "no diretório de dados por meio de checksums MD5. O arquivo "
        "<i>dataset.dvc</i> registra o hash do diretório e é versionado junto ao "
        "repositório Git, garantindo a reprodutibilidade dos experimentos ao "
        "associar cada execução de treinamento à versão exata do dataset utilizada.",
        s
    ))

    # ══════════════════════════════════════════════════════════════════════════
    # 4. RESULTADOS OBTIDOS
    # ══════════════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(H("4", "RESULTADOS OBTIDOS", s))
    story.append(P(
        "Esta seção apresenta os resultados dos experimentos de classificação de "
        "vocalizações de aves conduzidos neste trabalho, bem como uma avaliação "
        "da infraestrutura de MLOps implementada. Os experimentos foram realizados "
        "sobre o dataset de 10.794 espectrogramas STFT distribuídos em 27 espécies "
        "da família Columbidae, extraídos do WikiAves.",
        s, indent=False
    ))

    # 4.1
    story.append(H("4.1", "RESULTADOS DOS MODELOS DE CLASSIFICAÇÃO", s))
    story.append(P(
        "Dois modelos foram levados a resultados completos com relatório de teste: "
        "o CRNN_ConvNeXt_Tiny_BiLSTM e o CRNN_ConvNeXt_Tiny_DeepConv_BiLSTM. "
        "A Tabela 1 apresenta um comparativo das métricas obtidas no conjunto de "
        "teste para ambas as arquiteturas.",
        s
    ))

    # tabela comparativa
    header2 = ["Métrica", "CRNN-BiLSTM", "CRNN-DeepConv-BiLSTM"]
    data2 = [
        header2,
        ["Épocas treinadas (early stopping)",    "27", "25"],
        ["Melhor Acc. de Validação",              "85,08%", "86,10%"],
        ["Acurácia no Teste",                     "85,37%", "85,37%"],
        ["Precisão Ponderada (Weighted)",         "0,8592", "0,8656"],
        ["Recall Ponderado (Weighted)",           "0,8537", "0,8537"],
        ["F1-Score Ponderado (Weighted)",         "0,8542", "0,8567"],
        ["Tempo de Treinamento",                  "01:18:11", "01:15:40"],
        ["Classes (espécies)",                    "27", "27"],
    ]
    table2 = Table(data2, colWidths=[6.5 * cm, 4 * cm, 5 * cm])
    table2.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#1a5276")),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Times-Bold"),
        ("FONTNAME",     (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE",     (0, 0), (-1, -1), 11),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eaf4fb")]),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.grey),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("ALIGN",        (1, 0), (-1, -1), "CENTER"),
    ]))
    story.append(table2)
    story.append(Paragraph(
        "Tabela 1 – Comparativo de desempenho entre as arquiteturas avaliadas no "
        "conjunto de teste (27 classes, 1.080 amostras). Fonte: autoria própria.",
        s["caption"]
    ))

    story.append(P(
        "Ambas as arquiteturas atingiram acurácia de 85,37% no conjunto de teste. "
        "O modelo CRNN_ConvNeXt_Tiny_DeepConv_BiLSTM apresentou marginal superioridade "
        "em precisão ponderada (0,8656 contra 0,8592) e F1-Score ponderado "
        "(0,8567 contra 0,8542), sugerindo que o <i>neck</i> convolucional profundo "
        "contribui para uma distribuição de acertos mais equilibrada entre as classes, "
        "a despeito de não alterar a acurácia global. Ambos os modelos convergiram "
        "rapidamente, ativando o <i>early stopping</i> antes da 30ª época, com "
        "tempo total de treinamento inferior a 1 hora e 20 minutos em GPU.",
        s
    ))

    # 4.2 análise de erros
    story.append(H("4.2", "ANÁLISE POR CLASSE", s))
    story.append(P(
        "A análise das estatísticas por classe no modelo CRNN_ConvNeXt_Tiny_DeepConv_BiLSTM "
        "revela uma distribuição heterogênea de desempenho. As classes que atingiram "
        "F1-Score perfeito ou próximo de 1,0 incluem a classe 24 (F1 = 0,962) e "
        "as classes 1 e 11 (F1 = 0,950), indicando alta discriminabilidade acústica "
        "dessas espécies. Em contraste, as classes 18 (F1 = 0,653) e 6 (F1 = 0,700) "
        "apresentaram os maiores índices de confusão, o que pode ser atribuído à "
        "semelhança acústica entre espécies da família Columbidae, reconhecida na "
        "literatura como um dos principais desafios para sistemas de classificação "
        "bioacústica (Sanchez et al., 2021).",
        s
    ))
    story.append(P(
        "Destaca-se que o F1-Score mínimo obtido (0,653) é superior ao registrado "
        "por Sanchez et al. (2021) — que reportaram acurácia máxima de 75% em "
        "espécies de aves na base NIPS4Bplus — demonstrando que o uso do backbone "
        "ConvNeXt-Tiny pré-treinado proporciona vantagem significativa em relação a "
        "arquiteturas treinadas do zero sobre bases de menor escala.",
        s
    ))

    # 4.3 mlops
    story.append(H("4.3", "INFRAESTRUTURA DE MLOPS IMPLEMENTADA", s))
    story.append(P(
        "A infraestrutura de MLOps foi avaliada qualitativamente segundo os níveis "
        "de maturidade definidos pelo Google Cloud (2024). O sistema implementado "
        "atinge o Nível 1 de Maturidade, caracterizado pelo rastreamento sistemático "
        "de experimentos e pelo versionamento de dados e modelos, mas ainda sem a "
        "automação completa do pipeline de CI/CD/CT.",
        s
    ))
    story.append(P(
        "As funcionalidades efetivamente implementadas e validadas durante os "
        "experimentos foram: (1) rastreamento automático de hiperparâmetros e "
        "métricas por época via MLflow; (2) associação biunívoca entre versão do "
        "dataset (hash DVC) e execução de treinamento; (3) versionamento do código "
        "via tag Git associada ao run MLflow; (4) geração automática de relatório "
        "em PDF com curvas de aprendizado, matriz de confusão e estatísticas por "
        "classe; e (5) registro automático do modelo treinado no MLflow Model "
        "Registry com versionamento incremental.",
        s
    ))
    story.append(P(
        "As funcionalidades previstas no escopo original do Nível 2 — incluindo "
        "o monitoramento contínuo de drift em produção, o acionamento automático "
        "de re-treinamento via trigger de novos dados do WikiAves e o pipeline "
        "completo de CI/CD — não foram implementadas no prazo deste trabalho, "
        "constituindo uma limitação explícita do escopo executado, conforme "
        "discutido na Seção 5.",
        s
    ))

    # 4.4 comparação
    story.append(H("4.4", "COMPARAÇÃO COM TRABALHOS RELACIONADOS", s))

    header3 = ["Trabalho", "Dataset", "Classes", "Acc. / F1"]
    data3 = [
        header3,
        ["Sanchez et al. (2021)", "NIPS4Bplus",     "87",  "≤ 75% (aves)"],
        ["Este trabalho – CRNN-BiLSTM",
         "WikiAves (Columbidae)", "27",  "85,37% / 0,8542"],
        ["Este trabalho – CRNN-DeepConv-BiLSTM",
         "WikiAves (Columbidae)", "27",  "85,37% / 0,8567"],
    ]
    table3 = Table(data3, colWidths=[5 * cm, 4.5 * cm, 2 * cm, 4 * cm])
    table3.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#1a5276")),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Times-Bold"),
        ("FONTNAME",     (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE",     (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eaf4fb")]),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.grey),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("ALIGN",        (2, 0), (-1, -1), "CENTER"),
    ]))
    story.append(table3)
    story.append(Paragraph(
        "Tabela 2 – Comparativo com trabalho relacionado de classificação bioacústica "
        "de aves. Fonte: autoria própria com base em Sanchez et al. (2021).",
        s["caption"]
    ))

    # ══════════════════════════════════════════════════════════════════════════
    # 5. CONCLUSÃO
    # ══════════════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(H("5", "CONCLUSÃO", s))

    story.append(P(
        "Este trabalho apresentou o desenvolvimento de um sistema de classificação "
        "de vocalizações de aves brasileiras da família Columbidae, com suporte a "
        "uma infraestrutura de MLOps para rastreamento de experimentos e "
        "versionamento de dados e modelos. O escopo original propunha a "
        "implementação de uma arquitetura de Nível 2 de Maturidade MLOps — com "
        "monitoramento de drift em produção, pipeline de CI/CD e re-treinamento "
        "automático —, contudo, o escopo foi redimensionado durante a execução do "
        "trabalho, em alinhamento com a orientação da banca avaliadora, que indicou "
        "a necessidade de delimitar o objeto de estudo para garantir a profundidade "
        "e a qualidade científica dos resultados apresentados.",
        s, indent=False
    ))
    story.append(P(
        "O sistema desenvolvido operacionaliza as seguintes contribuições concretas: "
        "(1) a criação de uma base de dados bioacústica de 10.794 espectrogramas "
        "STFT de 27 espécies da família Columbidae, extraídos do WikiAves, "
        "constituindo um recurso inédito para a avifauna brasileira nesse nível "
        "de especificidade taxonômica; (2) a proposição e avaliação de duas "
        "arquiteturas de Deep Learning baseadas em ConvNeXt-Tiny, integrando "
        "mecanismos recorrentes (BiLSTM) e convolucionais profundos como <i>neck</i>, "
        "que alcançaram acurácia de 85,37% e F1-Score ponderado de até 0,857 em "
        "um cenário com 27 classes; e (3) a implementação de uma infraestrutura "
        "de MLOps de Nível 1 de Maturidade, que associa automaticamente cada "
        "experimento à versão exata do dataset e do código, promovendo "
        "reprodutibilidade e rastreabilidade sistemáticas.",
        s
    ))
    story.append(P(
        "Os resultados demonstram a viabilidade do uso de modelos de Deep Learning "
        "com transfer learning para a classificação de vocalizações de aves "
        "brasileiras a partir de dados de plataformas colaborativas. O desempenho "
        "obtido supera o reportado por Sanchez et al. (2021) para a classificação "
        "de aves na base NIPS4Bplus (≤ 75%), reforçando que o uso de backbones "
        "pré-treinados e técnicas modernas de augmentação são escolhas metodológicas "
        "decisivas para esse domínio.",
        s
    ))

    # Limitações
    story.append(H("5.1", "LIMITAÇÕES DO TRABALHO", s))
    story.append(P(
        "As principais limitações identificadas neste trabalho são: (i) o escopo "
        "taxonômico restrito a uma única família (Columbidae), com 27 classes, "
        "o que, embora adequado para a prova de conceito, não representa a "
        "diversidade da avifauna brasileira; (ii) a ausência de implementação "
        "do monitoramento de drift em produção e do re-treinamento automatizado, "
        "que constituíam objetivos centrais do escopo original; (iii) a limitação "
        "de 200 amostras por espécie imposta pela viabilidade de scraping no "
        "período da pesquisa, o que pode subestimar o potencial do método em "
        "cenários de maior escala; e (iv) a ausência de avaliação em dados "
        "externos ao WikiAves, o que impede generalizações sobre o comportamento "
        "do modelo em gravações de campo com condições acústicas distintas.",
        s
    ))

    # Trabalhos futuros
    story.append(H("5.2", "TRABALHOS FUTUROS", s))
    story.append(P(
        "Os resultados obtidos e as limitações identificadas apontam para um "
        "conjunto de direções de pesquisa relevantes, especialmente para "
        "continuação em nível de pós-graduação:",
        s
    ))
    story.append(P(
        "<b>Ampliação do escopo e escalabilidade da base de dados.</b> "
        "Uma extensão natural deste trabalho é a construção de uma base de dados "
        "abrangendo múltiplas famílias taxonômicas da avifauna brasileira — "
        "seguindo a tendência de modelos de larga escala como o Perch 2.0 "
        "(Merriënboer et al., 2025), treinado em mais de 1,5 milhão de gravações "
        "e 10.906 classes. A automação completa do pipeline de extração e "
        "processamento, com monitoramento contínuo da plataforma WikiAves, "
        "constituiria um sistema de aquisição de dados contínua para alimentar "
        "futuros modelos.",
        s
    ))
    story.append(P(
        "<b>Implementação da arquitetura MLOps de Nível 2.</b> "
        "O objetivo não alcançado neste trabalho — a implementação de um pipeline "
        "com CI/CD/CT e monitoramento de drift — representa uma agenda de pesquisa "
        "aplicada de alto valor. A integração de ferramentas como Evidently AI "
        "para detecção de data drift, Airflow ou Prefect para orquestração de "
        "pipelines, e Kubernetes para deploy escalável constituiria uma contribuição "
        "significativa para a área de MLOps aplicada à bioacústica.",
        s
    ))
    story.append(P(
        "<b>Investigação de arquiteturas baseadas em transformers.</b> "
        "Modelos como o Audio Spectrogram Transformer (AST) e variantes do "
        "Vision Transformer (ViT) aplicados a espectrogramas têm demonstrado "
        "desempenho superior a CRNNs em tarefas de classificação de áudio "
        "em benchmarks recentes. A comparação sistemática dessas arquiteturas "
        "com os modelos propostos neste trabalho, especialmente em cenários "
        "de poucos dados por classe (<i>few-shot learning</i>), representa "
        "uma lacuna relevante na literatura de bioacústica computacional.",
        s
    ))
    story.append(P(
        "<b>Avaliação em dados de monitoramento acústico passivo (PAM).</b> "
        "A validação dos modelos desenvolvidos em gravações obtidas por dispositivos "
        "de monitoramento acústico passivo implantados em campo — conforme descrito "
        "por Darras et al. (2019) — permitiria avaliar a capacidade de generalização "
        "real dos sistemas e sua viabilidade como ferramenta de conservação "
        "ambiental. Esse tipo de avaliação constitui o passo fundamental para "
        "a transição de um protótipo de pesquisa para uma aplicação operacional.",
        s
    ))
    story.append(P(
        "<b>Modelos multiclasse e multi-espécie.</b> "
        "O presente trabalho trata a classificação como um problema multiclasse "
        "(uma espécie por amostra). Cenários realistas de campo, porém, frequentemente "
        "envolvem vocalizações simultâneas de múltiplas espécies, exigindo abordagens "
        "de detecção multi-label. A adaptação do pipeline de dados e das arquiteturas "
        "para suportar esse cenário constitui uma linha de pesquisa relevante para "
        "aplicações de monitoramento de biodiversidade em larga escala.",
        s
    ))
    story.append(P(
        "Em síntese, este trabalho estabelece uma base metodológica e experimental "
        "sólida para pesquisas em bioacústica computacional aplicada à avifauna "
        "brasileira, com rastreabilidade assegurada pela infraestrutura de MLOps "
        "implementada. As contribuições aqui apresentadas, embora de escopo reduzido "
        "em relação ao projeto original, constituem um ponto de partida robusto "
        "para investigações de maior amplitude e complexidade, especialmente no "
        "contexto do mestrado em computação.",
        s
    ))

    # ══════════════════════════════════════════════════════════════════════════
    # REFERÊNCIAS (ABNT)
    # ══════════════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("REFERÊNCIAS", s["section"]))
    story.append(Spacer(1, 0.3 * cm))

    refs = [
        ("DARRAS, K. et al.",
         "Autonomous sound recording outperforms human observation for sampling birds: "
         "a systematic map and user guide. <i>Ecological Applications</i>, v. 29, n. 6, "
         "e01954, 2019."),
        ("DVC.",
         "<i>DVC — Data Version Control</i>. Disponível em: https://dvc.org. "
         "Acesso em: 24 nov. 2025."),
        ("GASC, A. et al.",
         "Assessing biodiversity with sound: Do acoustic diversity indices reflect "
         "phylogenetic and functional diversities of bird communities? "
         "<i>Ecological Indicators</i>, v. 25, p. 279–287, fev. 2013."),
        ("GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A.",
         "<i>Deep Learning</i>. Cambridge: MIT Press, 2016."),
        ("GOOGLE CLOUD.",
         "MLOps: Continuous delivery and automation pipelines in machine learning. "
         "<i>Google Cloud Architecture Center</i>. Disponível em: "
         "https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation"
         "-pipelines-in-machine-learning. Acesso em: 21 nov. 2025."),
        ("HE, K. et al.",
         "Deep Residual Learning for Image Recognition. In: "
         "<i>Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition "
         "(CVPR)</i>, 2016. p. 770–778."),
        ("KREUZBERGER, D.; KÜHL, N.; HIRSCHL, S.",
         "Machine Learning Operations (MLOps): Overview, Definition, and Architecture. "
         "<i>arXiv</i>, v. 3, 2022."),
        ("LIU, Z. et al.",
         "A ConvNet for the 2020s. In: <i>Proceedings of the IEEE/CVF Conference on "
         "Computer Vision and Pattern Recognition (CVPR)</i>, 2022. p. 11966–11976."),
        ("MERRIËNBOER, B. V. et al.",
         "Perch 2.0: The Bittern Lesson for Bioacoustics. <i>arXiv</i>, v. 1, ago. 2025."),
        ("MITCHELL, T. M.",
         "<i>Machine Learning</i>. New York: McGraw-Hill, 1997."),
        ("MLFLOW.",
         "<i>MLflow — An open source platform for the machine learning lifecycle</i>. "
         "Disponível em: https://mlflow.org. Acesso em: 24 nov. 2025."),
        ("MORFI, V. et al.",
         "NIPS4Bplus: a richly annotated birdsong audio dataset. <i>arXiv</i>, 2018. "
         "Disponível em: https://arxiv.org/abs/1811.02275."),
        ("RUSSELL, S. J.; NORVIG, P.",
         "<i>Artificial Intelligence: A Modern Approach</i>. 4. ed. New Jersey: Pearson, 2021."),
        ("SANCHEZ, F. J. B. et al.",
         "Bioacoustic classification of avian calls from raw sound waveforms with an "
         "open-source deep learning architecture. <i>Scientific Reports</i>, v. 11, "
         "n. 15733, p. 1–12, 2021."),
        ("SCULLEY, D. et al.",
         "Hidden Technical Debt in Machine Learning Systems. In: "
         "<i>Proceedings of the 28th Annual Conference on Neural Information Processing "
         "Systems (NeurIPS)</i>, 2015."),
        ("STOWELL, D.",
         "Computational bioacoustics with deep learning: a review and roadmap. "
         "<i>PeerJ</i>, Tilburg, v. 10, dez. 2021."),
        ("SZEGEDY, C. et al.",
         "Rethinking the Inception Architecture for Computer Vision. In: "
         "<i>Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition "
         "(CVPR)</i>, 2016. p. 2818–2826."),
        ("VILLEMEZ, N. R.",
         "Machine Learning Operations (MLOPS) Architecture Considerations for Deep "
         "Learning with a Passive Acoustic Vector Sensor. Dissertação (Mestrado) — "
         "Naval Postgraduate School, 2021."),
        ("WIKIAVES.",
         "<i>WikiAves — A enciclopédia das aves do Brasil</i>. Disponível em: "
         "https://www.wikiaves.com.br. Acesso em: 23 out. 2025."),
    ]

    for authors, text in refs:
        p_text = f"<b>{authors}</b> {text}"
        story.append(Paragraph(p_text, s["ref"]))
        story.append(Spacer(1, 4))

    return story


# ─── Numeração de páginas ─────────────────────────────────────────────────────

def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 12)
    page_num = canvas.getPageNumber()
    canvas.drawRightString(
        PAGE_WIDTH - RIGHT_MARGIN,
        PAGE_HEIGHT - TOP_MARGIN + 0.5 * cm,
        str(page_num)
    )
    canvas.restoreState()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=A4,
        leftMargin=LEFT_MARGIN,
        rightMargin=RIGHT_MARGIN,
        topMargin=TOP_MARGIN,
        bottomMargin=BOTTOM_MARGIN,
        title="Metodologia, Resultados e Conclusão – TCC WikiAves",
        author="Miguel"
    )

    styles = build_styles()
    story  = build_content(styles)

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"[OK] PDF gerado em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
