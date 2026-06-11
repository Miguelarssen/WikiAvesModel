import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from fpdf import FPDF
from datetime import datetime

class TrainingReport(FPDF):
    def __init__(self, model_name="Modelo"):
        super().__init__()
        self.model_name = model_name

    def header(self):
        # Barra superior estilizada
        self.set_fill_color(20, 40, 80)  # Azul escuro premium
        self.rect(0, 0, 210, 35, 'F')
        
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(255, 255, 255)
        self.set_y(10)
        self.cell(0, 10, "Relatorio de Treinamento - WikiAvesModel", 0, 1, "C")
        
        self.set_font("Helvetica", "", 10)
        self.cell(0, 8, f"Modelo: {self.model_name}", 0, 1, "C")
        
        # Resetar cor e posicionar abaixo do header
        self.set_text_color(0, 0, 0)
        self.set_y(40)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, f"Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | Pagina {self.page_no()}", 0, 0, "C")

def generate_report(save_dir, history, y_true, y_pred, class_names, training_time="N/A"):
    """
    Gera um relatorio PDF profissional com os resultados do treinamento.
    """
    model_name = os.path.basename(save_dir)
    print(f"[REPORT] Gerando relatorio para {model_name}...")
    
    # 1. Preparar caminhos para as imagens
    history_plot_path = os.path.join(save_dir, "report_history.png")
    cm_plot_path = os.path.join(save_dir, "report_cm.png")
    
    # 2. Plotar Historico (Loss e Acc)
    plt.style.use('seaborn-v0_8-muted')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Loss
    ax1.plot(history['train_loss'], label='Treino', linewidth=2)
    ax1.plot(history['val_loss'], label='Validação', linewidth=2)
    ax1.set_title('Histórico de Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Treino', linewidth=2)
    ax2.plot(history['val_acc'], label='Validação', linewidth=2)
    ax2.set_title('Histórico de Acurácia', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Acurácia (%)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(history_plot_path, dpi=150)
    plt.close()

    # 3. Plotar Matriz de Confusao (Usando TODAS as classes para consistência)
    all_labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusão Final (Teste)', fontsize=16, fontweight='bold')
    plt.ylabel('Classe Real', fontweight='bold')
    plt.xlabel('Classe Predita', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(cm_plot_path, dpi=150)
    plt.close()

    # 4. Calcular métricas globais
    acc = accuracy_score(y_true, y_pred) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    # 5. Criar PDF
    pdf = TrainingReport(model_name=model_name)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    
    # --- Seção 1: Resumo Executivo ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(20, 40, 80)
    pdf.cell(0, 10, "1. Resumo Executivo", 0, 1)
    pdf.ln(2)
    
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(0, 0, 0)
    
    # Tabela de métricas principais
    metrics_data = [
        ["Metrica", "Valor"],
        ["Total de Epocas", str(len(history['train_loss']))],
        ["Classes", str(len(class_names))],
        ["Tempo de Treinamento", training_time],
        ["Melhor Acc Validação", f"{max(history['val_acc']):.2f}%"],
        ["Acuracia no Teste", f"{acc:.2f}%"],
        ["Precision (Weighted)", f"{precision:.4f}"],
        ["Recall (Weighted)", f"{recall:.4f}"],
        ["F1-Score (Weighted)", f"{f1:.4f}"]
    ]
    
    with pdf.table(col_widths=(60, 40), text_align=("LEFT", "RIGHT")) as table:
        for row in metrics_data:
            row_cells = table.row()
            for cell in row:
                row_cells.cell(cell)
    
    pdf.ln(10)

    # --- Seção 2: Evolução do Treinamento ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(20, 40, 80)
    pdf.cell(0, 10, "2. Evolucao do Treinamento", 0, 1)
    pdf.image(history_plot_path, x=15, w=180)
    pdf.ln(5)
    
    # --- Seção 3: Matriz de Confusão ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(20, 40, 80)
    pdf.cell(0, 10, "3. Analise de Erros (Matriz de Confusao)", 0, 1)
    pdf.image(cm_plot_path, x=20, w=170)
    pdf.ln(10)

    # --- Seção 4: Estatísticas Detalhadas ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(20, 40, 80)
    pdf.cell(0, 10, "4. Estatisticas por Classe", 0, 1)
    pdf.ln(5)
    
    # Relatório de classificação formatado como tabela
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    
    table_data = [["Classe", "Precision", "Recall", "F1-Score", "Support"]]
    for cls_name, metrics in report_dict.items():
        if cls_name in class_names:
            table_data.append([
                cls_name,
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1-score']:.3f}",
                str(int(metrics['support']))
            ])
    
    pdf.set_font("Helvetica", "", 9)
    with pdf.table(col_widths=(50, 25, 25, 25, 25), text_align="CENTER") as table:
        for row in table_data:
            row_cells = table.row()
            for cell in row:
                row_cells.cell(cell)

    # Salvar PDF
    report_path = os.path.join(save_dir, "report.pdf")
    pdf.output(report_path)
    
    # Limpar imagens temporarias
    try:
        if os.path.exists(history_plot_path): os.remove(history_plot_path)
        # O usuario solicitou manter a matriz de confusao salva na pasta
        # if os.path.exists(cm_plot_path): os.remove(cm_plot_path)
    except:
        pass

    print("[OK] Relatorio salvo com sucesso: " + report_path)
