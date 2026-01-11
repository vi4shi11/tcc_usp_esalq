"""
TCC: Otimização da Inteligência de Negócios no E-commerce Brasileiro
Comparação do Modelo Fine-Tuned vs Baselines

Autor: Vinicius Barreiro Shishido
Orientador: Felipe Pinto Da Silva
Curso: MBA em Data Science e Analytics - USP ESALQ

Descrição:
    Este script compara o modelo ABSA fine-tuned com baselines:
    1. Baseline Review Score: usa nota 1-5 + regex
    2. Baseline BERTimbau Base: mesmo modelo sem fine-tuning
    3. Fine-Tuned ABSA: modelo treinado para a tarefa
    
Uso:
    python baseline_comparison.py --csv_rotulado ../dados/rotulacao_reviews.csv \
                                  --csv_balanceado ../dados/amostra_rotulagem_balanceada.csv \
                                  --modelo_path ../modelo/modelo_absa
"""

import os
import pandas as pd

# Diretório base do projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS_DIR = os.path.join(BASE_DIR, 'dados')
MODELO_DIR = os.path.join(BASE_DIR, 'modelo')
RESULTADOS_DIR = os.path.join(BASE_DIR, 'resultados')
import numpy as np
import torch
import torch.nn.functional as F
import re
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================

LABELS = [
    "Logística_Pos", "Logística_Neg",
    "Produto_Pos",   "Produto_Neg",
    "Atend_Pos",     "Atend_Neg",
    "Preço_Pos",     "Preço_Neg",
]

THRESHOLDS = {
    "Logística_Pos": 0.50, "Logística_Neg": 0.50,
    "Produto_Pos": 0.55,   "Produto_Neg": 0.45,
    "Atend_Pos": 0.30,     "Atend_Neg": 0.45,
    "Preço_Pos": 0.25,     "Preço_Neg": 0.30,
}

ASPECT_PATTERNS = {
    'logistica': re.compile(
        r'\b(entreg\w*|prazo|atraso\w*|atras\w*|demor\w*|cheg\w*|'
        r'frete|transport\w*|correio\w*|envi\w*|rastreio|'
        r'embalag\w*|caixa|pacote|extravi\w*)\b', re.IGNORECASE),
    'produto': re.compile(
        r'\b(produto|qualidade|material|acabamento|funciona\w*|defeito\w*|'
        r'quebr\w*|estrag\w*|danific\w*|original|falso|'
        r'tamanho|cor|modelo|foto|imagem|descri\w*)\b', re.IGNORECASE),
    'atendimento': re.compile(
        r'\b(vendedor\w*|loja|atendimento|atend\w*|resposta|respond\w*|'
        r'contato|comunica\w*|suporte|ajuda|solução|resolver|'
        r'educad\w*|grosseir\w*|ignor\w*|descaso)\b', re.IGNORECASE),
    'preco': re.compile(
        r'\b(preço|preco|valor|caro|barat\w*|custo|benefício|beneficio|'
        r'pag\w*|cobr\w*|desconto|promoção|promocao|oferta)\b', re.IGNORECASE)
}

BERTIMBAU_BASE = "neuralmind/bert-base-portuguese-cased"

# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

def criar_label(row):
    """Converte colunas de sentimento para vetor one-hot"""
    vetor = [0.0] * len(LABELS)
    mapa = {
        'logistica_sentimento':   (0, 1),
        'produto_sentimento':     (2, 3),
        'atendimento_sentimento': (4, 5),
        'preco_sentimento':       (6, 7),
    }
    for col, (pos, neg) in mapa.items():
        if col in row.index:
            val = str(row[col]).strip().lower() if pd.notna(row[col]) else ''
            if val == 'positivo':
                vetor[pos] = 1.0
            elif val == 'negativo':
                vetor[neg] = 1.0
    return vetor

def detectar_aspectos(text):
    """Detecta aspectos mencionados no texto via regex"""
    aspectos = []
    for aspecto, pattern in ASPECT_PATTERNS.items():
        if pattern.search(str(text)):
            aspectos.append(aspecto)
    return aspectos if aspectos else ['produto']

def calc_metrics(y_true, y_pred, name):
    """Calcula métricas de avaliação"""
    return {
        'Modelo': name,
        'F1-Micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'F1-Macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'F1-Weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'Exact Match': accuracy_score(y_true, y_pred),
    }

# ==============================================================================
# BASELINES
# ==============================================================================

def baseline_review_score(score, text):
    """
    Baseline 1: Usa nota 1-5 + regex para detectar aspectos
    Score >= 4 → Positivo | Score < 4 → Negativo
    """
    pred = [0.0] * len(LABELS)
    aspectos = detectar_aspectos(text)
    
    mapa_idx = {'logistica': (0,1), 'produto': (2,3), 
                'atendimento': (4,5), 'preco': (6,7)}
    
    for aspecto in aspectos:
        pos_idx, neg_idx = mapa_idx[aspecto]
        if score >= 4:
            pred[pos_idx] = 1.0
        else:
            pred[neg_idx] = 1.0
    
    return pred

class BERTimbauBaseline:
    """
    Baseline 2: BERTimbau sem fine-tuning
    Usa embeddings + similaridade cosseno para classificar sentimento
    """
    def __init__(self, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(BERTIMBAU_BASE)
        self.model = AutoModel.from_pretrained(BERTIMBAU_BASE)
        self.model.to(device)
        self.model.eval()
        
        # Palavras de referência para sentimento
        palavras_pos = ["bom", "ótimo", "excelente", "perfeito", "adorei", 
                        "amei", "recomendo", "rápido", "qualidade"]
        palavras_neg = ["ruim", "péssimo", "horrível", "terrível", "atraso", 
                        "quebrado", "defeito", "demora", "problema"]
        
        # Pré-computar embeddings de referência
        self.emb_pos = torch.stack([self._get_embedding(p) for p in palavras_pos]).mean(dim=0)
        self.emb_neg = torch.stack([self._get_embedding(p) for p in palavras_neg]).mean(dim=0)
    
    def _get_embedding(self, text):
        """Obtém embedding [CLS] do texto"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                                padding=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]
    
    def predict(self, text):
        """Prediz sentimento baseado em similaridade de embeddings"""
        pred = [0.0] * len(LABELS)
        aspectos = detectar_aspectos(text)
        
        try:
            emb_texto = self._get_embedding(str(text)[:512])
            sim_pos = F.cosine_similarity(emb_texto, self.emb_pos).item()
            sim_neg = F.cosine_similarity(emb_texto, self.emb_neg).item()
            
            mapa_idx = {'logistica': (0,1), 'produto': (2,3), 
                        'atendimento': (4,5), 'preco': (6,7)}
            
            for aspecto in aspectos:
                pos_idx, neg_idx = mapa_idx[aspecto]
                diff = sim_pos - sim_neg
                if diff > 0.02:
                    pred[pos_idx] = 1.0
                elif diff < -0.02:
                    pred[neg_idx] = 1.0
        except:
            pass
        
        return pred

class FineTunedABSA:
    """Modelo Fine-Tuned para ABSA"""
    def __init__(self, model_path, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
    
    def predict(self, text):
        """Prediz aspectos e sentimentos"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                                padding=True, max_length=128).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        pred = [0.0] * len(LABELS)
        for i, label in enumerate(LABELS):
            if probs[i] > THRESHOLDS[label]:
                pred[i] = 1.0
        
        # Resolver conflitos Pos/Neg do mesmo aspecto
        for i in range(0, len(LABELS), 2):
            if pred[i] == 1.0 and pred[i+1] == 1.0:
                if probs[i] > probs[i+1]:
                    pred[i+1] = 0.0
                else:
                    pred[i] = 0.0
        
        return pred

# ==============================================================================
# FUNÇÃO PRINCIPAL
# ==============================================================================

def main(csv_rotulado, csv_balanceado, modelo_path, output_dir="./"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")
    
    # 1. Carregar e combinar dados
    print("\n" + "="*60)
    print("CARREGANDO DADOS")
    print("="*60)
    
    df_rotulado = pd.read_csv(csv_rotulado)
    df_balanceado = pd.read_csv(csv_balanceado)
    
    if 'review_score' not in df_rotulado.columns:
        df_scores = df_balanceado[['review_id', 'review_score']].drop_duplicates()
        df = df_rotulado.merge(df_scores, on='review_id', how='left')
    else:
        df = df_rotulado.copy()
    
    if 'to_remove' in df.columns:
        df = df[df['to_remove'].str.strip().str.lower() != 'sim']
    
    df['labels'] = df.apply(criar_label, axis=1)
    text_col = 'review_text' if 'review_text' in df.columns else 'review_comment_message'
    df = df.rename(columns={text_col: 'text'})
    
    # Split teste
    _, test_df = train_test_split(df, test_size=0.20, random_state=42)
    texts = test_df['text'].tolist()
    y_true = np.array(test_df['labels'].tolist())
    test_scores = test_df['review_score'].fillna(3).values
    
    print(f"Amostras de teste: {len(texts)}")
    
    # 2. Baseline 1: Review Score
    print("\n" + "="*60)
    print("BASELINE 1: Review Score")
    print("="*60)
    y_pred_b1 = np.array([baseline_review_score(s, t) for s, t in zip(test_scores, texts)])
    print("✓ Calculado")
    
    # 3. Baseline 2: BERTimbau Base
    print("\n" + "="*60)
    print("BASELINE 2: BERTimbau Base")
    print("="*60)
    bertimbau_base = BERTimbauBaseline(device)
    y_pred_b2 = np.array([bertimbau_base.predict(t) for t in texts])
    print("✓ Calculado")
    
    # 4. Fine-Tuned ABSA
    print("\n" + "="*60)
    print("MODELO FINE-TUNED: ABSA")
    print("="*60)
    finetuned = FineTunedABSA(modelo_path, device)
    y_pred_ft = np.array([finetuned.predict(t) for t in texts])
    print("✓ Calculado")
    
    # 5. Calcular métricas
    print("\n" + "="*60)
    print("RESULTADOS")
    print("="*60)
    
    results = [
        calc_metrics(y_true, y_pred_b1, "Baseline: Review Score"),
        calc_metrics(y_true, y_pred_b2, "Baseline: BERTimbau Base"),
        calc_metrics(y_true, y_pred_ft, "Fine-Tuned: ABSA"),
    ]
    
    df_results = pd.DataFrame(results).set_index('Modelo')
    print("\n" + df_results.round(4).to_string())
    
    # 6. Melhoria incremental
    ft_f1 = df_results.loc["Fine-Tuned: ABSA", "F1-Macro"]
    b1_f1 = df_results.loc["Baseline: Review Score", "F1-Macro"]
    b2_f1 = df_results.loc["Baseline: BERTimbau Base", "F1-Macro"]
    
    m1 = ((ft_f1 - b1_f1) / max(b1_f1, 0.001)) * 100
    m2 = ((ft_f1 - b2_f1) / max(b2_f1, 0.001)) * 100
    
    print(f"\nMelhoria vs Review Score:   {m1:+.1f}%")
    print(f"Melhoria vs BERTimbau Base: {m2:+.1f}%")
    
    # 7. Métricas por classe
    print("\n" + "="*60)
    print("MÉTRICAS POR CLASSE (Fine-Tuned)")
    print("="*60)
    
    print(f"\n{'Classe':<18} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Supp':>6}")
    print("-"*50)
    for i, label in enumerate(LABELS):
        p = precision_score(y_true[:, i], y_pred_ft[:, i], zero_division=0)
        r = recall_score(y_true[:, i], y_pred_ft[:, i], zero_division=0)
        f = f1_score(y_true[:, i], y_pred_ft[:, i], zero_division=0)
        s = int(y_true[:, i].sum())
        print(f"{label:<18} {p:>8.2f} {r:>8.2f} {f:>8.2f} {s:>6}")
    
    # 8. Gráficos
    print("\n" + "="*60)
    print("GERANDO GRÁFICOS")
    print("="*60)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico 1: Comparação F1
    metrics = ['F1-Micro', 'F1-Macro', 'F1-Weighted']
    x = np.arange(len(metrics))
    width = 0.25
    colors = ['#ff7f0e', '#1f77b4', '#d62728']
    
    for i, (idx, row) in enumerate(df_results.iterrows()):
        axes[0].bar(x + i*width, [row[m] for m in metrics], width, label=idx, color=colors[i])
    
    axes[0].set_ylabel('Score')
    axes[0].set_title('Comparação de F1-Scores: Baselines vs Fine-Tuned')
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(metrics)
    axes[0].legend(loc='upper left', fontsize=9)
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Gráfico 2: F1 por classe
    f1_classes = [f1_score(y_true[:, i], y_pred_ft[:, i], zero_division=0) 
                  for i in range(len(LABELS))]
    colors_bar = ['#2ecc71' if 'Pos' in l else '#e74c3c' for l in LABELS]
    axes[1].barh(LABELS, f1_classes, color=colors_bar, alpha=0.8)
    axes[1].set_xlabel('F1-Score')
    axes[1].set_title('F1-Score por Classe (Modelo Fine-Tuned)')
    axes[1].set_xlim(0, 1)
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparacao_baselines.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Gráfico salvo: {output_dir}/comparacao_baselines.png")
    
    # 9. Salvar resultados
    df_results.to_csv(f'{output_dir}/resultados_baselines.csv')
    print(f"✓ Resultados salvos: {output_dir}/resultados_baselines.csv")
    
    # 10. Conclusão
    print("\n" + "="*60)
    print("CONCLUSÃO")
    print("="*60)
    print(f"""
    O modelo Fine-Tuned para ABSA alcançou F1-Macro de {ft_f1:.4f},
    demonstrando superioridade de:
    • {m1:+.1f}% sobre o baseline de Review Score
    • {m2:+.1f}% sobre o BERTimbau sem fine-tuning
    
    Isso valida a contribuição do fine-tuning para a tarefa de
    Aspect-Based Sentiment Analysis em e-commerce brasileiro.
    """)
    
    return df_results

# ==============================================================================
# EXECUÇÃO
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Comparação de Baselines para ABSA')
    parser.add_argument('--csv_rotulado', type=str, 
                        default=os.path.join(DADOS_DIR, 'rotulacao_reviews.csv'),
                        help='CSV com reviews rotulados')
    parser.add_argument('--csv_balanceado', type=str, 
                        default=os.path.join(DADOS_DIR, 'amostra_rotulagem_balanceada.csv'),
                        help='CSV com review_score')
    parser.add_argument('--modelo_path', type=str, 
                        default=os.path.join(MODELO_DIR, 'modelo_absa'),
                        help='Caminho para o modelo fine-tuned')
    parser.add_argument('--output_dir', type=str, 
                        default=RESULTADOS_DIR,
                        help='Diretório para salvar resultados')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args.csv_rotulado, args.csv_balanceado, args.modelo_path, args.output_dir)
