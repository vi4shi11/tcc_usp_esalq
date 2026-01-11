"""
TCC: Otimização da Inteligência de Negócios no E-commerce Brasileiro
Fine-Tuning do BERTimbau para ABSA (Aspect-Based Sentiment Analysis)

Autor: Vinicius Barreiro Shishido
Orientador: Felipe Pinto Da Silva
Curso: MBA em Data Science e Analytics - USP ESALQ

Descrição:
    Este script realiza o fine-tuning do modelo BERTimbau para a tarefa de
    Aspect-Based Sentiment Analysis (ABSA) em reviews de e-commerce brasileiro.
    
Uso:
    python fine_tuning_absa.py --csv_path ../dados/rotulacao_reviews.csv
"""

import os
import pandas as pd

# Diretório base do projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS_DIR = os.path.join(BASE_DIR, 'dados')
MODELO_DIR = os.path.join(BASE_DIR, 'modelo')
import torch
import torch.nn as nn
import numpy as np
import argparse
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
MAX_LENGTH = 128
SEED = 42

LABELS = [
    "Logística_Pos", "Logística_Neg",
    "Produto_Pos",   "Produto_Neg",
    "Atend_Pos",     "Atend_Neg",
    "Preço_Pos",     "Preço_Neg",
]
NUM_LABELS = len(LABELS)

# ==============================================================================
# FOCAL LOSS (para lidar com desbalanceamento)
# ==============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss para classificação multi-label com classes desbalanceadas.
    Referência: Lin et al., "Focal Loss for Dense Object Detection", 2017
    """
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal = ((1 - p_t) ** self.gamma) * bce
        
        if self.alpha is not None:
            focal = focal * self.alpha.to(logits.device)
        
        return focal.mean()

# ==============================================================================
# TRAINER CUSTOMIZADO
# ==============================================================================

class ABSATrainer(Trainer):
    """Trainer com Focal Loss para ABSA"""
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = FocalLoss(alpha=class_weights, gamma=2.0)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

def criar_label(row):
    """Converte colunas de sentimento para vetor one-hot"""
    vetor = [0.0] * NUM_LABELS
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

def compute_metrics(eval_pred):
    """Calcula métricas de avaliação"""
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.5).astype(int)
    
    return {
        'f1_micro': f1_score(labels, preds, average='micro', zero_division=0),
        'f1_macro': f1_score(labels, preds, average='macro', zero_division=0),
    }

def calcular_class_weights(labels_matrix, num_samples):
    """Calcula pesos inversamente proporcionais à frequência"""
    class_counts = labels_matrix.sum(axis=0)
    weights = []
    for count in class_counts:
        w = (num_samples / (count + 1)) ** 0.5
        weights.append(w)
    weights = np.array(weights)
    weights = weights / weights.min()
    return torch.tensor(weights, dtype=torch.float32)

# ==============================================================================
# FUNÇÃO PRINCIPAL
# ==============================================================================

def main(csv_path, output_dir="./modelo_absa"):
    print("="*60)
    print("FINE-TUNING BERTimbau PARA ABSA")
    print("="*60)
    
    # 1. Carregar dados
    print("\n[1/6] Carregando dados...")
    df = pd.read_csv(csv_path)
    
    if 'to_remove' in df.columns:
        df = df[df['to_remove'].str.strip().str.lower() != 'sim']
    
    cols = ['logistica_sentimento', 'produto_sentimento', 
            'atendimento_sentimento', 'preco_sentimento']
    df = df.dropna(subset=cols, how='all')
    
    df['labels'] = df.apply(criar_label, axis=1)
    text_col = 'review_text' if 'review_text' in df.columns else 'review_comment_message'
    df = df.rename(columns={text_col: 'text'})
    
    print(f"   Total de reviews: {len(df)}")
    
    # 2. Calcular class weights
    print("\n[2/6] Calculando class weights...")
    labels_matrix = np.array(df['labels'].tolist())
    class_weights = calcular_class_weights(labels_matrix, len(df))
    
    print(f"\n   {'Classe':<15} {'Qtd':>5} {'Peso':>6}")
    print("   " + "-"*28)
    for i, label in enumerate(LABELS):
        print(f"   {label:<15} {int(labels_matrix[:, i].sum()):>5} {class_weights[i]:>6.2f}")
    
    # 3. Preparar datasets
    print("\n[3/6] Preparando datasets...")
    train_df, val_df = train_test_split(
        df[['text', 'labels']], 
        test_size=0.15,
        random_state=SEED
    )
    print(f"   Treino: {len(train_df)} | Validação: {len(val_df)}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize(examples):
        enc = tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=MAX_LENGTH
        )
        enc["labels"] = [torch.tensor(l, dtype=torch.float32) for l in examples["labels"]]
        return enc
    
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))
    
    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    
    train_ds.set_format("torch")
    val_ds.set_format("torch")
    
    # 4. Carregar modelo
    print("\n[4/6] Carregando modelo BERTimbau...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification"
    )
    
    # 5. Treinar
    print("\n[5/6] Iniciando treinamento...")
    
    training_args = TrainingArguments(
        output_dir=f"{output_dir}_checkpoints",
        num_train_epochs=20,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=20,
        report_to="none",
        seed=SEED,
        fp16=torch.cuda.is_available(),
    )
    
    trainer = ABSATrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    trainer.train()
    
    # 6. Salvar modelo
    print(f"\n[6/6] Salvando modelo em '{output_dir}'...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("\n" + "="*60)
    print("✓ TREINAMENTO CONCLUÍDO!")
    print("="*60)
    
    return model, tokenizer

# ==============================================================================
# EXECUÇÃO
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tuning BERTimbau para ABSA')
    parser.add_argument('--csv_path', type=str, 
                        default=os.path.join(DADOS_DIR, 'rotulacao_reviews.csv'),
                        help='Caminho para o CSV com reviews rotulados')
    parser.add_argument('--output_dir', type=str, 
                        default=os.path.join(MODELO_DIR, 'modelo_absa'),
                        help='Diretório para salvar o modelo')
    
    args = parser.parse_args()
    main(args.csv_path, args.output_dir)
