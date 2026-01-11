import pandas as pd
import os

# Diretório base do projeto (um nível acima de codigo/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS_DIR = os.path.join(BASE_DIR, 'dados')

# Carregar o CSV
CSV_PATH = os.path.join(DADOS_DIR, "rotulacao_reviews.csv")
if not os.path.exists(CSV_PATH):
    # Tentar arquivo alternativo
    CSV_PATH = os.path.join(DADOS_DIR, "amostra_rotulagem_balanceada.csv")
    
df = pd.read_csv(CSV_PATH)
print(f"Arquivo: {CSV_PATH}")

print("="*60)
print("ANÁLISE DE DISTRIBUIÇÃO DO DATASET")
print("="*60)

# 1. Info geral
print(f"\n📊 Total de reviews: {len(df)}")

# 2. Quantos marcados para remoção?
if 'to_remove' in df.columns:
    remover = df['to_remove'].str.strip().str.lower().value_counts()
    print(f"\n🗑️  Marcados para remoção (to_remove):")
    print(remover.to_string())
    
    # Filtrar os que NÃO serão removidos
    df_filtrado = df[df['to_remove'].str.strip().str.lower() != 'sim']
    print(f"\n✅ Reviews válidos (após filtro): {len(df_filtrado)}")
else:
    df_filtrado = df

# 3. Distribuição por aspecto/sentimento
print("\n" + "="*60)
print("DISTRIBUIÇÃO POR ASPECTO E SENTIMENTO")
print("="*60)

colunas_sentimento = [
    'logistica_sentimento', 
    'produto_sentimento', 
    'atendimento_sentimento', 
    'preco_sentimento'
]

for col in colunas_sentimento:
    if col in df_filtrado.columns:
        print(f"\n📦 {col.replace('_sentimento', '').upper()}:")
        
        # Contar valores (incluindo vazios)
        contagem = df_filtrado[col].fillna('(vazio)').str.strip().value_counts()
        
        for valor, qtd in contagem.items():
            pct = qtd / len(df_filtrado) * 100
            barra = "█" * int(pct / 2)
            print(f"   {valor:12} → {qtd:4} ({pct:5.1f}%) {barra}")

# 4. Resumo para o modelo (formato one-hot)
print("\n" + "="*60)
print("RESUMO PARA O MODELO (10 CLASSES)")
print("="*60)

labels_list = [
    "Logística_Pos", "Logística_Neg",
    "Produto_Pos",   "Produto_Neg",
    "Atend_Pos",     "Atend_Neg",
    "Preço_Pos",     "Preço_Neg",
]

mapeamento = {
    'logistica_sentimento':   ("Logística_Pos", "Logística_Neg"),
    'produto_sentimento':     ("Produto_Pos", "Produto_Neg"),
    'atendimento_sentimento': ("Atend_Pos", "Atend_Neg"),
    'preco_sentimento':       ("Preço_Pos", "Preço_Neg"),
}

contagem_final = {}
for col, (nome_pos, nome_neg) in mapeamento.items():
    if col in df_filtrado.columns:
        valores = df_filtrado[col].fillna('').str.strip().str.lower()
        contagem_final[nome_pos] = (valores == 'positivo').sum()
        contagem_final[nome_neg] = (valores == 'negativo').sum()

# Ordenar por quantidade e exibir
print(f"\n{'Classe':<20} {'Qtd':>6} {'%':>8}  Distribuição")
print("-"*60)

total_labels = sum(contagem_final.values())
for classe in ["Logística_Pos", "Logística_Neg", "Produto_Pos", "Produto_Neg", 
               "Atend_Pos", "Atend_Neg", "Preço_Pos", "Preço_Neg"]:
    qtd = contagem_final.get(classe, 0)
    pct = qtd / len(df_filtrado) * 100 if len(df_filtrado) > 0 else 0
    barra = "█" * int(pct)
    print(f"{classe:<20} {qtd:>6} {pct:>7.1f}%  {barra}")

# 5. Alerta de desbalanceamento
print("\n" + "="*60)
print("⚠️  DIAGNÓSTICO DE DESBALANCEAMENTO")
print("="*60)

if contagem_final:
    max_classe = max(contagem_final.values())
    min_classe = min(v for v in contagem_final.values() if v > 0) if any(v > 0 for v in contagem_final.values()) else 0
    
    if min_classe > 0:
        ratio = max_classe / min_classe
        print(f"\nRazão max/min: {ratio:.1f}x")
        
        if ratio > 10:
            print("🔴 CRÍTICO: Desbalanceamento severo (>10x)")
            print("   → Recomendado: Class Weights + Focal Loss")
        elif ratio > 5:
            print("🟡 MODERADO: Desbalanceamento significativo (5-10x)")
            print("   → Recomendado: Class Weights")
        else:
            print("🟢 OK: Desbalanceamento aceitável (<5x)")
    else:
        print("🔴 ALERTA: Existem classes com 0 exemplos!")

# 6. Reviews sem nenhuma classificação
sem_classe = df_filtrado[colunas_sentimento].isna().all(axis=1).sum()
print(f"\n📭 Reviews sem nenhum aspecto rotulado: {sem_classe}")

