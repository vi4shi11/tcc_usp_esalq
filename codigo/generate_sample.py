# codigo/generate_sample.py
import pandas as pd
import os
import sys
import re

# Diretório base do projeto (um nível acima de codigo/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS_DIR = os.path.join(BASE_DIR, 'dados')

try:
    import kagglehub
except ImportError:
    print("Error: 'kagglehub' library not found. Please run 'pip install kagglehub'.")
    sys.exit(1)

# =============================================================================
# REGEX PATTERNS POR ASPECTO
# =============================================================================

PATTERNS = {
    'logistica': re.compile(
        r'\b(entreg\w*|prazo|atraso\w*|atras\w*|demor\w*|cheg\w*|'
        r'frete|transport\w*|correio\w*|envi\w*|rastreio|rastream\w*|'
        r'embalag\w*|caixa|pacote|extravi\w*|perder|perdid\w*)\b',
        re.IGNORECASE
    ),
    'produto': re.compile(
        r'\b(produto|qualidade|material|acabamento|funciona\w*|defeito\w*|'
        r'quebr\w*|estrag\w*|danific\w*|original|falso|falsific\w*|'
        r'tamanho|cor|modelo|foto|imagem|descri\w*|especifica\w*|'
        r'usado|novo|velho|bom|ruim|ótimo|péssim\w*|excelente|horrível)\b',
        re.IGNORECASE
    ),
    'atendimento': re.compile(
        r'\b(vendedor\w*|loja|atendimento|atend\w*|resposta|respond\w*|'
        r'contato|comunica\w*|suporte|ajuda|solução|resolver|'
        r'educad\w*|grosseir\w*|ignor\w*|descaso|reclama\w*)\b',
        re.IGNORECASE
    ),
    'preco': re.compile(
        r'\b(preço|preco|valor|caro|barat\w*|custo|benefício|beneficio|'
        r'pag\w*|cobr\w*|taxa|desconto|promoção|promocao|oferta|'
        r'dinheiro|reais|R\$|compens\w*|econômic\w*|economic\w*)\b',
        re.IGNORECASE
    )
}

def detectar_aspectos(texto):
    """Retorna lista de aspectos mencionados no texto"""
    if pd.isna(texto):
        return []
    aspectos = []
    for aspecto, pattern in PATTERNS.items():
        if pattern.search(texto):
            aspectos.append(aspecto)
    return aspectos

def generate_balanced_sample():
    print(">>> STEP 1: Downloading data from Kaggle...")
    
    dataset = 'olistbr/brazilian-ecommerce'
    file_name = 'olist_order_reviews_dataset.csv'
    
    try:
        print(f"Downloading dataset {dataset}...")
        path = kagglehub.dataset_download(dataset)
        print(f"Path to dataset files: {path}")
        
        file_path = os.path.join(path, file_name)
        
        if not os.path.exists(file_path):
            print(f"Error: File {file_name} not found.")
            return
        
        print(f"Found file at: {file_path}")

    except Exception as e:
        print(f"Error during Kaggle download: {e}")
        return

    print(f"\n>>> STEP 2: Loading and Processing...")
    df = pd.read_csv(file_path)
    
    # Filtrar reviews com texto
    print(f"Total original: {len(df)}")
    df = df.dropna(subset=['review_comment_message']).copy()
    print(f"Com texto: {len(df)}")
    
    # Filtrar comentários muito curtos (< 20 caracteres)
    df['text_len'] = df['review_comment_message'].str.len()
    df = df[df['text_len'] >= 20]
    print(f"Com >= 20 caracteres: {len(df)}")
    
    # =========================================================================
    # DETECTAR ASPECTOS MENCIONADOS (via regex)
    # =========================================================================
    
    print("\n>>> STEP 3: Detectando aspectos via regex...")
    df['aspectos'] = df['review_comment_message'].apply(detectar_aspectos)
    df['num_aspectos'] = df['aspectos'].apply(len)
    
    # Criar flags por aspecto
    for aspecto in PATTERNS.keys():
        df[f'tem_{aspecto}'] = df['aspectos'].apply(lambda x: aspecto in x)
    
    # Estatísticas
    print("\nDistribuição de aspectos detectados:")
    for aspecto in PATTERNS.keys():
        count = df[f'tem_{aspecto}'].sum()
        pct = count / len(df) * 100
        print(f"  {aspecto.capitalize():<12}: {count:>5} ({pct:.1f}%)")
    
    print(f"\nReviews com pelo menos 1 aspecto: {(df['num_aspectos'] > 0).sum()}")
    print(f"Reviews sem aspectos detectados: {(df['num_aspectos'] == 0).sum()}")
    
    # =========================================================================
    # AMOSTRAGEM ESTRATIFICADA POR REVIEW_SCORE + ASPECTOS
    # =========================================================================
    
    print(f"\n>>> STEP 4: Amostragem estratificada...")
    
    # Primeiro: garantir mínimo de cada aspecto (especialmente os raros)
    amostras = []
    ids_usados = set()
    
    # Priorizar aspectos raros: preço e atendimento
    aspectos_prioritarios = ['preco', 'atendimento', 'logistica', 'produto']
    min_por_aspecto = {'preco': 30, 'atendimento': 30, 'logistica': 40, 'produto': 40}
    
    print("\nGarantindo mínimo por aspecto:")
    for aspecto in aspectos_prioritarios:
        df_aspecto = df[(df[f'tem_{aspecto}']) & (~df['review_id'].isin(ids_usados))]
        
        # Balancear por score dentro do aspecto
        n_needed = min_por_aspecto[aspecto]
        
        # Pegar metade de scores baixos (1-2) e metade de altos (4-5)
        df_neg = df_aspecto[df_aspecto['review_score'].isin([1, 2])]
        df_pos = df_aspecto[df_aspecto['review_score'].isin([4, 5])]
        
        n_neg = min(len(df_neg), n_needed // 2)
        n_pos = min(len(df_pos), n_needed // 2)
        
        if n_neg > 0:
            sample_neg = df_neg.sample(n=n_neg, random_state=42)
            amostras.append(sample_neg)
            ids_usados.update(sample_neg['review_id'])
        
        if n_pos > 0:
            sample_pos = df_pos.sample(n=n_pos, random_state=42)
            amostras.append(sample_pos)
            ids_usados.update(sample_pos['review_id'])
        
        print(f"  {aspecto.capitalize():<12}: {n_neg} neg + {n_pos} pos = {n_neg + n_pos}")
    
    # Completar até 200 com distribuição por score
    amostra_parcial = pd.concat(amostras, ignore_index=True)
    faltam = 200 - len(amostra_parcial)
    
    if faltam > 0:
        print(f"\nCompletando com mais {faltam} reviews...")
        df_restante = df[~df['review_id'].isin(ids_usados)]
        
        # Distribuir igualmente por score
        por_score = faltam // 5
        for score in [1, 2, 3, 4, 5]:
            df_score = df_restante[df_restante['review_score'] == score]
            n = min(len(df_score), por_score)
            if n > 0:
                sample = df_score.sample(n=n, random_state=42+score)
                amostras.append(sample)
    
    amostra_final = pd.concat(amostras, ignore_index=True).drop_duplicates(subset='review_id')
    
    # Limitar a 200 e shuffle
    if len(amostra_final) > 200:
        amostra_final = amostra_final.sample(n=200, random_state=99)
    
    amostra_final = amostra_final.sample(frac=1, random_state=123).reset_index(drop=True)
    
    print(f"\nTotal amostrado: {len(amostra_final)}")
    
    # =========================================================================
    # PREPARAR PLANILHA PARA ROTULAGEM ABSA
    # =========================================================================
    
    print(f"\n>>> STEP 5: Preparando planilha para rotulagem ABSA...")
    
    # Recalcular aspectos para a amostra final
    amostra_final['aspectos'] = amostra_final['review_comment_message'].apply(detectar_aspectos)
    
    planilha = pd.DataFrame({
        'review_id': amostra_final['review_id'],
        'review_score': amostra_final['review_score'],
        'review_text': amostra_final['review_comment_message'],
        'aspectos_detectados': amostra_final['aspectos'].apply(lambda x: ', '.join(x) if x else ''),
        
        # Colunas para rotulagem (Positivo / Negativo / vazio)
        'logistica_sentimento': '',
        'produto_sentimento': '',
        'atendimento_sentimento': '',
        'preco_sentimento': '',
        
        'to_remove': 'Não'
    })
    
    # =========================================================================
    # EXPORTAR
    # =========================================================================
    
    output_xlsx = os.path.join(DADOS_DIR, 'Amostra_Rotulagem_ABSA_Balanceada.xlsx')
    output_csv = os.path.join(DADOS_DIR, 'amostra_rotulagem_balanceada.csv')
    
    # Garantir que o diretório existe
    os.makedirs(DADOS_DIR, exist_ok=True)
    
    planilha.to_excel(output_xlsx, index=False)
    planilha.to_csv(output_csv, index=False)
    
    print(f"\n>>> SUCCESS!")
    print(f"  Arquivo Excel: {output_xlsx}")
    print(f"  Arquivo CSV:   {output_csv}")
    print(f"  Total de amostras: {len(planilha)}")
    
    # Resumo da distribuição
    print(f"\n>>> Distribuição final por review_score:")
    print(planilha['review_score'].value_counts().sort_index())
    
    # Distribuição de aspectos na amostra final
    print(f"\n>>> Distribuição de aspectos na amostra:")
    for aspecto in PATTERNS.keys():
        count = planilha['aspectos_detectados'].str.contains(aspecto).sum()
        print(f"  {aspecto.capitalize():<12}: {count}")
    
    print(f"\n>>> Dica: A coluna 'aspectos_detectados' mostra os aspectos")
    print("    identificados pela regex para ajudar na rotulagem!")

if __name__ == "__main__":
    generate_balanced_sample()