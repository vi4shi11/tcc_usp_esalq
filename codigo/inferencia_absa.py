"""
TCC: Otimização da Inteligência de Negócios no E-commerce Brasileiro
Inferência do Modelo ABSA Fine-Tuned

Autor: Vinicius Barreiro Shishido
Orientador: Felipe Pinto Da Silva
Curso: MBA em Data Science e Analytics - USP ESALQ

Descrição:
    Este script realiza inferência usando o modelo BERTimbau fine-tuned
    para Aspect-Based Sentiment Analysis (ABSA).
    
Uso:
    python inferencia_absa.py
    python inferencia_absa.py --modelo_path ../modelo/modelo_absa
    python inferencia_absa.py --texto "Produto excelente, entrega rápida!"
"""

import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================

# Diretório base do projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELO_DIR = os.path.join(BASE_DIR, 'modelo')

# Labels do modelo
LABELS = [
    "Logística_Pos", "Logística_Neg",
    "Produto_Pos", "Produto_Neg",
    "Atend_Pos", "Atend_Neg",
    "Preço_Pos", "Preço_Neg"
]

# Thresholds otimizados (baseados na validação do baseline_comparison.py)
DEFAULT_THRESHOLDS = {
    "Logística_Pos": 0.50, "Logística_Neg": 0.50,
    "Produto_Pos": 0.55,   "Produto_Neg": 0.45,
    "Atend_Pos": 0.30,     "Atend_Neg": 0.45,
    "Preço_Pos": 0.25,     "Preço_Neg": 0.30,
}

MAX_LENGTH = 128

# ==============================================================================
# CLASSE DE INFERÊNCIA
# ==============================================================================

class ABSAClassifier:
    """Classificador ABSA para reviews de e-commerce brasileiro."""
    
    def __init__(self, modelo_path, thresholds=None, device=None):
        """
        Inicializa o classificador.
        
        Args:
            modelo_path: Caminho para o modelo treinado
            thresholds: Dicionário com thresholds por label (opcional)
            device: 'cuda', 'mps', 'cpu' ou None (auto-detect)
        """
        self.modelo_path = modelo_path
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        
        # Detectar dispositivo
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Dispositivo: {self.device}")
        
        # Carregar modelo e tokenizer
        print(f"📂 Carregando modelo de: {modelo_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(modelo_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(modelo_path)
        self.model.to(self.device)
        self.model.eval()
        print("✅ Modelo carregado!")
    
    def classificar(self, texto, verbose=True, resolver_conflitos=True):
        """
        Classifica um texto de review.
        
        Args:
            texto: Texto do review
            verbose: Se True, imprime os resultados
            resolver_conflitos: Se True, resolve conflitos Pos/Neg do mesmo aspecto
            
        Returns:
            Lista de tuplas (label, score) para labels detectados
        """
        # Tokenizar
        inputs = self.tokenizer(
            texto, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=MAX_LENGTH
        ).to(self.device)
        
        # Inferência
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Aplicar thresholds
        resultados = []
        for i, label in enumerate(LABELS):
            t = self.thresholds.get(label, 0.5)
            score = probs[i]
            if score > t:
                resultados.append((label, float(score)))
        
        # Resolver conflitos: Pos vs Neg do mesmo aspecto
        if resolver_conflitos:
            resultados = self._resolver_conflitos(resultados)
        
        # Ordenar por score
        resultados.sort(key=lambda x: x[1], reverse=True)
        
        if verbose:
            self._print_resultados(texto, resultados, probs)
        
        return resultados
    
    def _resolver_conflitos(self, resultados):
        """Se ambos Pos e Neg foram detectados para um aspecto, mantém o maior."""
        aspectos = ['Logística', 'Produto', 'Atend', 'Preço']
        resultados_filtrados = []
        
        for aspecto in aspectos:
            pos = next((r for r in resultados if r[0] == f"{aspecto}_Pos"), None)
            neg = next((r for r in resultados if r[0] == f"{aspecto}_Neg"), None)
            
            if pos and neg:
                # Conflito! Manter o de maior score
                if pos[1] > neg[1]:
                    resultados_filtrados.append(pos)
                else:
                    resultados_filtrados.append(neg)
            else:
                if pos:
                    resultados_filtrados.append(pos)
                if neg:
                    resultados_filtrados.append(neg)
        
        return resultados_filtrados
    
    def _print_resultados(self, texto, resultados, probs):
        """Imprime os resultados formatados."""
        if resultados:
            for label, score in resultados:
                emoji = "✅" if "_Pos" in label else "❌"
                print(f"  {emoji} {label}: {score:.1%}")
        else:
            # Mostrar top 3 para debug
            top3 = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:3]
            print("  ⚪ Nenhum acima do threshold. Top 3:")
            for idx, score in top3:
                t = self.thresholds.get(LABELS[idx], 0.5)
                print(f"     {LABELS[idx]}: {score:.1%} (threshold: {t:.0%})")
    
    def classificar_batch(self, textos, verbose=False):
        """
        Classifica múltiplos textos.
        
        Args:
            textos: Lista de textos
            verbose: Se True, imprime cada resultado
            
        Returns:
            Lista de resultados para cada texto
        """
        return [self.classificar(texto, verbose=verbose) for texto in textos]


# ==============================================================================
# TESTES
# ==============================================================================

def executar_testes(classifier):
    """Executa testes de inferência com exemplos diversos."""
    
    print("\n" + "="*60)
    print("TESTES DE INFERÊNCIA")
    print("="*60)
    
    testes = [
        # Casos mistos
        "O celular é incrível, mas a entrega demorou uma eternidade.",
        
        # Produto negativo
        "Produto de péssima qualidade, veio quebrado.",
        
        # Logística + Atendimento positivos
        "Entrega rápida, vendedor muito atencioso!",
        
        # Preço negativo
        "Caro demais pelo que oferece, não vale o preço.",
        
        # Tudo positivo
        "Amei! Super recomendo, entrega antes do prazo.",
        
        # Atendimento negativo
        "Péssimo atendimento, não responderam minhas mensagens.",
        
        # Preço positivo
        "Ótimo custo-benefício, preço justo pelo que entrega.",
        
        # Produto + Logística positivos
        "Produto excelente, chegou antes do prazo!",
        
        # Múltiplos negativos
        "Horrível, veio errado e o vendedor foi grosseiro.",
        
        # Neutro/ambíguo
        "Recebi o produto conforme descrito.",
    ]
    
    for texto in testes:
        print(f"\n📝 '{texto}'")
        classifier.classificar(texto)
    
    print("\n" + "="*60)
    print("TESTES CONCLUÍDOS")
    print("="*60)


# ==============================================================================
# EXECUÇÃO
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inferência ABSA com BERTimbau')
    parser.add_argument('--modelo_path', type=str, 
                        default=os.path.join(MODELO_DIR, 'modelo_absa'),
                        help='Caminho para o modelo treinado')
    parser.add_argument('--texto', type=str, default=None,
                        help='Texto para classificar (opcional)')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'mps', 'cpu'],
                        help='Dispositivo para inferência')
    parser.add_argument('--testes', action='store_true',
                        help='Executar testes de exemplo')
    
    args = parser.parse_args()
    
    # Verificar se modelo existe
    if not os.path.exists(args.modelo_path):
        print(f"❌ Modelo não encontrado em: {args.modelo_path}")
        print("   Execute primeiro: python fine_tuning_absa.py")
        exit(1)
    
    # Inicializar classificador
    classifier = ABSAClassifier(args.modelo_path, device=args.device)
    
    if args.texto:
        # Classificar texto específico
        print(f"\n📝 '{args.texto}'")
        classifier.classificar(args.texto)
    elif args.testes:
        # Executar testes
        executar_testes(classifier)
    else:
        # Modo interativo
        print("\n" + "="*60)
        print("MODO INTERATIVO")
        print("Digite um review para classificar (ou 'sair' para terminar)")
        print("="*60)
        
        while True:
            try:
                texto = input("\n📝 Review: ").strip()
                if texto.lower() in ['sair', 'exit', 'quit', 'q']:
                    print("👋 Até logo!")
                    break
                if texto:
                    classifier.classificar(texto)
            except KeyboardInterrupt:
                print("\n👋 Até logo!")
                break
