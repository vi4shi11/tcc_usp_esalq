# scripts/generate_labeling_html.py
import pandas as pd
import json
import os

def generate_labeling_html():
    """Generate an HTML file for labeling reviews with categories and sentiments."""
    
    # Read the Excel file (prefer balanced version if available)
    excel_file = 'Amostra_Rotulagem_ABSA_Balanceada.xlsx'
    if not os.path.exists(excel_file):
        excel_file = 'Amostra_Rotulagem_TCC.xlsx'
        if not os.path.exists(excel_file):
            print(f"Error: No sample file found. Please run generate_sample_balanced.py first.")
            return
    
    print(f"Using: {excel_file}")
    df = pd.read_excel(excel_file)
    
    # Convert to list of dictionaries for JSON
    # Handle different column names for text
    text_col = 'review_text' if 'review_text' in df.columns else 'review_comment_message'
    
    reviews = []
    for idx, row in df.iterrows():
        reviews.append({
            'id': str(row['review_id']),
            'text': str(row[text_col]) if pd.notna(row[text_col]) else '',
            'index': idx
        })
    
    # Create HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rotulação de Reviews - TCC</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 28px;
            margin-bottom: 10px;
        }}
        
        .header .progress {{
            margin-top: 15px;
            font-size: 16px;
            opacity: 0.9;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 8px;
            background: rgba(255,255,255,0.3);
            border-radius: 4px;
            margin-top: 10px;
            overflow: hidden;
        }}
        
        .progress-fill {{
            height: 100%;
            background: white;
            transition: width 0.3s ease;
            border-radius: 4px;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .review-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 25px;
            margin-bottom: 30px;
            border-left: 4px solid #667eea;
        }}
        
        .review-id {{
            font-size: 12px;
            color: #6c757d;
            margin-bottom: 10px;
            font-family: monospace;
        }}
        
        .review-text {{
            font-size: 18px;
            line-height: 1.6;
            color: #212529;
            margin-bottom: 20px;
        }}
        
        .label-section {{
            margin-bottom: 25px;
        }}
        
        .label-title {{
            font-size: 16px;
            font-weight: 600;
            color: #495057;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
        }}
        
        .label-title::before {{
            content: '';
            width: 4px;
            height: 20px;
            background: #667eea;
            margin-right: 10px;
            border-radius: 2px;
        }}
        
        .checkbox-group {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
        }}
        
        .checkbox-item {{
            display: flex;
            align-items: center;
            background: white;
            padding: 12px 20px;
            border-radius: 6px;
            border: 2px solid #dee2e6;
            cursor: pointer;
            transition: all 0.2s;
            user-select: none;
        }}
        
        .checkbox-item:hover {{
            border-color: #667eea;
            background: #f0f4ff;
        }}
        
        .checkbox-item input[type="checkbox"] {{
            margin-right: 10px;
            width: 18px;
            height: 18px;
            cursor: pointer;
        }}
        
        .checkbox-item.checked {{
            border-color: #667eea;
            background: #e7edff;
        }}
        
        .checkbox-item label {{
            cursor: pointer;
            font-size: 15px;
            color: #495057;
            margin: 0;
        }}
        
        .category-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 15px;
            border: 2px solid #dee2e6;
            transition: all 0.2s;
        }}
        
        .category-card:hover {{
            border-color: #667eea;
        }}
        
        .category-card.active {{
            border-color: #667eea;
            background: #e7edff;
        }}
        
        .category-name {{
            font-size: 16px;
            font-weight: 600;
            color: #495057;
            margin-bottom: 15px;
        }}
        
        .radio-group {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }}
        
        .radio-item {{
            display: flex;
            align-items: center;
            background: white;
            padding: 10px 18px;
            border-radius: 6px;
            border: 2px solid #dee2e6;
            cursor: pointer;
            transition: all 0.2s;
            user-select: none;
        }}
        
        .radio-item:hover {{
            border-color: #667eea;
            background: #f0f4ff;
        }}
        
        .radio-item input[type="radio"] {{
            margin-right: 8px;
            width: 16px;
            height: 16px;
            cursor: pointer;
        }}
        
        .radio-item.checked {{
            border-color: #667eea;
            background: #e7edff;
        }}
        
        .radio-item.checked.positivo {{
            border-color: #28a745;
            background: #d4edda;
        }}
        
        .radio-item.checked.negativo {{
            border-color: #dc3545;
            background: #f8d7da;
        }}
        
        .radio-item.checked.neutro {{
            border-color: #6c757d;
            background: #e2e3e5;
        }}
        
        .radio-item label {{
            cursor: pointer;
            font-size: 14px;
            color: #495057;
            margin: 0;
        }}
        
        .remove-flag {{
            background: #fff3cd;
            border: 2px solid #ffc107;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 25px;
        }}
        
        .remove-flag.checked {{
            background: #ffeaa7;
            border-color: #f39c12;
        }}
        
        .remove-flag-item {{
            display: flex;
            align-items: center;
            cursor: pointer;
        }}
        
        .remove-flag-item input[type="checkbox"] {{
            margin-right: 12px;
            width: 20px;
            height: 20px;
            cursor: pointer;
        }}
        
        .remove-flag-item label {{
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            color: #856404;
            margin: 0;
        }}
        
        .navigation {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 30px;
            padding-top: 30px;
            border-top: 2px solid #dee2e6;
        }}
        
        .nav-button {{
            padding: 12px 30px;
            font-size: 16px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
            font-weight: 600;
        }}
        
        .nav-button.prev {{
            background: #6c757d;
            color: white;
        }}
        
        .nav-button.prev:hover {{
            background: #5a6268;
        }}
        
        .nav-button.next {{
            background: #667eea;
            color: white;
        }}
        
        .nav-button.next:hover {{
            background: #5568d3;
        }}
        
        .nav-button:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}
        
        .review-counter {{
            font-size: 16px;
            color: #6c757d;
            font-weight: 500;
        }}
        
        .actions {{
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }}
        
        .action-button {{
            padding: 10px 20px;
            font-size: 14px;
            border: 2px solid #667eea;
            background: white;
            color: #667eea;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
            font-weight: 600;
        }}
        
        .action-button:hover {{
            background: #667eea;
            color: white;
        }}
        
        .action-button.export {{
            background: #28a745;
            border-color: #28a745;
            color: white;
        }}
        
        .action-button.export:hover {{
            background: #218838;
        }}
        
        .stats {{
            display: flex;
            gap: 20px;
            margin-top: 20px;
            padding: 15px;
            background: #e7edff;
            border-radius: 6px;
            font-size: 14px;
        }}
        
        .stat-item {{
            flex: 1;
        }}
        
        .stat-label {{
            color: #6c757d;
            font-size: 12px;
        }}
        
        .stat-value {{
            color: #667eea;
            font-size: 20px;
            font-weight: 600;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📝 Rotulação de Reviews</h1>
            <div class="progress">
                <div>Progresso: <span id="progress-text">0/200</span></div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progress-fill" style="width: 0%"></div>
                </div>
            </div>
        </div>
        
        <div class="content">
            <div class="review-card">
                <div class="review-id" id="review-id">ID: -</div>
                <div class="review-text" id="review-text">Carregando...</div>
            </div>
            
            <div class="label-section">
                <div class="label-title">Categorias e Sentimentos</div>
                <p style="color: #6c757d; margin-bottom: 20px; font-size: 14px;">Para cada categoria, selecione o sentimento (Positivo, Negativo ou Neutro). Você pode selecionar múltiplas categorias.</p>
                
                <div class="category-card" id="card-logistica">
                    <div class="category-name">📦 Logística</div>
                    <div class="radio-group">
                        <div class="radio-item" id="radio-logistica-positivo">
                            <input type="radio" name="sentiment-logistica" id="logistica-positivo" value="Positivo">
                            <label for="logistica-positivo">✅ Positivo</label>
                        </div>
                        <div class="radio-item" id="radio-logistica-negativo">
                            <input type="radio" name="sentiment-logistica" id="logistica-negativo" value="Negativo">
                            <label for="logistica-negativo">❌ Negativo</label>
                        </div>
                        <div class="radio-item" id="radio-logistica-neutro">
                            <input type="radio" name="sentiment-logistica" id="logistica-neutro" value="Neutro">
                            <label for="logistica-neutro">➖ Neutro</label>
                        </div>
                    </div>
                </div>
                
                <div class="category-card" id="card-produto">
                    <div class="category-name">🛍️ Produto</div>
                    <div class="radio-group">
                        <div class="radio-item" id="radio-produto-positivo">
                            <input type="radio" name="sentiment-produto" id="produto-positivo" value="Positivo">
                            <label for="produto-positivo">✅ Positivo</label>
                        </div>
                        <div class="radio-item" id="radio-produto-negativo">
                            <input type="radio" name="sentiment-produto" id="produto-negativo" value="Negativo">
                            <label for="produto-negativo">❌ Negativo</label>
                        </div>
                        <div class="radio-item" id="radio-produto-neutro">
                            <input type="radio" name="sentiment-produto" id="produto-neutro" value="Neutro">
                            <label for="produto-neutro">➖ Neutro</label>
                        </div>
                    </div>
                </div>
                
                <div class="category-card" id="card-atendimento">
                    <div class="category-name">👥 Atendimento</div>
                    <div class="radio-group">
                        <div class="radio-item" id="radio-atendimento-positivo">
                            <input type="radio" name="sentiment-atendimento" id="atendimento-positivo" value="Positivo">
                            <label for="atendimento-positivo">✅ Positivo</label>
                        </div>
                        <div class="radio-item" id="radio-atendimento-negativo">
                            <input type="radio" name="sentiment-atendimento" id="atendimento-negativo" value="Negativo">
                            <label for="atendimento-negativo">❌ Negativo</label>
                        </div>
                        <div class="radio-item" id="radio-atendimento-neutro">
                            <input type="radio" name="sentiment-atendimento" id="atendimento-neutro" value="Neutro">
                            <label for="atendimento-neutro">➖ Neutro</label>
                        </div>
                    </div>
                </div>
                
                <div class="category-card" id="card-preco">
                    <div class="category-name">💰 Preço</div>
                    <div class="radio-group">
                        <div class="radio-item" id="radio-preco-positivo">
                            <input type="radio" name="sentiment-preco" id="preco-positivo" value="Positivo">
                            <label for="preco-positivo">✅ Positivo</label>
                        </div>
                        <div class="radio-item" id="radio-preco-negativo">
                            <input type="radio" name="sentiment-preco" id="preco-negativo" value="Negativo">
                            <label for="preco-negativo">❌ Negativo</label>
                        </div>
                        <div class="radio-item" id="radio-preco-neutro">
                            <input type="radio" name="sentiment-preco" id="preco-neutro" value="Neutro">
                            <label for="preco-neutro">➖ Neutro</label>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="remove-flag" id="remove-flag-container">
                <div class="remove-flag-item">
                    <input type="checkbox" id="to-remove" value="true">
                    <label for="to-remove">🗑️ Marcar para remover</label>
                </div>
            </div>
            
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-label">Categorias rotuladas</div>
                    <div class="stat-value" id="cat-count">0</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Marcado para remover</div>
                    <div class="stat-value" id="remove-flag-status">Não</div>
                </div>
            </div>
            
            <div class="navigation">
                <button class="nav-button prev" id="prev-btn" onclick="previousReview()">← Anterior</button>
                <div class="review-counter">
                    Review <span id="current-index">1</span> de <span id="total-reviews">200</span>
                </div>
                <button class="nav-button next" id="next-btn" onclick="nextReview()">Próximo →</button>
            </div>
            
            <div class="actions">
                <button class="action-button" onclick="saveProgress()">💾 Salvar Progresso</button>
                <button class="action-button export" onclick="exportData()">📥 Exportar Dados</button>
                <button class="action-button" onclick="loadProgress()">🔄 Carregar Progresso</button>
                <button class="action-button" style="border-color: #dc3545; color: #dc3545;" onclick="clearCache()">🗑️ Limpar Cache</button>
            </div>
        </div>
    </div>
    
    <script>
        // Data from Excel
        const reviews = {json.dumps(reviews, ensure_ascii=False, indent=8)};
        
        // Labels storage
        let labels = JSON.parse(localStorage.getItem('review_labels') || '{{}}');
        
        let currentIndex = parseInt(localStorage.getItem('current_review_index') || '0');
        
        // Category mappings
        const categories = ['logistica', 'produto', 'atendimento', 'preco'];
        const categoryNames = {{
            'logistica': 'Logística',
            'produto': 'Produto',
            'atendimento': 'Atendimento',
            'preco': 'Preço'
        }};
        const sentiments = ['positivo', 'negativo', 'neutro'];
        
        // Initialize
        function init() {{
            loadReview(currentIndex);
            updateProgress();
            setupListeners();
        }}
        
        // Track previously selected radio for toggle functionality
        const previousSelection = {{}};
        
        function setupListeners() {{
            // Radio buttons for categories - with toggle (click again to unselect)
            categories.forEach(cat => {{
                previousSelection[cat] = null;
                
                sentiments.forEach(sent => {{
                    const radio = document.getElementById(`${{cat}}-${{sent}}`);
                    if (radio) {{
                        // Use click instead of change to detect re-clicks
                        radio.addEventListener('click', function(e) {{
                            if (previousSelection[cat] === sent) {{
                                // Clicking the same option again - unselect it
                                this.checked = false;
                                previousSelection[cat] = null;
                            }} else {{
                                // Selecting a new option
                                previousSelection[cat] = sent;
                            }}
                            updateStats();
                        }});
                    }}
                }});
            }});
            
            // Remove flag checkbox
            const removeCheckbox = document.getElementById('to-remove');
            if (removeCheckbox) {{
                removeCheckbox.addEventListener('change', updateStats);
            }}
        }}
        
        function loadReview(index) {{
            if (index < 0 || index >= reviews.length) return;
            
            currentIndex = index;
            const review = reviews[index];
            
            document.getElementById('review-id').textContent = `ID: ${{review.id}}`;
            document.getElementById('review-text').textContent = review.text;
            document.getElementById('current-index').textContent = index + 1;
            document.getElementById('total-reviews').textContent = reviews.length;
            
            // Load saved labels
            const reviewId = review.id;
            const savedLabels = labels[reviewId] || {{}};
            
            // Clear all radio buttons and category cards, reset previous selection tracking
            categories.forEach(cat => {{
                previousSelection[cat] = null;
                sentiments.forEach(sent => {{
                    const radio = document.getElementById(`${{cat}}-${{sent}}`);
                    if (radio) {{
                        radio.checked = false;
                        const radioItem = document.getElementById(`radio-${{cat}}-${{sent}}`);
                        if (radioItem) {{
                            radioItem.classList.remove('checked', 'positivo', 'negativo', 'neutro');
                        }}
                    }}
                }});
                const card = document.getElementById(`card-${{cat}}`);
                if (card) {{
                    card.classList.remove('active');
                }}
            }});
            
            // Restore category sentiments
            if (savedLabels.categories) {{
                const categoryKeyMap = {{
                    'Logística': 'logistica',
                    'Produto': 'produto',
                    'Atendimento': 'atendimento',
                    'Preço': 'preco'
                }};
                
                Object.keys(savedLabels.categories).forEach(catName => {{
                    const sentiment = savedLabels.categories[catName];
                    const catId = categoryKeyMap[catName];
                    
                    if (catId && sentiment && sentiments.includes(sentiment.toLowerCase())) {{
                        const radio = document.getElementById(`${{catId}}-${{sentiment.toLowerCase()}}`);
                        if (radio) {{
                            radio.checked = true;
                            // Track this as the previous selection for toggle functionality
                            previousSelection[catId] = sentiment.toLowerCase();
                            const radioItem = document.getElementById(`radio-${{catId}}-${{sentiment.toLowerCase()}}`);
                            if (radioItem) {{
                                radioItem.classList.add('checked', sentiment.toLowerCase());
                            }}
                            const card = document.getElementById(`card-${{catId}}`);
                            if (card) {{
                                card.classList.add('active');
                            }}
                        }}
                    }}
                }});
            }}
            
            // Restore remove flag
            const removeCheckbox = document.getElementById('to-remove');
            const removeContainer = document.getElementById('remove-flag-container');
            if (removeCheckbox && removeContainer) {{
                removeCheckbox.checked = savedLabels.toRemove || false;
                if (removeCheckbox.checked) {{
                    removeContainer.classList.add('checked');
                }} else {{
                    removeContainer.classList.remove('checked');
                }}
            }}
            
            // Update navigation buttons
            document.getElementById('prev-btn').disabled = index === 0;
            document.getElementById('next-btn').disabled = index === reviews.length - 1;
            
            updateStats();
        }}
        
        function saveCurrentLabels() {{
            const reviewId = reviews[currentIndex].id;
            const categorySentiments = {{}};
            
            // Get sentiment for each category
            categories.forEach(cat => {{
                sentiments.forEach(sent => {{
                    const radio = document.getElementById(`${{cat}}-${{sent}}`);
                    if (radio && radio.checked) {{
                        const catName = categoryNames[cat];
                        categorySentiments[catName] = sent.charAt(0).toUpperCase() + sent.slice(1);
                    }}
                }});
            }});
            
            // Get remove flag
            const removeCheckbox = document.getElementById('to-remove');
            const toRemove = removeCheckbox ? removeCheckbox.checked : false;
            
            labels[reviewId] = {{
                categories: categorySentiments,
                toRemove: toRemove
            }};
            
            localStorage.setItem('review_labels', JSON.stringify(labels));
            updateProgress();
        }}
        
        function previousReview() {{
            if (currentIndex > 0) {{
                saveCurrentLabels();
                loadReview(currentIndex - 1);
                localStorage.setItem('current_review_index', currentIndex.toString());
            }}
        }}
        
        function nextReview() {{
            if (currentIndex < reviews.length - 1) {{
                saveCurrentLabels();
                loadReview(currentIndex + 1);
                localStorage.setItem('current_review_index', currentIndex.toString());
            }}
        }}
        
        function updateStats() {{
            // Count labeled categories and update visual states
            let catCount = 0;
            categories.forEach(cat => {{
                let hasSelection = false;
                let selectedSentiment = null;
                
                // First, clear all radio items for this category
                sentiments.forEach(sent => {{
                    const radioItem = document.getElementById(`radio-${{cat}}-${{sent}}`);
                    if (radioItem) {{
                        radioItem.classList.remove('checked', 'positivo', 'negativo', 'neutro');
                    }}
                }});
                
                // Then, mark the selected one
                sentiments.forEach(sent => {{
                    const radio = document.getElementById(`${{cat}}-${{sent}}`);
                    if (radio && radio.checked) {{
                        hasSelection = true;
                        selectedSentiment = sent;
                        const radioItem = document.getElementById(`radio-${{cat}}-${{sent}}`);
                        if (radioItem) {{
                            radioItem.classList.add('checked', sent);
                        }}
                    }}
                }});
                
                // Update category card
                const card = document.getElementById(`card-${{cat}}`);
                if (card) {{
                    if (hasSelection) {{
                        card.classList.add('active');
                        catCount++;
                    }} else {{
                        card.classList.remove('active');
                    }}
                }}
            }});
            
            document.getElementById('cat-count').textContent = catCount;
            
            // Update remove flag visual state
            const removeCheckbox = document.getElementById('to-remove');
            const removeContainer = document.getElementById('remove-flag-container');
            if (removeCheckbox && removeContainer) {{
                if (removeCheckbox.checked) {{
                    removeContainer.classList.add('checked');
                    document.getElementById('remove-flag-status').textContent = 'Sim';
                }} else {{
                    removeContainer.classList.remove('checked');
                    document.getElementById('remove-flag-status').textContent = 'Não';
                }}
            }}
            
            // Auto-save on change
            saveCurrentLabels();
        }}
        
        function updateProgress() {{
            // Only count labels for reviews that exist in current dataset
            const reviewIds = new Set(reviews.map(r => r.id));
            const labeledCount = Object.keys(labels).filter(id => reviewIds.has(id)).length;
            const total = reviews.length;
            const percentage = (labeledCount / total) * 100;
            
            document.getElementById('progress-text').textContent = `${{labeledCount}}/${{total}}`;
            document.getElementById('progress-fill').style.width = `${{Math.min(percentage, 100)}}%`;
        }}
        
        function clearCache() {{
            if (confirm('⚠️ Isso irá apagar TODO o progresso salvo. Tem certeza?')) {{
                localStorage.removeItem('review_labels');
                localStorage.removeItem('current_review_index');
                labels = {{}};
                currentIndex = 0;
                loadReview(0);
                updateProgress();
                alert('Cache limpo! Começando do zero.');
            }}
        }}
        
        function saveProgress() {{
            saveCurrentLabels();
            alert('Progresso salvo com sucesso!');
        }}
        
        function loadProgress() {{
            labels = JSON.parse(localStorage.getItem('review_labels') || '{{}}');
            currentIndex = parseInt(localStorage.getItem('current_review_index') || '0');
            loadReview(currentIndex);
            updateProgress();
            alert('Progresso carregado!');
        }}
        
        function exportData() {{
            saveCurrentLabels();
            
            // Prepare export data
            const exportData = reviews.map(review => {{
                const reviewLabels = labels[review.id] || {{}};
                const categoryData = reviewLabels.categories || {{}};
                
                // Build category columns
                const logistica = categoryData['Logística'] || '';
                const produto = categoryData['Produto'] || '';
                const atendimento = categoryData['Atendimento'] || '';
                const preco = categoryData['Preço'] || '';
                
                return {{
                    review_id: review.id,
                    review_text: review.text,
                    logistica_sentimento: logistica,
                    produto_sentimento: produto,
                    atendimento_sentimento: atendimento,
                    preco_sentimento: preco,
                    to_remove: reviewLabels.toRemove ? 'Sim' : 'Não'
                }};
            }});
            
            // Create CSV
            const headers = ['review_id', 'review_text', 'logistica_sentimento', 'produto_sentimento', 'atendimento_sentimento', 'preco_sentimento', 'to_remove'];
            const csvRows = [headers.join(',')];
            
            exportData.forEach(row => {{
                const values = [
                    row.review_id,
                    `"${{row.review_text.replace(/"/g, '""')}}"`,
                    row.logistica_sentimento,
                    row.produto_sentimento,
                    row.atendimento_sentimento,
                    row.preco_sentimento,
                    row.to_remove
                ];
                csvRows.push(values.join(','));
            }});
            
            const csv = csvRows.join('\\n');
            const blob = new Blob([csv], {{ type: 'text/csv;charset=utf-8;' }});
            const link = document.createElement('a');
            const url = URL.createObjectURL(blob);
            link.setAttribute('href', url);
            link.setAttribute('download', 'rotulacao_reviews.csv');
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            // Also export JSON
            const jsonBlob = new Blob([JSON.stringify(exportData, null, 2)], {{ type: 'application/json' }});
            const jsonLink = document.createElement('a');
            const jsonUrl = URL.createObjectURL(jsonBlob);
            jsonLink.setAttribute('href', jsonUrl);
            jsonLink.setAttribute('download', 'rotulacao_reviews.json');
            jsonLink.style.visibility = 'hidden';
            document.body.appendChild(jsonLink);
            jsonLink.click();
            document.body.removeChild(jsonLink);
            
            alert('Dados exportados com sucesso!');
        }}
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowLeft' && !e.shiftKey && !e.ctrlKey && !e.metaKey) {{
                e.preventDefault();
                previousReview();
            }} else if (e.key === 'ArrowRight' && !e.shiftKey && !e.ctrlKey && !e.metaKey) {{
                e.preventDefault();
                nextReview();
            }}
        }});
        
        // Initialize on load
        init();
    </script>
</body>
</html>"""
    
    # Write HTML file
    output_file = 'rotulacao_reviews.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML file generated: {output_file}")
    print(f"📊 Total reviews: {len(reviews)}")
    print(f"🌐 Open {output_file} in your browser to start labeling!")

if __name__ == "__main__":
    generate_labeling_html()

