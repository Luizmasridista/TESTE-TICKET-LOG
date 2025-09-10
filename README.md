# 📊 Dashboard NPS Corporativo

Dashboard interativo desenvolvido em Python com Streamlit para análise de Net Promoter Score (NPS) corporativo.

## 🚀 Funcionalidades

### Métricas Principais
- **Média mensal das notas NPS**
- **Percentual de cada classificação** (Promotor, Neutro, Detrator) por mês
- **NPS mensal e geral**

### Visualizações
- **📈 Gráfico de Linha**: Evolução da média mensal das notas
- **📊 Gráfico de Barras**: Distribuição percentual das classificações por mês
- **📉 Gráfico de Pareto**: Análise dos motivos dos detratores

### Filtros Interativos
- Grupo Econômico Cliente
- Manutenção
- Produto Manutenção

## 📋 Formato dos Dados

O arquivo Excel deve conter as seguintes colunas:

| Coluna | Descrição | Exemplo |
|--------|-----------|----------|
| `data` | Data da avaliação | 2024-01-15 |
| `nota` | Nota NPS (0-10) | 8 |
| `motivos_nps` | Motivos da avaliação | "Atendimento ruim" |
| `grupo_economico_cliente` | Grupo do cliente | "Grupo A" |
| `manutencao` | Tipo de manutenção | "Preventiva" |
| `produto_manutencao` | Produto | "Sistema X" |

### Classificação NPS
- 🟢 **Promotores**: Notas 9-10
- 🟡 **Neutros**: Notas 7-8  
- 🔴 **Detratores**: Notas 0-6

**Fórmula NPS**: `((Promotores - Detratores) / Total) × 100`

## 🛠️ Instalação

### 1. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 2. Configurar API Key do Google Gemini (Obrigatório para IA)
```bash
# Copie o arquivo de exemplo
cp .env.example .env

# Edite o arquivo .env e adicione sua API key
# GOOGLE_API_KEY=sua_api_key_aqui
```

**📋 Como obter a API Key:**
- Acesse: https://makersuite.google.com/app/apikey
- Faça login com sua conta Google
- Clique em "Create API Key"
- Copie a chave gerada para o arquivo `.env`

### 3. Executar o Dashboard
```bash
streamlit run dashboard_nps.py
```

### 4. Acessar no Navegador
O dashboard será aberto automaticamente em: `http://localhost:8501`

## 📁 Estrutura do Projeto

```
📦 Dashboard NPS
├── 📄 dashboard_nps.py      # Aplicação principal
├── 📄 requirements.txt      # Dependências Python
├── 📄 README.md            # Documentação
└── 📊 USAR ESTE.xlsx       # Arquivo de dados padrão
```

## 🎯 Como Usar

1. **Executar o dashboard** com `streamlit run dashboard_nps.py`
2. **Upload de arquivo**: Use a barra lateral para carregar seu arquivo Excel
3. **Aplicar filtros**: Selecione os filtros desejados na barra lateral
4. **Analisar resultados**: Visualize as métricas e gráficos gerados

### Arquivo Padrão
O dashboard tentará carregar automaticamente o arquivo `USAR ESTE.xlsx` se nenhum arquivo for enviado.

## 📊 Interpretação dos Gráficos

### Gráfico de Linha - Média Mensal
- Mostra a evolução temporal da satisfação
- Identifica tendências de melhoria ou piora

### Gráfico de Barras - Distribuição
- Visualiza a proporção de cada tipo de cliente por mês
- Permite acompanhar mudanças na composição do NPS

### Gráfico de Pareto - Motivos dos Detratores
- Identifica os principais problemas que geram insatisfação
- Segue a regra 80/20: poucos motivos causam a maioria dos problemas
- Prioriza ações de melhoria

## 🔧 Dependências

- `streamlit==1.29.0` - Framework web para dashboards
- `pandas==2.1.4` - Manipulação de dados
- `plotly==5.17.0` - Gráficos interativos
- `openpyxl==3.1.2` - Leitura de arquivos Excel
- `numpy==1.24.3` - Computação numérica

## 🚨 Solução de Problemas

### Erro ao carregar arquivo Excel
- Verifique se o arquivo tem extensão `.xlsx` ou `.xls`
- Confirme se as colunas obrigatórias estão presentes
- Verifique se não há caracteres especiais nos nomes das colunas

### Gráficos não aparecem
- Verifique se existem dados suficientes após aplicar os filtros
- Confirme se as colunas `data` e `nota` estão no formato correto

### Performance lenta
- Para arquivos muito grandes (>100k linhas), considere filtrar os dados antes do upload
- Feche outras abas do navegador para liberar memória

## 📈 Exemplo de Uso

1. Carregue seus dados de NPS
2. Aplique filtro por "Grupo Econômico Cliente" = "Grupo Premium"
3. Observe no gráfico de linha se a satisfação está melhorando
4. Analise no gráfico de Pareto quais são os principais motivos de insatisfação
5. Use essas informações para planos de ação específicos

---

**Desenvolvido com ❤️ usando Python e Streamlit**