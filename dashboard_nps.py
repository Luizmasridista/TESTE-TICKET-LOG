import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import io
import google.generativeai as genai
import json
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Configuração da página
st.set_page_config(
    page_title="Dashboard NPS Corporativo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuração do Gemini AI
api_key = os.getenv('GOOGLE_API_KEY')
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    st.warning("⚠️ API Key do Google Gemini não configurada. Configure a variável de ambiente GOOGLE_API_KEY para usar a análise com IA.")
    model = None

# Inicializar estado da sessão para chat
if 'chat_active' not in st.session_state:
    st.session_state.chat_active = {}
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = {}

# Título principal
st.title("📊 Dashboard NPS Corporativo")

# Filtro de ordenação no cabeçalho
col_titulo, col_ordenacao = st.columns([3, 1])
with col_ordenacao:
    ordenacao_selecionada = st.selectbox(
        "🔄 Ordenação dos Dados",
        ["Tudo (Padrão)", "Maior Nota", "Menor Nota", "Mais Recentes", "Mais Antigos"],
        help="Selecione como os dados devem ser ordenados nos gráficos"
    )

st.markdown("---")

# Funções auxiliares
def calcular_classificacao_nps(nota):
    """Classifica a nota em Promotor, Neutro ou Detrator"""
    if nota >= 9:
        return "Promotor"
    elif nota >= 7:
        return "Neutro"
    else:
        return "Detrator"

def calcular_nps(df):
    """Calcula o NPS (Net Promoter Score)"""
    total = len(df)
    if total == 0:
        return 0
    
    promotores = len(df[df['classificacao_nps'] == 'Promotor'])
    detratores = len(df[df['classificacao_nps'] == 'Detrator'])
    
    nps = ((promotores - detratores) / total) * 100
    return round(nps, 2)

def processar_dados(df):
    """Processa os dados do Excel para análise"""
    # Converter colunas de data se necessário
    if 'data' in df.columns:
        df['data'] = pd.to_datetime(df['data'], errors='coerce')
        df['mes_ano'] = df['data'].dt.to_period('M')
    
    # Criar classificação NPS se não existir
    if 'nota' in df.columns and 'classificacao_nps' not in df.columns:
        df['classificacao_nps'] = df['nota'].apply(calcular_classificacao_nps)
    
    # Se já existe uma coluna 'classificacao', usar ela como base
    if 'classificacao' in df.columns and 'classificacao_nps' not in df.columns:
        df['classificacao_nps'] = df['classificacao']
    
    return df

# Funções do Sistema de IA
def extrair_indicadores_grafico(df, tipo_grafico):
    """Extrai indicadores técnicos específicos de cada gráfico"""
    indicadores = {
        "tipo_grafico": tipo_grafico,
        "total_registros": len(df),
        "periodo_analise": f"{df['data'].min().strftime('%m/%Y')} a {df['data'].max().strftime('%m/%Y')}" if 'data' in df.columns else "N/A"
    }
    
    if tipo_grafico == "linha_media_mensal":
        if 'mes_ano' in df.columns and 'nota' in df.columns:
            media_mensal = df.groupby('mes_ano')['nota'].mean()
            indicadores.update({
                "media_geral": round(df['nota'].mean(), 2),
                "tendencia": "crescente" if media_mensal.iloc[-1] > media_mensal.iloc[0] else "decrescente",
                "maior_media": round(media_mensal.max(), 2),
                "menor_media": round(media_mensal.min(), 2),
                "variacao_percentual": round(((media_mensal.iloc[-1] - media_mensal.iloc[0]) / media_mensal.iloc[0]) * 100, 2)
            })
    
    elif tipo_grafico == "barras_classificacao":
        if 'classificacao_nps' in df.columns:
            dist = df['classificacao_nps'].value_counts(normalize=True) * 100
            nps = calcular_nps(df)
            indicadores.update({
                "nps_score": nps,
                "promotores_pct": round(dist.get('Promotor', 0), 1),
                "neutros_pct": round(dist.get('Neutro', 0), 1),
                "detratores_pct": round(dist.get('Detrator', 0), 1),
                "classificacao_nps": "Excelente" if nps >= 75 else "Muito Bom" if nps >= 50 else "Razoável" if nps >= 0 else "Ruim"
            })
    
    return indicadores

def gerar_prompt_analise(indicadores):
    """Gera prompt estruturado para análise do Gemini"""
    tipo = indicadores['tipo_grafico']
    
    if tipo == "linha_media_mensal":
        prompt = f"""
        Analise os seguintes indicadores de um gráfico de linha que mostra a média mensal das notas NPS:
        
        📊 DADOS DO GRÁFICO:
        - Período: {indicadores['periodo_analise']}
        - Total de registros: {indicadores['total_registros']}
        - Média geral: {indicadores['media_geral']}
        - Tendência: {indicadores['tendencia']}
        - Maior média: {indicadores['maior_media']}
        - Menor média: {indicadores['menor_media']}
        - Variação percentual: {indicadores['variacao_percentual']}%
        
        Por favor, forneça:
        1. Uma análise concisa da tendência observada
        2. Insights sobre a performance do NPS
        3. Recomendações práticas para melhoria
        4. Alertas sobre pontos de atenção
        
        Mantenha a resposta objetiva e focada em ações.
        """
    
    elif tipo == "barras_classificacao":
        prompt = f"""
        Analise os seguintes indicadores de um gráfico de barras que mostra a distribuição das classificações NPS:
        
        📊 DADOS DO GRÁFICO:
        - Período: {indicadores['periodo_analise']}
        - Total de registros: {indicadores['total_registros']}
        - NPS Score: {indicadores['nps_score']}%
        - Promotores: {indicadores['promotores_pct']}%
        - Neutros: {indicadores['neutros_pct']}%
        - Detratores: {indicadores['detratores_pct']}%
        - Classificação: {indicadores['classificacao_nps']}
        
        Por favor, forneça:
        1. Avaliação do NPS atual e sua classificação
        2. Análise da distribuição entre promotores, neutros e detratores
        3. Estratégias para converter neutros em promotores
        4. Ações para reduzir o número de detratores
        
        Mantenha a resposta prática e acionável.
        """
    
    return prompt

def consultar_gemini(prompt):
    """Consulta a API do Gemini com o prompt fornecido"""
    if model is None:
        return "⚠️ API Key do Google Gemini não configurada. Configure a variável de ambiente GOOGLE_API_KEY para usar a análise com IA."
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Erro ao consultar IA: {str(e)}"

def renderizar_chat_interface(grafico_id, df, tipo_grafico):
    """Renderiza a interface de chat para um gráfico específico"""
    chat_key = f"chat_{grafico_id}"
    
    # Botão de IA
    if st.button("🤖 Análise IA", key=f"ai_btn_{grafico_id}", help="Clique para obter análise inteligente deste gráfico"):
        st.session_state.chat_active[chat_key] = True
        
        # Extrair indicadores e gerar análise automática
        indicadores = extrair_indicadores_grafico(df, tipo_grafico)
        prompt = gerar_prompt_analise(indicadores)
        
        with st.spinner("🤖 Analisando dados..."):
            analise = consultar_gemini(prompt)
        
        # Inicializar mensagens do chat
        if chat_key not in st.session_state.chat_messages:
            st.session_state.chat_messages[chat_key] = []
        
        st.session_state.chat_messages[chat_key].append({
            "role": "assistant",
            "content": analise
        })
    
    # Interface de chat ativa
    if st.session_state.chat_active.get(chat_key, False):
        st.markdown("---")
        
        # Container do chat com estilo moderno
        chat_container = st.container()
        
        with chat_container:
            st.markdown("### 💬 Chat com IA - Análise do Gráfico")
            
            # Exibir mensagens do chat
            if chat_key in st.session_state.chat_messages:
                for msg in st.session_state.chat_messages[chat_key]:
                    if msg["role"] == "assistant":
                        st.markdown(f"""<div style="
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white;
                            padding: 15px;
                            border-radius: 15px;
                            margin: 10px 0;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                        ">
                            🤖 <strong>IA:</strong><br>{msg['content']}
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""<div style="
                            background: #f0f2f6;
                            color: #333;
                            padding: 15px;
                            border-radius: 15px;
                            margin: 10px 0;
                            margin-left: 50px;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        ">
                            👤 <strong>Você:</strong><br>{msg['content']}
                        </div>""", unsafe_allow_html=True)
            
            # Input para nova mensagem
            col1, col2 = st.columns([4, 1])
            
            with col1:
                nova_mensagem = st.text_input(
                    "Digite sua pergunta:", 
                    key=f"input_{chat_key}",
                    placeholder="Ex: Como posso melhorar esses indicadores?"
                )
            
            with col2:
                if st.button("Enviar", key=f"send_{chat_key}"):
                    if nova_mensagem:
                        # Adicionar mensagem do usuário
                        st.session_state.chat_messages[chat_key].append({
                            "role": "user",
                            "content": nova_mensagem
                        })
                        
                        # Gerar resposta da IA
                        indicadores = extrair_indicadores_grafico(df, tipo_grafico)
                        contexto = f"Contexto do gráfico: {json.dumps(indicadores, indent=2)}"
                        prompt_completo = f"{contexto}\n\nPergunta do usuário: {nova_mensagem}"
                        
                        with st.spinner("🤖 Pensando..."):
                            resposta = consultar_gemini(prompt_completo)
                        
                        st.session_state.chat_messages[chat_key].append({
                            "role": "assistant",
                            "content": resposta
                        })
                        
                        st.rerun()
            
            # Botão para fechar chat
            if st.button("❌ Fechar Chat", key=f"close_{chat_key}"):
                st.session_state.chat_active[chat_key] = False
                st.rerun()

def criar_grafico_linha_media_mensal(df, ordenacao=None):
    """Cria gráfico de linha com média mensal das notas"""
    if 'mes_ano' not in df.columns or 'nota' not in df.columns:
        st.error("Dados insuficientes para criar o gráfico de linha. Verifique se existem colunas 'data' e 'nota'.")
        return None
    
    media_mensal = df.groupby('mes_ano')['nota'].mean().reset_index()
    media_mensal['mes_ano_str'] = media_mensal['mes_ano'].astype(str)
    
    # Aplicar ordenação se especificada
    if ordenacao == "Maior Nota":
        media_mensal = media_mensal.sort_values('nota', ascending=False)
    elif ordenacao == "Menor Nota":
        media_mensal = media_mensal.sort_values('nota', ascending=True)
    elif ordenacao == "Mais Recentes":
        media_mensal = media_mensal.sort_values('mes_ano', ascending=False)
    elif ordenacao == "Mais Antigos":
        media_mensal = media_mensal.sort_values('mes_ano', ascending=True)
    
    # Adicionar informações extras para tooltip mais claro
    contagem_mensal = df.groupby('mes_ano').size().reset_index(name='total_avaliacoes')
    media_mensal = media_mensal.merge(contagem_mensal, on='mes_ano')
    
    fig = px.line(
        media_mensal, 
        x='mes_ano_str', 
        y='nota',
        title='Média Mensal das Notas NPS',
        labels={'mes_ano_str': 'Mês/Ano', 'nota': 'Média da Nota'},
        markers=True,
        hover_data={'total_avaliacoes': True}
    )
    
    # Customizar tooltip para ser mais claro
    fig.update_traces(
        hovertemplate="<b>Período:</b> %{x}<br>" +
                      "<b>Média da Nota:</b> %{y:.2f}<br>" +
                      "<b>Total de Avaliações:</b> %{customdata[0]}<br>" +
                      "<extra></extra>"
    )
    
    fig.update_layout(
        xaxis_title="Período",
        yaxis_title="Média da Nota",
        hovermode='x unified'
    )
    
    return fig

def criar_grafico_barras_classificacao(df, ordenacao=None):
    """Cria gráfico de barras com distribuição das classificações por mês"""
    if 'mes_ano' not in df.columns or 'classificacao_nps' not in df.columns:
        st.error("Dados insuficientes para criar o gráfico de barras.")
        return None
    
    # Calcular percentuais e contagens por mês
    classificacao_mensal = df.groupby(['mes_ano', 'classificacao_nps']).size().unstack(fill_value=0)
    classificacao_percentual = classificacao_mensal.div(classificacao_mensal.sum(axis=1), axis=0) * 100
    classificacao_percentual = classificacao_percentual.reset_index()
    classificacao_percentual['mes_ano_str'] = classificacao_percentual['mes_ano'].astype(str)
    
    # Aplicar ordenação se especificada
    if ordenacao == "Mais Recentes":
        classificacao_percentual = classificacao_percentual.sort_values('mes_ano', ascending=False)
    elif ordenacao == "Mais Antigos":
        classificacao_percentual = classificacao_percentual.sort_values('mes_ano', ascending=True)
    elif ordenacao in ["Maior Nota", "Menor Nota"]:
        # Para gráfico de barras, ordenar por percentual de promotores
        if 'Promotor' in classificacao_percentual.columns:
            ascending = ordenacao == "Menor Nota"
            classificacao_percentual = classificacao_percentual.sort_values('Promotor', ascending=ascending)
    
    # Adicionar contagens absolutas para tooltip
    classificacao_contagem = classificacao_mensal.reset_index()
    classificacao_contagem['mes_ano_str'] = classificacao_contagem['mes_ano'].astype(str)
    
    fig = go.Figure()
    
    cores = {'Promotor': '#2E8B57', 'Neutro': '#FFD700', 'Detrator': '#DC143C'}
    
    for classificacao in ['Promotor', 'Neutro', 'Detrator']:
        if classificacao in classificacao_percentual.columns:
            # Preparar dados para tooltip mais informativo
            percentuais = classificacao_percentual[classificacao]
            contagens = classificacao_contagem[classificacao] if classificacao in classificacao_contagem.columns else [0] * len(percentuais)
            
            fig.add_trace(go.Bar(
                name=classificacao,
                x=classificacao_percentual['mes_ano_str'],
                y=percentuais,
                marker_color=cores[classificacao],
                customdata=contagens,
                hovertemplate="<b>Período:</b> %{x}<br>" +
                             f"<b>{classificacao}:</b> %{{y:.1f}}%<br>" +
                             "<b>Quantidade:</b> %{customdata} avaliações<br>" +
                             "<extra></extra>"
            ))
    
    fig.update_layout(
        title='Distribuição Percentual das Classificações NPS por Mês',
        xaxis_title='Período',
        yaxis_title='Percentual (%)',
        barmode='stack',
        hovermode='x unified'
    )
    
    return fig

def criar_grafico_pareto_motivos(df, ordenacao=None):
    """Cria gráfico de Pareto dos motivos NPS para detratores"""
    # Verificar se existe coluna de motivos (aceita ambos os nomes)
    coluna_motivos = None
    if 'motivos_nps' in df.columns:
        coluna_motivos = 'motivos_nps'
    elif 'motivo_nps' in df.columns:
        coluna_motivos = 'motivo_nps'
    
    if coluna_motivos is None or 'classificacao_nps' not in df.columns:
        st.error("Dados insuficientes para criar o gráfico de Pareto. Verifique se existe a coluna 'motivos_nps' ou 'motivo_nps'.")
        return None
    
    # Filtrar apenas detratores
    detratores = df[df['classificacao_nps'] == 'Detrator']
    
    if len(detratores) == 0:
        st.warning("Não há detratores nos dados para análise de Pareto.")
        return None
    
    # Contar motivos
    motivos_count = detratores[coluna_motivos].value_counts().reset_index()
    motivos_count.columns = ['motivo', 'count']
    
    # Aplicar ordenação se especificada (antes dos cálculos de percentual)
    if ordenacao == "Maior Nota":
        # Para Pareto, ordenar por quantidade (descendente)
        motivos_count = motivos_count.sort_values('count', ascending=False)
    elif ordenacao == "Menor Nota":
        # Para Pareto, ordenar por quantidade (ascendente)
        motivos_count = motivos_count.sort_values('count', ascending=True)
    elif ordenacao in ["Mais Recentes", "Mais Antigos"]:
        # Para Pareto, manter ordenação por frequência (padrão)
        motivos_count = motivos_count.sort_values('count', ascending=False)
    
    # Calcular percentuais, acumulados e informações contextuais
    total_detratores = motivos_count['count'].sum()
    motivos_count['percentual'] = (motivos_count['count'] / total_detratores) * 100
    motivos_count['percentual_acumulado'] = motivos_count['percentual'].cumsum()
    motivos_count['ranking'] = range(1, len(motivos_count) + 1)
    
    # Classificar impacto baseado na regra 80/20
    motivos_count['impacto'] = motivos_count['percentual_acumulado'].apply(
        lambda x: 'Crítico' if x <= 80 else 'Moderado'
    )
    
    # Adicionar informação sobre regra 80/20
    motivos_count['regra_8020'] = motivos_count['percentual_acumulado'].apply(
        lambda x: 'Top 80%' if x <= 80 else 'Demais 20%'
    )
    
    # Criar gráfico de Pareto
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Barras com tooltip melhorado
    fig.add_trace(
        go.Bar(
            x=motivos_count['motivo'],
            y=motivos_count['count'],
            name='Quantidade',
            marker_color='lightblue',
            customdata=list(zip(
                motivos_count['percentual'],
                motivos_count['ranking'],
                motivos_count['impacto'],
                motivos_count['regra_8020']
            )),
            hovertemplate="<b>%{x}</b><br>" +
                          "<b>📊 Quantidade:</b> %{y} detratores<br>" +
                          "<b>📈 Percentual Individual:</b> %{customdata[0]:.1f}%<br>" +
                          "<b>🏆 Ranking:</b> %{customdata[1]}º motivo mais frequente<br>" +
                          "<b>⚡ Nível de Impacto:</b> %{customdata[2]}<br>" +
                          "<b>🎯 Regra 80/20:</b> %{customdata[3]}<br>" +
                          "<b>📋 Total de Detratores:</b> " + str(total_detratores) + "<br>" +
                          "<extra></extra>"
        ),
        secondary_y=False,
    )
    
    # Linha do percentual acumulado com tooltip melhorado
    fig.add_trace(
        go.Scatter(
            x=motivos_count['motivo'],
            y=motivos_count['percentual_acumulado'],
            mode='lines+markers',
            name='% Acumulado',
            line=dict(color='red', width=2),
            marker=dict(size=6),
            customdata=list(zip(
                motivos_count['percentual'],
                motivos_count['ranking']
            )),
            hovertemplate="<b>%{x}</b><br>" +
                          "<b>📈 Percentual Acumulado:</b> %{y:.1f}%<br>" +
                          "<b>📊 Percentual Individual:</b> %{customdata[0]:.1f}%<br>" +
                          "<b>🏆 Posição:</b> %{customdata[1]}º no ranking<br>" +
                          "<b>💡 Análise:</b> " + 
                          ("Foco prioritário (80% dos problemas)" if motivos_count.loc[motivos_count.index[0], 'percentual_acumulado'] <= 80 else "Impacto secundário") + "<br>" +
                          "<extra></extra>"
        ),
        secondary_y=True,
    )
    
    # Configurar eixos
    fig.update_xaxes(title_text="Motivos NPS")
    fig.update_yaxes(title_text="Quantidade de Detratores", secondary_y=False)
    fig.update_yaxes(title_text="Percentual Acumulado (%)", secondary_y=True)
    
    fig.update_layout(
        title='Gráfico de Pareto - Motivos dos Detratores',
        hovermode='x unified'
    )
    
    return fig



def criar_treemap_submotivos_detratores(df, ordenacao=None):
    """Cria treemap dos submotivos de detratores"""
    # Verificar se existem as colunas necessárias
    if 'submotivo_nps' not in df.columns or 'classificacao_nps' not in df.columns:
        # Tentar usar 'classificacao' se 'classificacao_nps' não existir
        if 'classificacao' in df.columns:
            classificacao_col = 'classificacao'
        else:
            st.error("Dados insuficientes para criar o treemap. Verifique se existem as colunas 'submotivo_nps' e 'classificacao'.")
            return None
    else:
        classificacao_col = 'classificacao_nps'
    
    # Filtrar apenas detratores
    detratores = df[df[classificacao_col] == 'Detrator']
    
    if len(detratores) == 0:
        st.warning("Não há detratores nos dados para análise de submotivos.")
        return None
    
    # Contar submotivos e remover valores nulos
    submotivos_count = detratores['submotivo_nps'].value_counts().reset_index()
    submotivos_count.columns = ['submotivo', 'count']
    
    # Remover linhas com submotivos nulos ou vazios
    submotivos_count = submotivos_count.dropna(subset=['submotivo'])
    submotivos_count = submotivos_count[submotivos_count['submotivo'] != '']
    
    if len(submotivos_count) == 0:
        st.warning("Não há submotivos válidos para criar o treemap.")
        return None
    
    # Calcular percentuais e ranking
    submotivos_count['percentual'] = (submotivos_count['count'] / submotivos_count['count'].sum()) * 100
    submotivos_count['ranking'] = submotivos_count['count'].rank(method='dense', ascending=False).astype(int)
    
    # Aplicar ordenação se especificada
    if ordenacao == "Maior Nota":
        submotivos_count = submotivos_count.sort_values('count', ascending=False)
    elif ordenacao == "Menor Nota":
        submotivos_count = submotivos_count.sort_values('count', ascending=True)
    else:
        # Padrão: ordenar por quantidade (descendente)
        submotivos_count = submotivos_count.sort_values('count', ascending=False)
    
    # Adicionar informações contextuais
    total_detratores = submotivos_count['count'].sum()
    submotivos_count['impacto'] = submotivos_count['percentual'].apply(
        lambda x: 'Alto' if x >= 20 else 'Médio' if x >= 10 else 'Baixo'
    )
    
    # Criar treemap
    fig = px.treemap(
        submotivos_count,
        path=['submotivo'],
        values='count',
        title='Treemap - Submotivos dos Detratores',
        color='count',
        color_continuous_scale='Reds',
        custom_data=['percentual', 'ranking', 'impacto']
    )
    
    # Personalizar tooltip para ser mais amigável
    fig.update_traces(
        textinfo="label+value+percent parent",
        textfont_size=12,
        textposition="middle center",
        hovertemplate="<b>%{label}</b><br>" +
                      "<b>📊 Quantidade:</b> %{value} detratores<br>" +
                      "<b>📈 Percentual:</b> %{customdata[0]:.1f}% do total<br>" +
                      "<b>🏆 Ranking:</b> %{customdata[1]}º mais frequente<br>" +
                      "<b>⚡ Nível de Impacto:</b> %{customdata[2]}<br>" +
                      "<b>🎯 Total de Detratores:</b> " + str(total_detratores) + "<br>" +
                      "<extra></extra>"
    )
    
    fig.update_layout(
        font_size=12,
        margin=dict(t=50, l=25, r=25, b=25)
    )
    
    return fig

def criar_metricas_nps(df):
    """Cria as métricas principais de NPS"""
    # Verificar qual coluna de classificação usar
    if 'classificacao_nps' in df.columns:
        classificacao_col = 'classificacao_nps'
    elif 'classificacao' in df.columns:
        classificacao_col = 'classificacao'
    else:
        return None, None, None, None
    
    promotores = len(df[df[classificacao_col] == 'Promotor'])
    neutros = len(df[df[classificacao_col] == 'Neutro'])
    detratores = len(df[df[classificacao_col] == 'Detrator'])
    nps_geral = calcular_nps(df)
    
    return promotores, neutros, detratores, nps_geral

# Interface principal
st.sidebar.header("📁 Upload de Dados")

# Upload de arquivo
uploaded_file = st.sidebar.file_uploader(
    "Escolha um arquivo Excel",
    type=['xlsx', 'xls'],
    help="Faça upload do arquivo Excel com os dados de NPS"
)

# Carregar arquivo padrão se disponível
if uploaded_file is None:
    arquivo_padrao = "USAR ESTE.xlsx"
    try:
        df = pd.read_excel(arquivo_padrao)
        st.sidebar.success(f"📊 Arquivo padrão carregado: {arquivo_padrao}")
    except FileNotFoundError:
        st.sidebar.warning("⚠️ Arquivo padrão não encontrado. Faça upload de um arquivo Excel.")
        df = None
else:
    try:
        df = pd.read_excel(uploaded_file)
        st.sidebar.success("✅ Arquivo carregado com sucesso!")
    except Exception as e:
        st.sidebar.error(f"❌ Erro ao carregar arquivo: {str(e)}")
        df = None

if df is not None:
    # Processar dados
    df = processar_dados(df)
    
    # Aplicar ordenação selecionada
    if ordenacao_selecionada == "Maior Nota":
        df = df.sort_values('nota', ascending=False)
    elif ordenacao_selecionada == "Menor Nota":
        df = df.sort_values('nota', ascending=True)
    elif ordenacao_selecionada == "Mais Recentes":
        if 'data' in df.columns:
            df = df.sort_values('data', ascending=False)
    elif ordenacao_selecionada == "Mais Antigos":
        if 'data' in df.columns:
            df = df.sort_values('data', ascending=True)
    # "Tudo (Padrão)" mantém a ordenação original
    
    # Mostrar informações do dataset
    st.sidebar.markdown("### 📋 Informações do Dataset")
    st.sidebar.write(f"**Registros:** {len(df)}")
    st.sidebar.write(f"**Colunas:** {len(df.columns)}")
    
    # Filtros
    st.sidebar.markdown("### 🔍 Filtros")
    
    # Filtro por grupo econômico
    if 'grupo_economico_cliente' in df.columns:
        grupos = ['Todos'] + list(df['grupo_economico_cliente'].unique())
        grupo_selecionado = st.sidebar.selectbox("Grupo Econômico Cliente", grupos)
        if grupo_selecionado != 'Todos':
            df = df[df['grupo_economico_cliente'] == grupo_selecionado]
    
    # Filtro por manutenção
    if 'manutencao' in df.columns:
        manutencoes = ['Todos'] + list(df['manutencao'].unique())
        manutencao_selecionada = st.sidebar.selectbox("Manutenção", manutencoes)
        if manutencao_selecionada != 'Todos':
            df = df[df['manutencao'] == manutencao_selecionada]
    
    # Filtro por produto manutenção
    if 'produto_manutencao' in df.columns:
        produtos = ['Todos'] + list(df['produto_manutencao'].unique())
        produto_selecionado = st.sidebar.selectbox("Produto Manutenção", produtos)
        if produto_selecionado != 'Todos':
            df = df[df['produto_manutencao'] == produto_selecionado]
    
    # Métricas principais
    st.markdown("### 💎 Métricas Principais")
    
    promotores, neutros, detratores, nps_geral = criar_metricas_nps(df)
    
    if promotores is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="
                background: #ffffff;
                border-left: 4px solid #4CAF50;
                border-radius: 8px;
                padding: 24px 20px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
                margin-bottom: 16px;
                transition: all 0.3s ease;
            ">
                <div style="display: flex; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 1.5em; margin-right: 8px;">🟢</span>
                    <h4 style="color: #37474F; margin: 0; font-weight: 600; font-size: 0.95em; text-transform: uppercase; letter-spacing: 0.5px;">Promotores</h4>
                </div>
                <h2 style="color: #263238; margin: 0 0 8px 0; font-size: 2.8em; font-weight: 700; line-height: 1;">{promotores}</h2>
                <p style="color: #4CAF50; margin: 0; font-size: 1.1em; font-weight: 600;">{(promotores/len(df)*100):.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="
                background: #ffffff;
                border-left: 4px solid #FF9800;
                border-radius: 8px;
                padding: 24px 20px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
                margin-bottom: 16px;
                transition: all 0.3s ease;
            ">
                <div style="display: flex; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 1.5em; margin-right: 8px;">🟡</span>
                    <h4 style="color: #37474F; margin: 0; font-weight: 600; font-size: 0.95em; text-transform: uppercase; letter-spacing: 0.5px;">Neutros</h4>
                </div>
                <h2 style="color: #263238; margin: 0 0 8px 0; font-size: 2.8em; font-weight: 700; line-height: 1;">{neutros}</h2>
                <p style="color: #FF9800; margin: 0; font-size: 1.1em; font-weight: 600;">{(neutros/len(df)*100):.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="
                background: #ffffff;
                border-left: 4px solid #F44336;
                border-radius: 8px;
                padding: 24px 20px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
                margin-bottom: 16px;
                transition: all 0.3s ease;
            ">
                <div style="display: flex; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 1.5em; margin-right: 8px;">🔴</span>
                    <h4 style="color: #37474F; margin: 0; font-weight: 600; font-size: 0.95em; text-transform: uppercase; letter-spacing: 0.5px;">Detratores</h4>
                </div>
                <h2 style="color: #263238; margin: 0 0 8px 0; font-size: 2.8em; font-weight: 700; line-height: 1;">{detratores}</h2>
                <p style="color: #F44336; margin: 0; font-size: 1.1em; font-weight: 600;">{(detratores/len(df)*100):.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            cor_border = "#4CAF50" if nps_geral >= 0 else "#F44336"
            cor_valor = "#4CAF50" if nps_geral >= 0 else "#F44336"
            emoji_nps = "📈" if nps_geral >= 0 else "📉"
            
            st.markdown(f"""
            <div style="
                background: #ffffff;
                border-left: 4px solid {cor_border};
                border-radius: 8px;
                padding: 24px 20px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
                margin-bottom: 16px;
                transition: all 0.3s ease;
            ">
                <div style="display: flex; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 1.5em; margin-right: 8px;">{emoji_nps}</span>
                    <h4 style="color: #37474F; margin: 0; font-weight: 600; font-size: 0.95em; text-transform: uppercase; letter-spacing: 0.5px;">NPS Geral</h4>
                </div>
                <h2 style="color: #263238; margin: 0 0 8px 0; font-size: 2.8em; font-weight: 700; line-height: 1;">{nps_geral}%</h2>
                <p style="color: {cor_valor}; margin: 0; font-size: 1.1em; font-weight: 600;">{'Positivo' if nps_geral >= 0 else 'Negativo'}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Gráficos principais
    st.markdown("### 📊 Análise Temporal e Distribuição")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Média Mensal das Notas")
        fig_linha = criar_grafico_linha_media_mensal(df, ordenacao_selecionada)
        if fig_linha:
            st.plotly_chart(fig_linha, use_container_width=True)
        
        # Interface de IA para gráfico de linha
        renderizar_chat_interface("linha_media", df, "linha_media_mensal")
    
    with col2:
        st.markdown("### 📊 Distribuição das Classificações")
        fig_barras = criar_grafico_barras_classificacao(df, ordenacao_selecionada)
        if fig_barras:
            st.plotly_chart(fig_barras, use_container_width=True)
        
        # Interface de IA para gráfico de barras
        renderizar_chat_interface("barras_classificacao", df, "barras_classificacao")
    
    # Análise dos Detratores
    st.markdown("### 📉 Análise dos Detratores")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Pareto - Motivos dos Detratores")
        fig_pareto = criar_grafico_pareto_motivos(df, ordenacao_selecionada)
        if fig_pareto:
            st.plotly_chart(fig_pareto, use_container_width=True)
    
    with col2:
        st.markdown("#### 🗂️ Treemap - Submotivos dos Detratores")
        fig_treemap = criar_treemap_submotivos_detratores(df, ordenacao_selecionada)
        if fig_treemap:
            st.plotly_chart(fig_treemap, use_container_width=True)
    

    

    # Tabela de dados (opcional)
    if st.checkbox("🔍 Mostrar dados brutos"):
        st.markdown("### 📋 Dados Brutos")
        st.dataframe(df, use_container_width=True)

else:
    st.info("👆 Faça upload de um arquivo Excel para começar a análise.")
    
    # Informações sobre o formato esperado
    st.markdown("### 📋 Formato Esperado do Arquivo")
    st.markdown("""
    O arquivo Excel deve conter as seguintes colunas:
    
    - **data**: Data da avaliação
    - **nota**: Nota do NPS (0-10)
    - **motivos_nps**: Motivos da avaliação
    - **grupo_economico_cliente**: Grupo econômico do cliente
    - **manutencao**: Tipo de manutenção
    - **produto_manutencao**: Produto da manutenção
    
    **Classificação NPS:**
    - 🟢 **Promotores**: Notas 9-10
    - 🟡 **Neutros**: Notas 7-8
    - 🔴 **Detratores**: Notas 0-6
    """)

# Rodapé
st.markdown("---")
st.markdown("**Dashboard NPS Corporativo** - Desenvolvido com Streamlit 📊")