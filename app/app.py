import streamlit as st
import sys
import os

# A configuração da página DEVE ser o primeiro comando do Streamlit
st.set_page_config(page_title="SafeVision", page_icon="🛡️", layout="wide")

# --- INÍCIO DO CSS GLOBAL (Degradê e Ícones) ---
st.markdown("""
    <style>
    /* Importa os ícones vetoriais (flat) do Bootstrap para todo o projeto */
    @import url('https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css');

    /* Aplica um degradê executivo no fundo da área principal (Dark Slate) */
    .stApp {
        background: linear-gradient(135deg, #0E1117 0%, #1c212c 100%);
    }
    
    /* Aplica um degradê sutil na barra lateral */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111318 0%, #0a0c10 100%);
    }
    </style>
""", unsafe_allow_html=True)
# --- FIM DO CSS GLOBAL ---

# Adiciona o diretório raiz do projeto ao Path do Python para resolver os imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importações de todos os nossos módulos
from components.sidebar import render_sidebar
from telas import overview, detection, comparison, analytics, compliance, dataset, models, video

# Renderiza a barra lateral e captura a escolha do usuário
modulo_ativo = render_sidebar()

# Mapeamento limpo entre o nome do menu e a função de renderização correspondente
rotas = {
    "Visão Geral": overview.render_page,
    "Detecção em Imagem": detection.render_page,
    "Comparação em Imagem": comparison.render_page,
    "Telemetria em Vídeo": video.render_page,
    "Laudo de Conformidade": compliance.render_page,
    "Dashboard Analytics": analytics.render_page,
    "Dataset de Treino": dataset.render_page,
    "Modelos (Baseline)": models.render_page
}

# Roteador central dinâmico
if modulo_ativo in rotas:
    rotas[modulo_ativo]() # Os parênteses executam a função escolhida dinamicamente
else:
    st.title(modulo_ativo)
    st.info("Página de configurações gerais do sistema (Em breve).")