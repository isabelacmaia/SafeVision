import streamlit as st
import sys
import os

# A configuração da página DEVE ser o primeiro comando do Streamlit
st.set_page_config(page_title="SafeVision", page_icon="🛡️", layout="wide")

# Adiciona o diretório raiz do projeto ao Path do Python para resolver os imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importações de todos os nossos módulos
from components.sidebar import render_sidebar
from telas import overview, detection, comparison, analytics, compliance, dataset, models

# Renderiza a barra lateral e captura a escolha do usuário
modulo_ativo = render_sidebar()

# Mapeamento limpo entre o nome do menu e a função de renderização correspondente
rotas = {
    "Visão Geral": overview.render_page,
    "Detecção": detection.render_page,
    "Comparação": comparison.render_page,
    "Analytics": analytics.render_page,
    "Conformidade": compliance.render_page,
    "Dataset": dataset.render_page,
    "Modelos": models.render_page
}

# Roteador central dinâmico
if modulo_ativo in rotas:
    rotas[modulo_ativo]() # Os parênteses executam a função escolhida dinamicamente
else:
    st.title(modulo_ativo)
    st.info("Página de configurações gerais do sistema (Em breve).")