# SafeVision - Sidebar Component
import streamlit as st

def render_sidebar():
    st.sidebar.title("🛡️ SafeVision")
    st.sidebar.markdown("Detecção e Análise de EPIs")
    st.sidebar.divider()

    # Roteador do menu
    menu = [
        "Visão Geral", "Detecção", "Modelos", "Comparação", 
        "Analytics", "Dataset", "Conformidade", "Configurações"
    ]
    
    modulo_ativo = st.sidebar.radio("Navegação", menu)
    return modulo_ativo