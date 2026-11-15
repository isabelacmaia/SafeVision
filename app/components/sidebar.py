import streamlit as st
from streamlit_option_menu import option_menu

def render_sidebar():
    with st.sidebar:
        # Cabeçalho executivo com tipografia reduzida
        st.markdown("<h2 style='text-align: center; color: #FF9900; font-size: 24px; margin-bottom: 0;'>🛡️ SafeVision</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #A8A8A8; font-size: 12px; margin-top: 0;'>Monitoramento de EPIs</p>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Menu fluido com ícones
        modulo_ativo = option_menu(
            menu_title=None,
            options=[
                "Visão Geral", 
                "Detecção", 
                "Comparação", 
                "Analytics",
                "Conformidade", 
                "Dataset", 
                "Modelos"
            ],
            icons=[
                "house",           
                "camera-video",    
                "layout-split",    
                "graph-up",        
                "shield-check",    
                "database",        
                "cpu"              
            ],
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#FF9900", "font-size": "16px"}, # Ícone ligeiramente menor
                "nav-link": {
                    "font-size": "13px", # Letra do menu reduzida para um padrão SaaS mais limpo
                    "text-align": "left", 
                    "margin": "2px 0px", 
                    "--hover-color": "#333333"
                },
                "nav-link-selected": {
                    "background-color": "#FF9900", 
                    "color": "#111111", 
                    "font-weight": "bold"
                }
            }
        )
        
        st.markdown("---")
        st.markdown("<p style='text-align: center; color: #666; font-size: 11px;'>© 2026 SafeVision AI</p>", unsafe_allow_html=True)
        
    return modulo_ativo