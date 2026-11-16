# SafeVision - Detection Page
import streamlit as st
from PIL import Image
from models.yolo_detector import YoloDetector
from core.compliance import ComplianceAnalyzer
import json

@st.cache_resource
def carregar_detector():
    """Faz o cache da instância da nossa classe YoloDetector na memória."""
    try:
        return YoloDetector(model_path="pesos/yolo11s_best.pt")
    except FileNotFoundError as e:
        st.error(f"⚠️ Erro de carregamento: {e}")
        return None

def render_page():
    st.title("🎯 Detecção de EPIs")
    st.write("Faça o upload de uma imagem para analisar a presença de equipamentos de proteção.")
    
    detector = carregar_detector()
    
    if detector is None:
        return

    arquivo_imagem = st.file_uploader("Escolha uma imagem (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if arquivo_imagem is not None:
        imagem = Image.open(arquivo_imagem)
        
        col_esq, col_dir = st.columns(2)
        with col_esq:
            st.subheader("Imagem Original")
            st.image(imagem, use_container_width=True)
        
        confianca = st.slider("Limiar de Confiança", min_value=0.0, max_value=1.0, value=0.45, help="Define o nível mínimo de certeza exigido da Inteligência Artificial. Valores mais baixos forçam o modelo a exibir mais objetos (aumentando o risco de falsos positivos), enquanto valores altos filtram a cena e exibem apenas as detecções com alta margem de segurança.")
        
        if st.button("Executar Inferência", type="primary"):
            with st.spinner("Processando Inteligência e Associação Espacial..."):
                # 1. Inferência Bruta
                imagem_anotada, dados_brutos = detector.predict(imagem, conf_threshold=confianca)
                
                with col_dir:
                    st.subheader("Resultado da Detecção")
                    st.image(imagem_anotada, use_container_width=True)
                
                # 2. Análise de Conformidade (Nossa heurística matemática)
                st.divider()
                st.subheader("📋 Laudo de Conformidade Individual")
                
                analyzer = ComplianceAnalyzer()
                relatorio = analyzer.analyze(dados_brutos)
                
                if not relatorio:
                    st.info("Nenhum trabalhador identificado nesta imagem para análise.")
                else:
                    for trab in relatorio:
                        id_trab = trab["id_trabalhador"]
                        has_capacete = trab["capacete"]
                        has_colete = trab["colete"]
                        
                        # Lógica de renderização de alertas
                        if has_capacete and has_colete:
                            st.success(f"👷 **Trabalhador {id_trab}**: Totalmente Conforme (Capacete e Colete detectados).")
                        elif has_capacete or has_colete:
                            ausente = "Colete" if has_capacete else "Capacete"
                            presente = "Capacete" if has_capacete else "Colete"
                            st.warning(f"⚠️ **Trabalhador {id_trab}**: Conformidade Parcial. {presente} detectado, mas falta **{ausente}**.")
                        else:
                            st.error(f"❌ **Trabalhador {id_trab}**: Não Conforme. Sem Capacete e sem Colete!")

                # Logs crus para auditoria/debug
                with st.expander("Ver JSON de Detecções (Logs brutos)"):
                    st.json(dados_brutos.to_json())