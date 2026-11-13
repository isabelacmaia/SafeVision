# SafeVision - Comparison Page
import streamlit as st
from PIL import Image
from models.yolo_detector import YoloDetector

# Tentamos importar o detector RT-DETR (que criaremos em breve)
try:
    from models.rtdetr_detector import RTDETRDetector
except ImportError:
    RTDETRDetector = None

@st.cache_resource
def carregar_modelos():
    """Carrega os modelos na memória de forma resiliente."""
    # Instancia o YOLO
    try:
        yolo = YoloDetector(model_path="pesos/yolo11s_best.pt")
    except FileNotFoundError:
        yolo = None
        
    # Instancia o RT-DETR
    try:
        if RTDETRDetector:
            rtdetr = RTDETRDetector(model_path="pesos/rtdetr_best.pt")
        else:
            rtdetr = None
    except FileNotFoundError:
        rtdetr = None
        
    return yolo, rtdetr

def render_page():
    st.title("⚖️ Comparação de Modelos")
    st.write("Compare as detecções do YOLO11s e do RT-DETR-L lado a lado na mesma imagem.")
    
    yolo_model, rtdetr_model = carregar_modelos()
    
    arquivo_imagem = st.file_uploader("Escolha uma imagem de teste (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if arquivo_imagem is not None:
        imagem = Image.open(arquivo_imagem)
        
        # Mostra a imagem original centralizada antes da inferência
        with st.expander("Ver Imagem Original"):
            st.image(imagem, use_container_width=True)
            
        confianca = st.slider("Limiar de Confiança Único (Aplica a ambos os modelos)", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
        
        if st.button("Executar Comparação Lado a Lado", type="primary"):
            st.divider()
            
            # Divide a tela exatamente ao meio
            col_yolo, col_rtdetr = st.columns(2)
            
            with st.spinner("Executando inferências simultâneas..."):
                
                # --- COLUNA ESQUERDA: YOLO11s ---
                with col_yolo:
                    st.subheader("🤖 YOLO11s")
                    if yolo_model:
                        img_yolo, dados_yolo = yolo_model.predict(imagem, conf_threshold=confianca)
                        st.image(img_yolo, use_container_width=True)
                        st.caption(f"Tempo de inferência estimado: {dados_yolo.speed['inference']:.1f}ms")
                    else:
                        st.warning("Modelo YOLO11s não encontrado. Treine-o e adicione em pesos/yolo11s_best.pt")
                        
                # --- COLUNA DIREITA: RT-DETR-L ---
                with col_rtdetr:
                    st.subheader("👁️ RT-DETR-L")
                    if rtdetr_model:
                        img_rtdetr, dados_rtdetr = rtdetr_model.predict(imagem, conf_threshold=confianca)
                        st.image(img_rtdetr, use_container_width=True)
                        st.caption(f"Tempo de inferência estimado: {dados_rtdetr.speed['inference']:.1f}ms")
                    else:
                        st.info("Modelo RT-DETR-L ainda não implementado/treinado.")
                        st.markdown("Aguardando arquivo `pesos/rtdetr_best.pt`.")