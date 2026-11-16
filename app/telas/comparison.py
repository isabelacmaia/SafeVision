import streamlit as st
from PIL import Image
import json
import pandas as pd
from models.yolo_detector import YoloDetector

# Tentamos importar o detector RT-DETR
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
    st.markdown("<h1><i class='bi bi-images' style='color: #FF9900;'></i> Comparação de Modelos</h1>", unsafe_allow_html=True)
    st.write("Compare as detecções do YOLO11s e do RT-DETR-L lado a lado na mesma imagem. Avalie a precisão de identificação, a capacidade de lidar com oclusões e a latência de inferência simultaneamente.")
    
    yolo_model, rtdetr_model = carregar_modelos()
    
    arquivo_imagem = st.file_uploader("Escolha uma imagem de teste (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if arquivo_imagem is not None:
        imagem = Image.open(arquivo_imagem)
        
        with st.expander("Ver Imagem Original"):
            st.image(imagem, use_container_width=True)
            
        confianca = st.slider(
            "Limiar de Confiança Único (Aplica a ambos os modelos)", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.45, 
            step=0.05,
            help="Define o nível mínimo de certeza exigido. Ao aplicar o mesmo limiar para ambas as redes, você pode avaliar matematicamente qual arquitetura é mais confiável e qual apresenta menos falsos positivos sob a mesma exigência."
        )
        
        if st.button("Executar Comparação Lado a Lado", type="primary"):
            st.divider()
            
            # Classes padrão do dataset
            CLASS_PERSON = 0
            CLASS_HARDHAT = 1
            CLASS_VEST = 3
            
            dados_relatorio = []
            
            col_yolo, col_rtdetr = st.columns(2)
            
            with st.spinner("Executando inferências simultâneas..."):
                
                # COLUNA ESQUERDA: YOLO11s
                with col_yolo:
                    st.markdown("<h4 style='text-align: center;'><i class='bi bi-lightning-charge' style='color: #FF9900;'></i> YOLO11s (CNN)</h4>", unsafe_allow_html=True)
                    if yolo_model:
                        img_yolo, dados_yolo = yolo_model.predict(imagem, conf_threshold=confianca)
                        st.image(img_yolo, use_container_width=True)
                        
                        # Extração de métricas
                        qtd_pessoas, qtd_capacetes, qtd_coletes = 0, 0, 0
                        if dados_yolo.boxes is not None:
                            classes = dados_yolo.boxes.cls.cpu().numpy().astype(int)
                            qtd_pessoas = (classes == CLASS_PERSON).sum()
                            qtd_capacetes = (classes == CLASS_HARDHAT).sum()
                            qtd_coletes = (classes == CLASS_VEST).sum()
                        latencia_yolo = dados_yolo.speed['inference']
                        
                        # Painel de Telemetria Individual
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Latência", f"{latencia_yolo:.1f} ms")
                        m2.metric("Pessoas", int(qtd_pessoas))
                        m3.metric("Capacetes", int(qtd_capacetes))
                        m4.metric("Coletes", int(qtd_coletes))
                        
                        with st.expander("Ver JSON de Saída (YOLO)"):
                            st.json(json.loads(dados_yolo.to_json()))
                            
                        dados_relatorio.append({
                            "Arquitetura": "YOLO11s (CNN)", 
                            "Latência (ms)": f"{latencia_yolo:.1f}", 
                            "Trabalhadores": qtd_pessoas, 
                            "Capacetes": qtd_capacetes, 
                            "Coletes": qtd_coletes
                        })
                    else:
                        st.warning("Modelo YOLO11s não encontrado. Treine-o e adicione em pesos/yolo11s_best.pt")
                        
                # COLUNA DIREITA: RT-DETR-L
                with col_rtdetr:
                    st.markdown("<h4 style='text-align: center;'><i class='bi bi-eye' style='color: #FF9900;'></i> RT-DETR-L (Transformer)</h4>", unsafe_allow_html=True)
                    if rtdetr_model:
                        img_rtdetr, dados_rtdetr = rtdetr_model.predict(imagem, conf_threshold=confianca)
                        st.image(img_rtdetr, use_container_width=True)
                        
                        # Extração de métricas
                        qtd_pessoas, qtd_capacetes, qtd_coletes = 0, 0, 0
                        if dados_rtdetr.boxes is not None:
                            classes = dados_rtdetr.boxes.cls.cpu().numpy().astype(int)
                            qtd_pessoas = (classes == CLASS_PERSON).sum()
                            qtd_capacetes = (classes == CLASS_HARDHAT).sum()
                            qtd_coletes = (classes == CLASS_VEST).sum()
                        latencia_rtdetr = dados_rtdetr.speed['inference']
                        
                        # Painel de Telemetria Individual
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Latência", f"{latencia_rtdetr:.1f} ms")
                        m2.metric("Pessoas", int(qtd_pessoas))
                        m3.metric("Capacetes", int(qtd_capacetes))
                        m4.metric("Coletes", int(qtd_coletes))
                        
                        with st.expander("Ver JSON de Saída (RT-DETR)"):
                            st.json(json.loads(dados_rtdetr.to_json()))
                            
                        dados_relatorio.append({
                            "Arquitetura": "RT-DETR-L (Transformer)", 
                            "Latência (ms)": f"{latencia_rtdetr:.1f}", 
                            "Trabalhadores": qtd_pessoas, 
                            "Capacetes": qtd_capacetes, 
                            "Coletes": qtd_coletes
                        })
                    else:
                        st.info("Modelo RT-DETR-L não encontrado. Aguardando arquivo `pesos/rtdetr_best.pt`.")
            
            # TABELA DE SÍNTESE COMPARATIVA
            if len(dados_relatorio) > 0:
                st.markdown("---")
                st.markdown("### <i class='bi bi-table' style='color: #FF9900;'></i> Síntese do Embate", unsafe_allow_html=True)
                df_relatorio = pd.DataFrame(dados_relatorio)
                st.dataframe(df_relatorio, use_container_width=True, hide_index=True)