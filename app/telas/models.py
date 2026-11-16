# SafeVision - Models Page
import streamlit as st
import os

def render_page():
    st.title("🧠 Catálogo de Modelos")
    st.write("Especificações técnicas das arquiteturas de Visão Computacional implementadas no SafeVision.")
    
    # 1. Card: YOLO11s (O modelo base)
    st.subheader("Detectores Principais")
    
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### YOLO11s (You Only Look Once - Small)")
            st.markdown("**Arquitetura:** Rede Neural Convolucional (CNN) - Single-Stage")
            st.write(
                "O YOLO11s é a iteração mais recente da família YOLO. Destaca-se por processar a imagem inteira "
                "em uma única passagem de rede, oferecendo um equilíbrio excepcional entre velocidade e precisão. "
                "No SafeVision, atua como o modelo de linha de base para inferências em tempo real."
            )
            
            st.markdown("""
            * **Parâmetros:** ~9.4 Milhões
            * **Custo Computacional:** 21.7 GFLOPs
            * **Tamanho de Entrada:** 640x640 pixels
            """)
        with col2:
            if os.path.exists("pesos/yolo11s_best.pt"):
                st.success("✅ **Status:** Disponível")
                st.caption("Arquivo: `yolo11s_best.pt`")
            else:
                st.error("❌ **Status:** Pendente")
                
    # 2. Card: RT-DETR-L (O modelo avançado)
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### RT-DETR-L (Real-Time DEtection TRansformer - Large)")
            st.markdown("**Arquitetura:** Vision Transformer (ViT) com Mecanismo de Atenção")
            st.write(
                "Diferente do YOLO que utiliza convoluções, o RT-DETR utiliza mecanismos de atenção "
                "para focar em partes específicas da imagem. Ele elimina a necessidade do algoritmo NMS "
                "(Non-Maximum Suppression), sendo excelente para cenas com alta oclusão (trabalhadores sobrepostos)."
            )
            
            st.markdown("""
            * **Parâmetros:** ~32 Milhões
            * **Custo Computacional:** ~114 GFLOPs
            * **Tamanho de Entrada:** 640x640 pixels
            """)
        with col2:
            if os.path.exists("pesos/rtdetr_best.pt"):
                st.success("✅ **Status:** Disponível")
                st.caption("Arquivo: `rtdetr_best.pt`")
            else:
                st.warning("⏳ **Status:** Em Treinamento")

    st.divider()
    
    # 3. Modelos de Referência (Futuros)
    st.subheader("Modelos de Referência (Benchmark)")
    st.write("Arquiteturas clássicas planejadas para fins de comparação acadêmica de baseline.")
    
    col_ssd, col_rcnn = st.columns(2)
    
    with col_ssd:
        with st.container(border=True):
            st.markdown("#### <i class='bi bi-box-seam' style='color: #FF9900;'></i> SSD (Single Shot MultiBox Detector)", unsafe_allow_html=True)
            st.write("Modelo pioneiro de detecção em estágio único. Estudado na fundamentação como precursor arquitetural das redes modernas.")
            st.info("Status: Baseline Teórico")
            
    with col_rcnn:
        with st.container(border=True):
            st.markdown("#### <i class='bi bi-boxes' style='color: #FF9900;'></i> Faster R-CNN", unsafe_allow_html=True)
            st.write("Modelo clássico de dois estágios. Extremamente preciso, porém computacionalmente inviável para processamento de CFTV em tempo real.")
            st.info("Status: Baseline Teórico")