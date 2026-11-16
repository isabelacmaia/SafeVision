import streamlit as st
import cv2
import tempfile
import os
import time
import pandas as pd
from ultralytics import YOLO, RTDETR

def render_page():
    st.markdown("<h1><i class='bi bi-film' style='color: #FF9900;'></i> Processamento de Vídeo</h1>", unsafe_allow_html=True)
    st.write("Teste a inferência em tempo real e obtenha um relatório consolidado ao final do vídeo.")

    modo = st.radio(
        "Selecione o modo de análise:", 
        ["YOLO11s (Foco em FPS)", "RT-DETR-L (Foco em Precisão)", "Comparação Lado a Lado (Mais lento)"], 
        horizontal=True
    )

    arquivo_video = st.file_uploader("Envie um arquivo de vídeo (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])

    if arquivo_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(arquivo_video.read())
        tfile.close()
        
        model_yolo, model_rtdetr = None, None
        
        try:
            if "YOLO" in modo or "Comparação" in modo:
                model_yolo = YOLO("pesos/yolo11s_best.pt")
            if "RT-DETR" in modo or "Comparação" in modo:
                model_rtdetr = RTDETR("pesos/rtdetr_best.pt")
        except Exception as e:
            st.error(f"Erro ao carregar os pesos dos modelos. Verifique a pasta 'pesos/'. Detalhe: {e}")
            return

        cap = cv2.VideoCapture(tfile.name)
        
        st.markdown("### <i class='bi bi-speedometer2' style='color: #FF9900;'></i> Telemetria em Tempo Real", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1: fps_placeholder = st.empty()
        with col2: pessoas_placeholder = st.empty()
        with col3: capacetes_placeholder = st.empty()
        with col4: coletes_placeholder = st.empty()
        
        st.markdown("---")
        
        if "Comparação" in modo:
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.markdown("<h4 style='text-align: center;'>YOLO11s</h4>", unsafe_allow_html=True)
                frame_yolo = st.empty()
            with col_v2:
                st.markdown("<h4 style='text-align: center;'>RT-DETR-L</h4>", unsafe_allow_html=True)
                frame_rtdetr = st.empty()
        else:
            frame_placeholder = st.empty()
            
        stop_button = st.button("Parar Processamento", type="primary")

        CLASS_PERSON, CLASS_HARDHAT, CLASS_VEST = 0, 1, 3

        # Estruturas para armazenar os IDs únicos e confianças ao longo do vídeo
        telemetria = {
            "YOLO": {"pessoas": set(), "capacetes": set(), "coletes": set(), "confs": []},
            "RT-DETR": {"pessoas": set(), "capacetes": set(), "coletes": set(), "confs": []}
        }

        # Função auxiliar para extrair dados da inferência
        def extrair_dados(resultados, modelo_nome):
            if resultados.boxes is not None and resultados.boxes.id is not None:
                classes = resultados.boxes.cls.cpu().numpy().astype(int)
                ids = resultados.boxes.id.cpu().numpy().astype(int)
                confs = resultados.boxes.conf.cpu().numpy()
                
                for cls, obj_id, conf in zip(classes, ids, confs):
                    telemetria[modelo_nome]["confs"].append(conf)
                    if cls == CLASS_PERSON: telemetria[modelo_nome]["pessoas"].add(obj_id)
                    elif cls == CLASS_HARDHAT: telemetria[modelo_nome]["capacetes"].add(obj_id)
                    elif cls == CLASS_VEST: telemetria[modelo_nome]["coletes"].add(obj_id)

        while cap.isOpened() and not stop_button:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret: break

            qtd_pessoas, qtd_capacetes, qtd_coletes = 0, 0, 0
            ann_yolo, ann_rtdetr = None, None
            
            # --- Inferência YOLO ---
            if model_yolo is not None:
                res_yolo = model_yolo.track(frame, persist=True, conf=0.45, verbose=False)[0]
                ann_yolo = cv2.cvtColor(res_yolo.plot(), cv2.COLOR_BGR2RGB)
                extrair_dados(res_yolo, "YOLO")
                
                if "YOLO" in modo or "Comparação" in modo:
                    if res_yolo.boxes is not None:
                        classes = res_yolo.boxes.cls.cpu().numpy().astype(int)
                        qtd_pessoas, qtd_capacetes, qtd_coletes = (classes==CLASS_PERSON).sum(), (classes==CLASS_HARDHAT).sum(), (classes==CLASS_VEST).sum()

            # --- Inferência RT-DETR ---
            if model_rtdetr is not None:
                res_rtdetr = model_rtdetr.track(frame, persist=True, conf=0.45, verbose=False)[0]
                ann_rtdetr = cv2.cvtColor(res_rtdetr.plot(), cv2.COLOR_BGR2RGB)
                extrair_dados(res_rtdetr, "RT-DETR")
                
                if "RT-DETR" in modo:
                    if res_rtdetr.boxes is not None:
                        classes = res_rtdetr.boxes.cls.cpu().numpy().astype(int)
                        qtd_pessoas, qtd_capacetes, qtd_coletes = (classes==CLASS_PERSON).sum(), (classes==CLASS_HARDHAT).sum(), (classes==CLASS_VEST).sum()

            # --- Atualiza Interface de Vídeo ---
            if "Comparação" in modo:
                frame_yolo.image(ann_yolo, channels="RGB", use_container_width=True)
                frame_rtdetr.image(ann_rtdetr, channels="RGB", use_container_width=True)
            elif "YOLO" in modo:
                frame_placeholder.image(ann_yolo, channels="RGB", use_container_width=True)
            elif "RT-DETR" in modo:
                frame_placeholder.image(ann_rtdetr, channels="RGB", use_container_width=True)

            # --- Atualiza as Métricas em Tempo Real ---
            end_time = time.time()
            fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
            fps_placeholder.metric("FPS", f"{fps:.1f}")
            pessoas_placeholder.metric("Trabalhadores", int(qtd_pessoas))
            capacetes_placeholder.metric("Capacetes", int(qtd_capacetes))
            coletes_placeholder.metric("Coletes", int(qtd_coletes))

        cap.release()
        os.unlink(tfile.name)
        
        # ==========================================================
        # RELATÓRIO FINAL CONSOLIDADO (Aparece quando o vídeo acaba)
        # ==========================================================
        st.success("✅ Processamento de vídeo concluído!")
        st.markdown("### <i class='bi bi-table' style='color: #FF9900;'></i> Relatório Final de Detecções Únicas", unsafe_allow_html=True)
        
        # Funções auxiliares para montar o relatório
        def calc_media_conf(lista): return sum(lista)/len(lista) if len(lista) > 0 else 0
        def len_set(modelo, chave): return len(telemetria[modelo][chave])
        
        dados_tabela = []
        
        if "YOLO" in modo or "Comparação" in modo:
            dados_tabela.append({
                "Modelo / Fonte": "YOLO11s",
                "Trabalhadores": len_set("YOLO", "pessoas"),
                "Capacetes": len_set("YOLO", "capacetes"),
                "Coletes": len_set("YOLO", "coletes"),
                "Certeza Média": f"{calc_media_conf(telemetria['YOLO']['confs']) * 100:.1f}%"
            })
            
        if "RT-DETR" in modo or "Comparação" in modo:
            dados_tabela.append({
                "Modelo / Fonte": "RT-DETR-L",
                "Trabalhadores": len_set("RT-DETR", "pessoas"),
                "Capacetes": len_set("RT-DETR", "capacetes"),
                "Coletes": len_set("RT-DETR", "coletes"),
                "Certeza Média": f"{calc_media_conf(telemetria['RT-DETR']['confs']) * 100:.1f}%"
            })
            
        if "Comparação" in modo:
            # Consolidação cruzada (assume o valor máximo detectado e a média das confianças)
            max_pessoas = max(len_set("YOLO", "pessoas"), len_set("RT-DETR", "pessoas"))
            max_capacetes = max(len_set("YOLO", "capacetes"), len_set("RT-DETR", "capacetes"))
            max_coletes = max(len_set("YOLO", "coletes"), len_set("RT-DETR", "coletes"))
            conf_cruzada = (calc_media_conf(telemetria['YOLO']['confs']) + calc_media_conf(telemetria['RT-DETR']['confs'])) / 2
            
            dados_tabela.append({
                "Modelo / Fonte": "Consolidado (Cruzamento)",
                "Trabalhadores": max_pessoas,
                "Capacetes": max_capacetes,
                "Coletes": max_coletes,
                "Certeza Média": f"{conf_cruzada * 100:.1f}%"
            })
            
        df_relatorio = pd.DataFrame(dados_tabela)
        
        # Renderiza a tabela de forma estilizada
        st.dataframe(df_relatorio, use_container_width=True, hide_index=True)
        
        # Mensagem final interpretativa
        if "Comparação" in modo:
            st.info(f"**Análise Inteligente:** Cruzando os dados de ambas as arquiteturas, o sistema conclui com **{conf_cruzada * 100:.1f}% de certeza global** que **{max_pessoas} trabalhador(es)** exclusivo(s) passaram pela cena, utilizando um total de **{max_capacetes} capacete(s)** e **{max_coletes} colete(s)** durante o período.")