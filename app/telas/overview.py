# SafeVision - Overview Page
import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import datetime, timedelta

def render_page():
    st.title("🛡️ Visão Geral do Sistema")
    st.markdown("Painel de controle e indicadores de detecção de Equipamentos de Proteção Individual (EPIs).")
    
    # 1. Checagem Dinâmica de Modelos na pasta do projeto
    yolo_exists = os.path.exists("pesos/yolo11s_best.pt")
    rtdetr_exists = os.path.exists("pesos/rtdetr_best.pt")
    modelos_ativos = sum([yolo_exists, rtdetr_exists])
    
    # 2. Indicadores Principais (KPIs)
    st.subheader("Indicadores Gerais de Operação")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Imagens Analisadas", "1,245") 
    col2.metric("Trabalhadores Identificados", "3,412", "12 hoje") 
    col3.metric("Taxa de Conformidade", "82.4%", "+2.1%") 
    col4.metric("Modelos Disponíveis", f"{modelos_ativos} de 2")
    
    st.divider()
    
    # 3. Construção dos Gráficos com Plotly
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("📈 Histórico de Monitoramento")
        st.write("Evolução de detecções nos últimos 7 dias.")
        
        # Gerando dados simulados para os últimos 7 dias
        hoje = datetime.today()
        datas = [(hoje - timedelta(days=i)).strftime('%d/%m') for i in range(6, -1, -1)]
        
        df_historico = pd.DataFrame({
            "Data": datas,
            "Trabalhadores": [45, 52, 60, 48, 55, 70, 82],
            "Inconformidades": [12, 10, 15, 8, 14, 12, 18]
        })
        
        # Gráfico de Linha Múltipla
        fig_linha = px.line(
            df_historico, 
            x="Data", 
            y=["Trabalhadores", "Inconformidades"],
            markers=True,
            color_discrete_map={"Trabalhadores": "#1f77b4", "Inconformidades": "#d62728"}
        )
        fig_linha.update_layout(legend_title_text="", xaxis_title="", yaxis_title="Quantidade")
        st.plotly_chart(fig_linha, use_container_width=True)
        
    with col_chart2:
        st.subheader("⚠️ Distribuição de Irregularidades")
        st.write("Proporção de ausência de EPIs detectada.")
        
        df_pizza = pd.DataFrame({
            "Tipo de Infração": ["Sem Capacete", "Sem Colete", "Faltam Ambos"],
            "Quantidade": [45, 30, 14]
        })
        
        # Gráfico de Rosca (Donut)
        fig_pizza = px.pie(
            df_pizza, 
            values="Quantidade", 
            names="Tipo de Infração", 
            hole=0.4, # Faz o gráfico virar uma rosca
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_pizza, use_container_width=True)
        
    # 4. Status de Integração dos Modelos
    st.divider()
    st.subheader("Status dos Motores de Inferência")
    
    if yolo_exists:
        st.success("🟢 **YOLO11s:** Carregado e pronto para uso (`pesos/yolo11s_best.pt`).")
    else:
        st.error("🔴 **YOLO11s:** Arquivo de pesos não encontrado.")
        
    if rtdetr_exists:
        st.success("🟢 **RT-DETR-L:** Carregado e pronto para uso (`pesos/rtdetr_best.pt`).")
    else:
        st.warning("🟡 **RT-DETR-L:** Arquivo de pesos não encontrado. Aguardando treinamento da Etapa 2.")