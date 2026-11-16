# SafeVision - Analytics Page
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def render_page():
    st.markdown("<h1><i class='bi bi-graph-up' style='color: #FF9900;'></i> Analytics e Desempenho (Resultados Reais)</h1>", unsafe_allow_html=True)
    st.write("Análise comparativa das métricas oficiais de validação geradas após o treinamento no Google Colab.")

    # 1. INSIRA AQUI OS SEUS DADOS REAIS DO COLAB
    dados_reais = {
        "Modelo": ["YOLO11s (CNN)", "RT-DETR-L (Transformer)"],
        "Precisão (%)": [88.7, 89.4],
        "Revocação (%)": [73.7, 80.1],
        "mAP@50 (%)": [79.7, 84.3],
        "mAP@50-95 (%)": [57.0, 61.6],
        "Latência (ms)": [17.4, 22.4]
    }
    
    df = pd.DataFrame(dados_reais)
    
    # 2. RENDERIZAÇÃO DOS GRÁFICOS
    st.markdown("<h3><i class='bi bi-table' style='color: #FF9900;'></i> Tabela Consolidada de Desempenho</h3>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.divider()
    
    col_grafico1, col_grafico2 = st.columns(2)
    
    with col_grafico1:
        st.markdown("### Qualidade de Detecção (mAP@50)")
        st.write("Métrica principal que define a assertividade do modelo em localizar os EPIs e os trabalhadores.")
        
        fig_map = px.bar(
            df, 
            x="Modelo", 
            y="mAP@50 (%)",
            color="Modelo",
            text="mAP@50 (%)",
            color_discrete_sequence=["#1f77b4", "#ff7f0e"]
        )
        fig_map.update_traces(textposition='outside')
        fig_map.update_layout(yaxis_range=[0, 100], showlegend=False)
        st.plotly_chart(fig_map, use_container_width=True)

    with col_grafico2:
        st.markdown("### Custo Computacional (Latência)")
        st.write("Tempo necessário para processar uma única imagem (quanto menor, melhor para tempo real).")
        
        fig_lat = px.bar(
            df, 
            x="Modelo", 
            y="Latência (ms)",
            color="Modelo",
            text="Latência (ms)",
            color_discrete_sequence=["#1f77b4", "#ff7f0e"]
        )
        fig_lat.update_traces(textposition='outside')
        fig_lat.update_layout(showlegend=False)
        st.plotly_chart(fig_lat, use_container_width=True)
        
    st.divider()
    
    # 3. Gráfico de Radar (Radar Chart) para Comparação Multidimensional
    st.markdown("<h3><i class='bi bi-radar' style='color: #FF9900;'></i> Comparação Multidimensional</h3>", unsafe_allow_html=True)
    st.write("O gráfico de radar ilustra o balanço (trade-off) entre os modelos.")
    
    fig_radar = go.Figure()
    
    categorias = ['Precisão (%)', 'Revocação (%)', 'mAP@50 (%)', 'mAP@50-95 (%)']
    
    for i, linha in df.iterrows():
        valores = [linha['Precisão (%)'], linha['Revocação (%)'], linha['mAP@50 (%)'], linha['mAP@50-95 (%)']]
        # Fecha o polígono repetindo o primeiro valor no final
        valores.append(valores[0])
        categorias_fechadas = categorias + [categorias[0]]
        
        fig_radar.add_trace(go.Scatterpolar(
            r=valores,
            theta=categorias_fechadas,
            fill='toself',
            name=linha['Modelo']
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[50, 100])),
        showlegend=True
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)