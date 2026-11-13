# SafeVision - Analytics Page
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def render_page():
    st.title("📊 Analytics e Resultados Experimentais")
    st.write("Análise estatística do desempenho dos modelos treinados sobre o conjunto de teste.")
    
    # 1. Criação do DataFrame com os resultados simulados dos experimentos
    # No futuro, você pode carregar isso de um arquivo CSV gerado na avaliação.
    dados_experimentos = {
        "Modelo": ["YOLO11s", "RT-DETR-L", "SSD", "Faster R-CNN"],
        "Precisao": [0.887, 0.912, 0.754, 0.830],
        "Recall": [0.737, 0.805, 0.612, 0.790],
        "F1_Score": [0.805, 0.855, 0.675, 0.809],
        "mAP_50": [0.797, 0.842, 0.640, 0.775],
        "mAP_50_95": [0.570, 0.620, 0.410, 0.530],
        "FPS": [57, 32, 45, 18] # Quadros por segundo (desempenho computacional)
    }
    df = pd.DataFrame(dados_experimentos)
    
    # Exibe a tabela de dados brutos
    with st.expander("Visualizar Tabela de Métricas Brutas", expanded=False):
        st.dataframe(df, use_container_width=True)

    # Cria abas para organizar os gráficos
    aba_metricas, aba_map, aba_desempenho = st.tabs([
        "Métricas de Classificação", 
        "Mean Average Precision (mAP)", 
        "Desempenho Computacional (FPS)"
    ])
    
    # --- ABA 1: Precisão, Recall e F1-Score ---
    with aba_metricas:
        st.subheader("Comparação de Precision, Recall e F1-Score")
        # Prepara os dados para o Plotly Express (derretendo a tabela para agrupar as barras)
        df_melted = df.melt(id_vars="Modelo", value_vars=["Precisao", "Recall", "F1_Score"], 
                            var_name="Métrica", value_name="Valor")
        
        # Plotly Express: Gráfico de barras agrupadas
        fig_metricas = px.bar(
            df_melted, 
            x="Modelo", 
            y="Valor", 
            color="Métrica", 
            barmode="group",
            text_auto='.2f', # Mostra o valor em cima da barra com 2 casas decimais
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_metricas.update_layout(yaxis_title="Pontuação (0 a 1)", yaxis_range=[0, 1])
        
        # Injeta o gráfico do Plotly dentro do Streamlit
        st.plotly_chart(fig_metricas, use_container_width=True)
        st.caption("*Precision mede os acertos sobre o que foi detectado. Recall mede o quanto dos objetos reais foram encontrados.*")

    # --- ABA 2: mAP (Mean Average Precision) ---
    with aba_map:
        st.subheader("Avaliação de Precisão Média (mAP)")
        
        # Plotly Graph Objects: Para gráficos mais customizados (como linhas sobrepostas)
        fig_map = go.Figure()
        fig_map.add_trace(go.Scatter(
            x=df["Modelo"], y=df["mAP_50"], 
            mode='lines+markers+text', 
            name='mAP@50',
            text=[f"{v:.3f}" for v in df["mAP_50"]],
            textposition="top center",
            line=dict(color='firebrick', width=3, dash='solid'),
            marker=dict(size=10)
        ))
        fig_map.add_trace(go.Scatter(
            x=df["Modelo"], y=df["mAP_50_95"], 
            mode='lines+markers+text', 
            name='mAP@50:95',
            text=[f"{v:.3f}" for v in df["mAP_50_95"]],
            textposition="bottom center",
            line=dict(color='royalblue', width=3, dash='dot'),
            marker=dict(size=10)
        ))
        fig_map.update_layout(yaxis_title="mAP Score", yaxis_range=[0, 1])
        st.plotly_chart(fig_map, use_container_width=True)
        st.caption("*mAP@50 é a métrica padrão com IoU de 50%. mAP@50:95 é muito mais rigorosa, exigindo alta precisão na caixa delimitadora.*")

    # --- ABA 3: Desempenho Computacional ---
    with aba_desempenho:
        st.subheader("Benchmark de Velocidade (Frames per Second)")
        
        # Gráfico de barras simples ordenado pela velocidade
        df_sorted = df.sort_values(by="FPS", ascending=True)
        fig_fps = px.bar(
            df_sorted, 
            x="FPS", 
            y="Modelo", 
            orientation='h', # Gráfico horizontal
            text="FPS",
            color="FPS",
            color_continuous_scale="Viridis"
        )
        fig_fps.update_layout(xaxis_title="Quadros processados por segundo (Maior é melhor)")
        st.plotly_chart(fig_fps, use_container_width=True)
        st.caption("*Testes executados sob as mesmas condições de hardware (GPU NVIDIA T4).*")