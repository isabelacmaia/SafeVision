# SafeVision - Compliance Page
import streamlit as st
import pandas as pd
from PIL import Image
from models.yolo_detector import YoloDetector
from core.compliance import ComplianceAnalyzer

@st.cache_resource
def carregar_detector():
    """Carrega o modelo base para a auditoria de conformidade."""
    try:
        return YoloDetector(model_path="pesos/yolo11s_best.pt")
    except FileNotFoundError:
        return None

def render_page():
    st.title("✅ Auditoria de Conformidade (EPIs)")
    st.write("Análise detalhada de segurança utilizando heurística espacial para associar equipamentos aos trabalhadores.")
    
    detector = carregar_detector()
    if detector is None:
        st.error("⚠️ Modelo YOLO11s não encontrado. A auditoria precisa do modelo treinado na pasta 'pesos/'.")
        return
        
    arquivo_imagem = st.file_uploader("Envie uma imagem do canteiro de obras para auditoria", type=["jpg", "jpeg", "png"])
    
    if arquivo_imagem is not None:
        imagem = Image.open(arquivo_imagem)
        
        # Mostra a imagem original menorzinhas para contexto
        st.image(imagem, caption="Imagem sob auditoria", use_container_width=True)
        
        if st.button("Executar Auditoria de Segurança", type="primary"):
            with st.spinner("Analisando imagem e calculando heurística espacial..."):
                # 1. Inferência Silenciosa (não mostramos a imagem anotada de cara)
                img_anotada, dados_brutos = detector.predict(imagem)
                
                # 2. Análise de Conformidade
                analyzer = ComplianceAnalyzer()
                relatorio = analyzer.analyze(dados_brutos)
                
                st.divider()
                
                if not relatorio:
                    st.info("Nenhum trabalhador detectado na cena para compor o laudo.")
                else:
                    # 3. Cálculo de Indicadores Chave (KPIs do Laudo)
                    total_trabalhadores = len(relatorio)
                    trabalhadores_seguros = sum(1 for t in relatorio if t["capacete"] and t["colete"])
                    taxa_conformidade = (trabalhadores_seguros / total_trabalhadores) * 100
                    
                    # 4. Renderização do Dashboard de Resumo
                    st.subheader("📊 Resumo da Auditoria")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Trabalhadores Identificados", total_trabalhadores)
                    col2.metric("Totalmente Conformes", trabalhadores_seguros)
                    
                    # Colore a métrica de conformidade dependendo do nível de segurança
                    if taxa_conformidade == 100:
                        col3.metric("Taxa de Conformidade", f"{taxa_conformidade:.1f}%", "Perfeito")
                    elif taxa_conformidade >= 50:
                        col3.metric("Taxa de Conformidade", f"{taxa_conformidade:.1f}%", "- Atenção", delta_color="off")
                    else:
                        col3.metric("Taxa de Conformidade", f"{taxa_conformidade:.1f}%", "- Risco Crítico", delta_color="inverse")
                    
                    st.divider()
                    
                    # 5. Transformando os dados crus em um DataFrame Pandas (Tabela)
                    st.subheader("📋 Relatório Detalhado por Trabalhador")
                    
                    dados_tabela = []
                    for trab in relatorio:
                        status_capacete = "✅ Presente" if trab["capacete"] else "❌ Ausente"
                        status_colete = "✅ Presente" if trab["colete"] else "❌ Ausente"
                        
                        # Define o status geral
                        if trab["capacete"] and trab["colete"]:
                            status_geral = "🟢 Conforme"
                        elif trab["capacete"] or trab["colete"]:
                            status_geral = "🟡 Parcial"
                        else:
                            status_geral = "🔴 Risco Crítico"
                            
                        dados_tabela.append({
                            "Identificação": f"Trabalhador {trab['id_trabalhador']}",
                            "Capacete": status_capacete,
                            "Colete": status_colete,
                            "Status Geral": status_geral
                        })
                        
                    df_relatorio = pd.DataFrame(dados_tabela)
                    
                    # Exibe a tabela interativa (permite ordenar as colunas com um clique)
                    st.dataframe(df_relatorio, use_container_width=True, hide_index=True)
                    
                    # 6. Evidência Visual Oculta (apenas se o auditor quiser ver as caixas)
                    with st.expander("Ver Evidência Visual (Imagem com Bounding Boxes)"):
                        st.image(img_anotada, use_container_width=True, caption="As caixas confirmam a leitura do relatório acima.")