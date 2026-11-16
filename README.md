# 🛡️ SafeVision

Sistema de visão computacional em Python com interface web interativa para monitoramento e detecção automatizada de Equipamentos de Proteção Individual (EPIs) em canteiros de obras. 

O projeto compara o desempenho prático de modelos baseados em Redes Neurais Convolucionais (YOLO11) e Transformers (RT-DETR). Além da detecção clássica, o sistema conta com rastreamento contínuo em vídeo (Tracking), geração de telemetria em tempo real e emissão de laudos de conformidade utilizando uma heurística própria de Inclusão de Centroide Setorial.

## ✨ Principais Funcionalidades
- **Telemetria de Vídeo:** Rastreamento em tempo real com fusão de dados e relatório consolidado de objetos únicos.
- **Embate Lado a Lado (Side-by-Side):** Comparação visual e estatística simultânea entre arquiteturas CNN e Transformer.
- **Laudo Automatizado:** Validação rigorosa de EPIs cruzando coordenadas dos equipamentos com sub-regiões anatômicas dos trabalhadores.

## 📂 Estrutura do Projeto (Arquitetura Limpa)

A arquitetura do software foi desenhada de forma modular, separando a interface da regra de negócio:

- `app/`: Contém o arquivo principal de roteamento (`app.py`), componentes visuais e a pasta `telas/` com as páginas da interface no padrão *Dispatcher*.
- `core/`: Lógica central de processamento, regras de negócio e o algoritmo de conformidade geométrica (`compliance.py`).
- `models/`: Classes controladoras de inferência para os modelos YOLO e RT-DETR.
- `pesos/`: Diretório de armazenamento local dos arquivos de parâmetros das redes neurais treinadas (`.pt`). *(Nota: os pesos são ignorados no Git devido ao tamanho)*.
- `notebooks/`: (Opcional) Notebooks utilizados para treinamento e experimentação no Google Colab.

## ⚙️ Pré-requisitos e Instalação

Certifique-se de ter o [Python 3.9+](https://www.python.org/) instalado. 

**1. Clone o repositório e acesse a pasta:**
git clone [https://github.com/isabelacmaia/SafeVision.git](https://github.com/isabelacmaia/SafeVision.git)
cd SafeVision

**2. Crie e ative um ambiente virtual (Recomendado):**
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate

**3. Instale as dependências:**
pip install streamlit ultralytics pandas plotly opencv-python

**4. Adicione os pesos dos modelos:**
Para que as detecções funcionem, coloque os arquivos de pesos treinados (ex: yolo11s_best.pt e rtdetr_best.pt) dentro da pasta pesos/ na raiz do projeto.

## 🚀 Como Iniciar a Aplicação
O sistema foi construído sobre o framework Streamlit. Para iniciar o servidor local, certifique-se de estar na raiz do projeto (SafeVision/) e execute o comando abaixo no terminal:
```bash
streamlit run app/app.py
O sistema abrirá automaticamente no seu navegador padrão (geralmente no endereço http://localhost:8501).

## 🖥️ Módulos do Sistema

- Navegue pelo menu lateral moderno para acessar as diferentes ferramentas:

- Visão Geral: Dashboard inicial com a introdução acadêmica do projeto.

- Detecção em Imagem: Inferência em imagens estáticas com controle de limiar de confiança, painel de telemetria individual e extração do payload JSON bruto.

- Comparação em Imagem: Embate lado a lado das arquiteturas. Visualize o YOLO11s e o RT-DETR-L inferindo simultaneamente, extraindo uma Tabela de Síntese com métricas cruzadas e medição de latência real.

- Telemetria em Vídeo (NOVO): Processamento de fluxos de vídeo (.mp4) com rastreamento (Tracking) quadro a quadro. Exibe um painel de telemetria dinâmico e gera um laudo final inteligente fundindo as certezas dos dois modelos para contabilizar objetos únicos na cena.

- Laudo de Conformidade: O coração das regras de segurança. O sistema aplica a heurística espacial para cruzar as coordenadas dos EPIs com as sub-regiões corporais dos trabalhadores, gerando um status avaliativo (Conforme, Parcial ou Risco Crítico).

- Dashboard Analytics: Painel executivo interativo com gráficos Plotly evidenciando os resultados oficiais de validação dos modelos (mAP, Precisão, Revocação e Latência).

- Dataset de Treino: Visão geral da amostragem e balanceamento dos dados utilizados.

- Modelos (Baseline): Ficha técnica dinâmica com a justificativa de recorte metodológico para modelos clássicos (como SSD e Faster R-CNN) que compõem o baseline teórico.