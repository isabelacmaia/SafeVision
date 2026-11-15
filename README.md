# 🛡️ SafeVision

Sistema de visão computacional em Python com interface web interativa para monitoramento e detecção automatizada de Equipamentos de Proteção Individual (EPIs) em canteiros de obras. 

O projeto compara o desempenho prático de modelos baseados em Redes Neurais Convolucionais (YOLO11) e Transformers (RT-DETR), utilizando uma heurística própria de Inclusão de Centroide Setorial para emitir laudos de conformidade.

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
```bash
git clone [https://github.com/SEU_USUARIO/SafeVision.git](https://github.com/SEU_USUARIO/SafeVision.git)
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
Para que as detecções funcionem, coloque os arquivos de pesos treinados (ex: yolo11s.pt e rtdetr_best.pt) dentro da pasta pesos/ na raiz do projeto.

## 🚀 Como Iniciar a Aplicação
O sistema foi construído sobre o framework Streamlit. Para iniciar o servidor local, certifique-se de estar na raiz do projeto (SafeVision/) e execute o comando abaixo no terminal:
streamlit run app/app.py

O sistema abrirá automaticamente no seu navegador padrão (geralmente no endereço http://localhost:8501).

## 🖥️ Como Utilizar
Navegue pelo menu lateral para acessar os diferentes módulos do sistema:

Visão Geral: Dashboard inicial com a introdução do projeto.

Comparação: Permite o upload de uma imagem do canteiro de obras para processamento simultâneo. Visualize o YOLO11s e o RT-DETR-L inferindo lado a lado para comparar a qualidade de detecção em tempo real.

Conformidade: O coração das regras de segurança. Faça o upload de uma imagem e o sistema aplicará a heurística espacial para cruzar as coordenadas dos EPIs com as sub-regiões corporais dos trabalhadores, gerando um laudo em tabela (Conforme, Parcial ou Risco Crítico).

Analytics: Painel executivo interativo com gráficos Plotly mostrando as métricas oficiais obtidas (mAP, Precisão, Revocação e Latência).

Modelos: Ficha técnica dinâmica (cache em memória) com o status dos modelos integrados à aplicação.
