# SafeVision - RT-DETR Detector
import os
from ultralytics import RTDETR

class RTDETRDetector:
    def __init__(self, model_path="pesos/rtdetr_best.pt"):
        self.model_path = model_path
        self.model = self._load_model()
        
    def _load_model(self):
        """Carrega os pesos do modelo RT-DETR. Levanta erro se o arquivo não existir."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Pesos não encontrados no caminho: {self.model_path}")
        # A única diferença estrutural é a chamada da classe RTDETR em vez de YOLO
        return RTDETR(self.model_path)
        
    def predict(self, image, conf_threshold=0.5):
        """
        Executa a inferência na imagem recebida usando a arquitetura Transformer.
        Mantemos a mesma assinatura do YoloDetector para garantir compatibilidade
        nas telas de Comparação e Detecção.
        """
        # Executa o modelo
        resultados = self.model(image, conf=conf_threshold)
        
        # Pega o primeiro resultado (pois passamos apenas 1 imagem por vez)
        resultado_atual = resultados[0]
        
        # Gera a matriz da imagem com as caixas e rótulos desenhados
        imagem_anotada = resultado_atual.plot()
        
        return imagem_anotada, resultado_atual