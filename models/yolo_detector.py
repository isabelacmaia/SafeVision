# SafeVision - YOLO Detector
import os
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path="pesos/yolo11s_best.pt"):
        self.model_path = model_path
        self.model = self._load_model()
        
    def _load_model(self):
        """Carrega os pesos do modelo. Levanta erro se o arquivo não existir."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Pesos não encontrados no caminho: {self.model_path}")
        return YOLO(self.model_path)
        
    def predict(self, image, conf_threshold=0.5):
        """
        Executa a inferência na imagem recebida.
        
        Args:
            image: Imagem aberta (PIL Image ou array OpenCV).
            conf_threshold: Limiar de confiança para filtrar detecções (default 50%).
            
        Returns:
            imagem_anotada: Imagem com as Bounding Boxes desenhadas.
            dados_brutos: Objeto de resultados do Ultralytics (necessário para a Etapa 5).
        """
        # Executa o modelo
        resultados = self.model(image, conf=conf_threshold)
        
        # Pega o primeiro resultado (pois passamos apenas 1 imagem por vez)
        resultado_atual = resultados[0]
        
        # Gera a matriz da imagem com as caixas e rótulos desenhados
        imagem_anotada = resultado_atual.plot()
        
        return imagem_anotada, resultado_atual