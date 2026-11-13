# SafeVision - Compliance Core
class ComplianceAnalyzer:
    def __init__(self):
        # Nomes das classes exatas mapeadas pelo dataset construction-ppe
        self.person_classes = ['Person']
        self.helmet_classes = ['helmet']
        self.vest_classes = ['vest']
        # Classes de violação explícitas que o dataset já traz
        self.violation_classes = ['no_helmet', 'no_vest']

    def _get_center(self, box):
        """Calcula o ponto central geométrico (X, Y) de uma Bounding Box."""
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return center_x, center_y

    def _is_in_upper_third(self, item_box, person_box):
        """
        Verifica se o centro geométrico do item (ex: capacete) 
        está dentro do terço superior da caixa do trabalhador (região da cabeça).
        """
        item_x, item_y = self._get_center(item_box)
        px1, py1, px2, py2 = person_box
        
        altura_trabalhador = py2 - py1
        limite_inferior_cabeca = py1 + (altura_trabalhador * 0.35)
        
        dentro_x = px1 <= item_x <= px2
        dentro_y = py1 <= item_y <= limite_inferior_cabeca
        
        return dentro_x and dentro_y

    def _is_in_torso(self, item_box, person_box):
        """
        Verifica se o centro do item (ex: colete) está na região central do trabalhador.
        """
        item_x, item_y = self._get_center(item_box)
        px1, py1, px2, py2 = person_box
        
        altura_trabalhador = py2 - py1
        # Colete geralmente fica entre os 20% e 70% da altura do corpo
        limite_superior_tronco = py1 + (altura_trabalhador * 0.20)
        limite_inferior_tronco = py1 + (altura_trabalhador * 0.70)
        
        dentro_x = px1 <= item_x <= px2
        dentro_y = limite_superior_tronco <= item_y <= limite_inferior_tronco
        
        return dentro_x and dentro_y

    def analyze(self, ultralytics_results):
        """
        Recebe os dados brutos (Result) do YOLO e cruza as coordenadas 
        para emitir o relatório de conformidade.
        """
        boxes = ultralytics_results.boxes.xyxy.cpu().numpy() # Coordenadas [x1, y1, x2, y2]
        classes = ultralytics_results.boxes.cls.cpu().numpy() # IDs das classes
        names = ultralytics_results.names # Dicionário de ID para Nome da classe
        
        trabalhadores = []
        capacetes = []
        coletes = []
        
        # 1. Separa as detecções em listas específicas
        for box, cls_id in zip(boxes, classes):
            class_name = names[int(cls_id)]
            if class_name in self.person_classes:
                trabalhadores.append(box)
            elif class_name in self.helmet_classes:
                capacetes.append(box)
            elif class_name in self.vest_classes:
                coletes.append(box)
                
        relatorio = []
        
        # 2. Faz o cruzamento espacial (Heurística)
        for i, t_box in enumerate(trabalhadores):
            status = {
                "id_trabalhador": i + 1,
                "capacete": False,
                "colete": False,
                "box": t_box.tolist()
            }
            
            # Verifica cada capacete encontrado na imagem
            for c_box in capacetes:
                if self._is_in_upper_third(c_box, t_box):
                    status["capacete"] = True
                    break # Achou um capacete para este trabalhador, vai pro próximo
                    
            # Verifica cada colete encontrado na imagem
            for v_box in coletes:
                if self._is_in_torso(v_box, t_box):
                    status["colete"] = True
                    break
                    
            relatorio.append(status)
            
        return relatorio