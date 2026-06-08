from ultralytics import YOLO
import supervision as sv 
import pickle
import os
import cv2
import numpy as np
import sys
import pandas as pd
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


# Object Detection & Object Tracking
# usa o YOLO para achar onde estão as pessoas e a bola, e usa um algoritmo (ByteTrack) para dar um "RG" (ID) para cada um
# assim, o jogador 59 no frame 1 continua sendo o jogador 59 no frame 2 (exemplo)
class Tracker:
    def __init__(self, model_path): # inicializar o rastreadoor
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(sekf,tracks): #posicao do jogador em relacao a bbox
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self,ball_positions): # detectar a bola em todos os frames
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions # posicoes original da bola

    def detect_Frames(self,frames): 
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detection_batch = self.model.predict(frames[i:i+batch_size], conf=0.1) # processar os quadros em lotes com nivel de confiança 0.1 bom o suficiente
            detections += detection_batch
        return detections

    def get_object_Tracks(self,frames, read_from_stub= False, stub_path=None): #rastreammento de objetos - dicionario
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f) #carregar os dados de rastreamento de um arquivo usando pickle, que permite serializar e desserializar objetos Python, converter objetos Python em um formato que pode ser salvo em um arquivo e posteriormente carregado de volta para a memória
            return tracks
        
        detections = self.detect_Frames(frames)

        tracks={
            "players": [],
            "referees": [],
            "ball": []
        }

        #0 é bola, 1 é goleiro, 2 é jogador, 3 é arbitro
        for frame_num, detection in enumerate(detections): #pecorrendo as deteccoes em um loop e colocando no indice da lista
                cls_names = detection.names
                cls_names_inv =  {v:k for k,v in cls_names.items()}
                # print(cls_names)
                #converter a supervision para detection format
                detection_supervision = sv.Detections.from_ultralytics(detection) 

                #convertendo goleiro para jogador objeto
                for object_ind, class_id in enumerate(detection_supervision.class_id):
                    if cls_names[class_id] == "goalkeeper":
                        detection_supervision.class_id[object_ind] = cls_names_inv["player"] #converter goleiro para jogador

                
                #track objetos
                detection_with_tracks = self.tracker.update_with_detections(detection_supervision) #atualizar o rastreamento com as detecções atuais, recebe as detecções convertidas e atualiza o estado do rastreador, associando as detecções atuais com os objetos rastreados anteriormente, usando algoritmos de associação para determinar quais detecções correspondem a quais objetos rastreados, e atribui IDs de rastreamento exclusivos a cada objeto para manter o acompanhamento ao longo do tempo
                
                tracks["players"].append({})
                tracks["referees"].append({})
                tracks["ball"].append({})

                for frame_detection in detection_with_tracks: # iterar sobre as detecções com rastreamento atualizado, cada detecção contém informações sobre a posição do objeto, a classe do objeto e o ID de rastreamento atribuído pelo rastreador
                    bbox = frame_detection[0].tolist() # converter a caixa delimitadora para uma lista, a caixa delimitadora é geralmente representada como um array ou tensor contendo as coordenadas do retângulo que envolve o objeto detectado, convertendo para uma lista torna mais fácil de manipular e armazenar as informações da caixa delimitadora
                    cls_id = frame_detection[3]
                    track_id = frame_detection[4]
                    
                    if cls_id == cls_names_inv["player"]:
                        tracks["players"][frame_num][track_id] = {"bbox":bbox} # adicionar o ID de rastreamento do jogador ao dicionário de jogadores para o quadro atual
                
                    if cls_id == cls_names_inv["referee"]:
                        tracks["referees"][frame_num][track_id] = {"bbox":bbox} # adicionar o ID de rastreamento do arbitro ao dicionário de arbitros para o quadro atual

                for frame_detection in detection_supervision:
                    bbox = frame_detection[0].tolist() # converter a caixa delimitadora para uma lista, a caixa delimitadora é geralmente representada como um array ou tensor contendo as coordenadas do retângulo que envolve o objeto detectado, convertendo para uma lista torna mais fácil de manipular e armazenar as informações da caixa delimitadora
                    cls_id = frame_detection[3]
                    
                    if cls_id == cls_names_inv["ball"]:
                        tracks["ball"][frame_num][1] = {"bbox":bbox} # adicionar a caixa delimitadora da bola ao dicionário de bolas para o quadro atual
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f) #salvar os dados de rastreamento em um arquivo usando pickle,que permite serializar e desserializar objetos Python,  converter objetos Python em um formato que pode ser salvo em um arquivo e posteriormente carregado de volta para a memória
        
        
        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3]) #fundo
        x_center,_ = get_center_of_bbox(bbox) #obter o centro da caixa delimitadora usando a função get_center_of_bbox, que calcula as coordenadas do centro com base nas coordenadas da caixa delimitadora
        width = get_bbox_width(bbox) #obter a largura da caixa delimitadora usando a função get_bbox_width, que calcula a largura subtraindo as coordenadas x1 e x2 da caixa delimitadora

        cv2.ellipse(
            frame,
            center=(x_center, y2), # coordenadas do centro da elipse
            axes=(int(width), int(0.35*width)),
            angle=0.0, # ângulo de rotação da elipse
            startAngle=-45, # ângulo inicial da elipse
            endAngle=235, # ângulo final da elipse
            color=color, # cor da elipse
            thickness=2, # espessura da linha da elipse
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height//2) +15
        y2_rect = (y2 + rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED) # desenhar um retângulo preenchido para o ID de rastreamento

            x1_text = x1_rect+12
            if track_id>99:
                x1_text -= 10
            
            cv2.putText(
                frame,
                f"{track_id}", # texto a ser desenhado, que é o ID de rastreamento convertido para string
                (int(x1_text), int(y1_rect)+15),
                cv2.FONT_HERSHEY_SIMPLEX, # tipo de fonte a ser usada para o texto
                0.6,
                (0,0,0),
                2
            )


        return frame


    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # retangulo semi transparente para aparecer a estatistica 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)#controle de bola do time 1
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)#controle de bola do time 2

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3) 
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3) 

        return frame


    def draw_annotations(self,video_frames,tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            #Draw players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255)) # definindo cor da elipse de cada time
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id) # desenhar uma elipse ao redor do jogador usando as coordenadas da caixa delimitadora, a cor azul (0,0,255) e o ID de rastreamento do jogador

                if player.get('has_ball',False):
                    frame = self.draw_traingle(frame, player["bbox"],(0,0,255)) #dando o triangulo vermelho ao jogador com a bola

            #Draw referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0,255,255))

            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))

            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)  # cntrole de bola da equipe

            output_video_frames.append(frame)


        return output_video_frames
