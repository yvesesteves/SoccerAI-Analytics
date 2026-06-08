from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

#tudo funcionando
def main():
    #ler video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    # inicializar o rastreador
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_Tracks(video_frames,
                                        read_from_stub=True,
                                        stub_path='stubs/track_stubs.pkl')
    # pegar posicoes de objetos
    tracker.add_position_to_tracks(tracks)

    # movimento estimado de came
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)

    # trajsformação
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    
    # Interpolar posições de bola
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # estimador de velocidade e distancia
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Atribuir equipes de jogadores
    team_assigner = TeamAssigner() 
    team_assigner.assign_team_color(video_frames[0],
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']): # 
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team] # colocando cor nos aglomerados
    

        #save cropped image of a player - salvando a imagem de um jogador pra distinguir o time pela cor da camisa
            # for track_id, player in tracks['players'][0].items():
            #     bbox = player['bbox']
            #     frame = video_frames[0]

            #     #crop bbox from frame
            #     cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

            #     #save cropped image
            #     cv2.imwrite(f'output_videos/cropped_image.jpg', cropped_image)

            #     break


    # atribuindo aquisicao da bola para o jogador
    player_assigner =PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox) #Atribuir bola ao jogador designado

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True #player esta com a bola
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])  #posse de bola do time
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)

    ## desenhando tracks de objetos
    output_video_frames = tracker.draw_annotations(video_frames, tracks,team_ball_control)

    ## desenhando camera de movimento
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    ## desenhar estimador de distancia e velocidade
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

    # salvar video
    save_video(output_video_frames, 'output_videos/output_video.avi')


if __name__ == '__main__':
    main()