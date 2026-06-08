# criar recursos pra ler e salvar o video

import cv2

def read_video(video_path): # retornar a lista de quadros para o video (o que é imagem)
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True: # captura de video e inicia lista vazia de frames, entra em loop para ler cada frame do video, 
        ret, frame = cap.read()# se não tiver mais frames, sai do loop e retorna a lista de frames
        if not ret:
            break
        frames.append(frame)
    return frames  #devolve lista de quadros do videop

def save_video(ouput_video_frames,output_video_path): # salvar o video a partir da lista de quadros, recebe a lista de quadros e o caminho do video de saída, define o codec de video e cria um objeto VideoWriter para escrever os quadros no arquivo de saída, itera sobre cada quadro na lista e escreve no arquivo usando o método write do VideoWriter, depois libera os recursos do VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))#24 frames por segundo
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()
