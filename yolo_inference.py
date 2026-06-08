# criação de caixas de detecção de pessoas/objetos

from ultralytics import YOLO

model = YOLO('models/best.pt')

results = model.predict('input_videos/08fd33_4.mp4', save=True)
print(results[0])
print('--------------------------')
for box in results[0].boxes:
    print(box)

# ao executar o código com 'best.pt' ele irá criar um arquivo de vídeo com as caixas de
# bolas,goleiros,jogadores arbitros, etc 
    
