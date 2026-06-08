# **SoccerAI Analytics - Computer Vision Pipeline**
## **Visão Geral**
- Este projeto é um pipeline completo de Visão Computacional e Ciência de Dados aplicado à análise tática e métricas de desempenho no futebol. Construído como etapa prática para o processo seletivo de Iniciação Científica (IC) em SoccerAI, o sistema processa vídeos brutos de partidas e extrai dados estruturados sobre a movimentação dos jogadores, posse de bola e cinemática (velocidade e distância).

## **Arquitetura e Pipeline de Dados**
**1.** Detecção e Rastreamento (Object Tracking): Utiliza o modelo YOLOv8 treinado de forma customizada para identificar jogadores, árbitros e a bola. O algoritmo ByteTrack é aplicado para manter a identidade (ID) de cada objeto ao longo do tempo.

**2.** Estimação de Movimento de Câmera (Optical Flow): Aplica o método de Lucas-Kanade para monitorar os pixels do fundo do estádio, calculando e descontando o movimento de pan e tilt da câmera da transmissão oficial.

**3.** Transformação de Perspectiva (Homografia): Converte a visão distorcida da câmera (achatada) para uma projeção 2D real (vista superior) através de cálculos de matriz, permitindo a conversão de pixels para metros reais.

**4.** Clusterização de Equipes (Machine Learning): Recorta a região correspondente à camisa de cada jogador detectado e aplica o algoritmo não supervisionado K-Means Clustering para separá-los dinamicamente entre Time 1 e Time 2, sem depender de regras estáticas de cores.

**5.** Cálculo de Cinemática e Posse:

  - Cinemática: Calcula a distância euclidiana percorrida e a velocidade em km/h de cada atleta.

 - Posse de Bola: Utiliza o cálculo da distância entre a base da bounding box do jogador e as coordenadas interpoladas da bola para definir atribuição de posse quadro a quadro.


## **Abordagem Matemática**
Para garantir a precisão das métricas extraídas no módulo speed_and_distance_estimator.py, a distância física percorrida entre o frame atual e o anterior no plano transformado é dada pela distância Euclidiana:

- d = raiz quadrada de ( (x2-x1)^2 + (y2-y1)^2 )

A velocidade escalar é então derivada pela razão do deslocamento pelo intervalo de tempo baseado na taxa de quadros do vídeo

 - v = (d/delta t) x 3.6

## **Estrutura do Projeto** 
```
📦 FOOTBALL_ANALYSIS
 ┣ 📂 input_videos/               # Arquivos de vídeo brutos (.mp4)
 ┣ 📂 models/                     # Pesos do modelo treinado (best.pt)
 ┣ 📂 output_videos/              # Vídeos processados com anotações e métricas
 ┣ 📂 stubs/                      # Checkpoints de rastreamento para desenvolvimento rápido (.pkl)
 ┣ 📜 main.py                     # Orquestrador do pipeline de dados
 ┣ 📜 tracker.py                  # Integração YOLO e ByteTrack
 ┣ 📜 team_assigner.py            # Lógica K-Means para divisão de times
 ┣ 📜 player_ball_assigner.py     # Lógica de atribuição de posse de bola
 ┣ 📜 camera_movement_estimator.py # Desconto de movimento de câmera (Optical Flow)
 ┣ 📜 view_transformer.py         # Conversão Pixel -> Metros (Homografia)
 ┣ 📜 speed_and_distance_estimator.py # Cálculos cinemáticos
 ┗ 📜 utils/                      # Funções auxiliares (I/O de vídeo e BBox matemáticas)
```
--- 



