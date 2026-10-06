import os
import math
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Baixando o modelo hand_landmarker.task...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,  # só uma mão
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

FRAMES_ESTAVEIS = 10  # quantos quadros seguidos para aceitar a jogada


def dist(a, b, w, h):
    # distância em pixels entre dois pontos
    return math.hypot((a.x - b.x) * w, (a.y - b.y) * h)


def dedo_esticado(mao, ponta, junta, w, h):
    # esticado se a ponta está mais longe do pulso do que a junta do meio
    return dist(mao[0], mao[ponta], w, h) > dist(mao[0], mao[junta], w, h)


def ler_jogada(mao, w, h):
    indicador = dedo_esticado(mao, 8, 6, w, h)
    medio = dedo_esticado(mao, 12, 10, w, h)
    anelar = dedo_esticado(mao, 16, 14, w, h)
    mindinho = dedo_esticado(mao, 20, 18, w, h)

    if not (indicador or medio or anelar or mindinho):
        return "PEDRA"
    if indicador and medio and anelar and mindinho:
        return "PAPEL"
    if indicador and medio and not anelar and not mindinho:
        return "TESOURA"
    return None


cap = cv2.VideoCapture(0)

ultima = None
contagem = 0
jogada_estavel = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # espelha antes de detectar, assim o texto na tela não fica invertido
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)

    atual = None
    if result.hand_landmarks:
        mao = result.hand_landmarks[0]
        atual = ler_jogada(mao, w, h)

        for landmark in mao:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    # só aceita a jogada depois de vários quadros iguais
    if atual == ultima:
        contagem += 1
    else:
        ultima = atual
        contagem = 0

    if contagem >= FRAMES_ESTAVEIS:
        jogada_estavel = atual

    # aqui você usa jogada_estavel (envio pro Arduino, placar etc.)
    texto = jogada_estavel if jogada_estavel else "..."
    cv2.putText(frame, texto, (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 255, 255), 3)

    cv2.imshow("Pedra, Papel e Tesoura", frame)
    if cv2.waitKey(5) & 0xFF in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()