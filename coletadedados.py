import cv2
import mediapipe as mp
import numpy as np
import csv
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Nome do arquivo CSV para salvar os dados
DATASET_FILE = 'libras_dataset.csv'

# Cabeçalho do CSV (21 landmarks * 3 coordenadas (x,y,z) + label)
HEADER = ['label']
for i in range(21):
    HEADER.extend([f'x{i}', f'y{i}', f'z{i}'])

# Criar o arquivo CSV se não existir e escrever o cabeçalho
if not os.path.exists(DATASET_FILE):
    with open(DATASET_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)

def extract_features(hand_landmarks):
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks

cap = cv2.VideoCapture(0)

print("\nModo de Coleta de Dados de Libras ativado.")
print("Pressione uma tecla (a-z) para a letra correspondente e ENTER para salvar a pose.")
print("Pressione 'q' para sair.\n")

current_label = ''

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1) # Espelhar a imagem para uma visualização mais intuitiva
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            features = extract_features(hand_landmarks)
            
            # Exibir a label atual para coleta
            cv2.putText(img, f'Coletando: {current_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(img, 'Pressione uma tecla (a-z) e ENTER para salvar', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, 'Pressione Q para sair', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

            key = cv2.waitKey(1) & 0xFF

            if key >= ord('a') and key <= ord('z'):
                current_label = chr(key).upper()
                print(f"Pronto para coletar para a letra: {current_label}")
            elif key == 13: # Tecla ENTER
                if current_label and features:
                    row = [current_label] + features
                    with open(DATASET_FILE, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(row)
                    print(f"Dados para a letra {current_label} salvos com sucesso!")
                    current_label = '' # Limpar a label após salvar
                else:
                    print("Nenhuma label definida ou landmarks não detectadas. Tente novamente.")
            elif key == ord('q'):
                break
    else:
        cv2.putText(img, 'Nenhuma mao detectada', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img, 'Pressione Q para sair', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.imshow("Libras Data Collector", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()