import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

DATASET_FILE = 'libras_dataset.csv'
MODEL_FILE = 'libras_model.pkl'

def extract_features(hand_landmarks):
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks

# --- Treinamento do Modelo ---
model = None

if os.path.exists(DATASET_FILE):
    print(f"Carregando dados de {DATASET_FILE}...")
    df = pd.read_csv(DATASET_FILE)

    # Separar features (X) e labels (y)
    X = df.drop('label', axis=1).values
    y = df['label'].values

    if len(np.unique(y)) > 1: # Verificar se há mais de uma classe para treinar
        print("Treinando o modelo de Machine Learning...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Usando MLPClassifier (Rede Neural Simples)
        model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, activation='relu', solver='adam', random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        print(f"Acurácia do modelo: {accuracy_score(y_test, y_pred):.2f}")

        # Salvar o modelo treinado
        import joblib
        joblib.dump(model, MODEL_FILE)
        print(f"Modelo salvo em {MODEL_FILE}")
    else:
        print("AVISO: Apenas uma classe encontrada no dataset. O modelo não será treinado adequadamente.")
        print("Por favor, colete dados para mais letras usando libras_data_collector.py")
else:
    print(f"AVISO: Arquivo {DATASET_FILE} não encontrado. O modelo não será treinado. \nPor favor, use libras_data_collector.py para coletar dados.")

# Carregar modelo se existir e não tiver sido treinado agora
if model is None and os.path.exists(MODEL_FILE):
    import joblib
    model = joblib.load(MODEL_FILE)
    print(f"Modelo carregado de {MODEL_FILE}")

def classify_hand_gesture(features):
    if model is None:
        return "Modelo Nao Treinado"
    if len(features) != 63: # 21 landmarks * 3 coordenadas
        return "Desconhecido (formato de features incorreto)"
    
    try:
        prediction = model.predict(np.array(features).reshape(1, -1))
        return prediction[0]
    except Exception as e:
        return f"Erro na predicao: {e}"

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1) # Espelhar a imagem para uma visualização mais intuitiva
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    recognized_letter = "Nenhuma"
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            features = extract_features(hand_landmarks)
            recognized_letter = classify_hand_gesture(features)

    cv2.putText(img, f"Letra: {recognized_letter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Libras Recognizer", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
