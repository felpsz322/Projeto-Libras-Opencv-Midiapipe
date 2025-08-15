import cv2
import math
import time
import numpy as np
import mediapipe as mp

# =========================
# Utilidades de Geometria
# =========================
def dist(p1, p2):
    return math.dist((p1.x, p1.y), (p2.x, p2.y))

def is_finger_extended(landmarks, tip_idx, pip_idx):
    # No sistema de imagem do OpenCV, y cresce para baixo.
    # Dedo estendido: ponta (tip) acima (menor y) do PIP.
    return landmarks[tip_idx].y < landmarks[pip_idx].y

def thumb_extended(landmarks, handedness_label):
    # Para o polegar, usamos a projeção horizontal (x) + pequena checagem vertical
    # Right: polegar estendido aponta para a direita (tip.x > ip.x), Left: esquerda.
    tip = landmarks[4]
    ip = landmarks[3]
    if handedness_label == "Right":
        horiz_open = tip.x > ip.x
    else:
        horiz_open = tip.x < ip.x
    # também exigir que a ponta não esteja muito baixa em relação à base (evita confusões)
    return horiz_open and (tip.y < landmarks[5].y + 0.05)

def fingers_state(landmarks, handedness_label):
    # Índices dos pontos por dedo (MediaPipe Hands)
    # Thumb: 4 (tip), 3 (ip)
    # Index: 8 (tip), 6 (pip)
    # Middle: 12 (tip), 10 (pip)
    # Ring: 16 (tip), 14 (pip)
    # Pinky: 20 (tip), 18 (pip)
    thumb = thumb_extended(landmarks, handedness_label)
    index = is_finger_extended(landmarks, 8, 6)
    middle = is_finger_extended(landmarks, 12, 10)
    ring = is_finger_extended(landmarks, 16, 14)
    pinky = is_finger_extended(landmarks, 20, 18)
    return dict(thumb=thumb, index=index, middle=middle, ring=ring, pinky=pinky)

def is_ok_gesture(landmarks):
    # “OK”: distância entre polegar e indicador pequena; demais dedos não todos estendidos
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    d = dist(thumb_tip, index_tip)
    # Normaliza pela largura da mão (distância entre MCP do indicador e do mínimo)
    ref = dist(landmarks[5], landmarks[17]) + 1e-6
    return (d / ref) < 0.25

def classify_gesture(landmarks, handedness_label):
    f = fingers_state(landmarks, handedness_label)

    # Punho (fist)
    if not any(f.values()):
        return "Punho "

    # Mão aberta
    if all(f.values()):
        return "Mão aberta"

    # Joinha (thumbs up): polegar estendido, outros retraídos
    if f["thumb"] and not (f["index"] or f["middle"] or f["ring"] or f["pinky"]):
        # opcional: reforçar que o polegar aponta "para cima"
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        if thumb_tip.y < thumb_ip.y:
            return "Joinha "
        else:
            return "Polegar "

  
    if f["index"] and f["middle"] and (not f["ring"]) and (not f["pinky"]) and (not f["thumb"]):
        # separação entre index/middle razoável
        if dist(landmarks[8], landmarks[12]) > 0.06:
            return "Paz "

    
    if f["index"] and (not f["middle"]) and (not f["ring"]) and f["pinky"]:
        return "Rock"

    if is_ok_gesture(landmarks):
        return "OK "

    if f["index"] and (not f["middle"]) and (not f["ring"]) and (not f["pinky"]) and (not f["thumb"]):
        return "Apontando "

    return "Desconhecido"

# =========================
# Pipeline de Vídeo
# =========================
def main():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Não foi possível acessar a webcam.")
        return

    # Ajuste opcional de resolução
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_t = time.time()
    fps = 0.0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # BGR -> RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            # Desenho + Classificação
            if results.multi_hand_landmarks:
                for hand_landmarks, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Desenhar landmarks
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

                    label = handed.classification[0].label  # "Left" ou "Right"
                    lm = hand_landmarks.landmark

                    gesto = classify_gesture(lm, label)

                    # Caixa de texto perto do punho (wrist = idx 0)
                    h, w = frame.shape[:2]
                    x = int(lm[0].x * w)
                    y = int(lm[0].y * h)

                    cv2.rectangle(frame, (x - 10, y - 40), (x + 210, y - 10), (0, 0, 0), -1)
                    cv2.putText(frame, f"{label} - {gesto}", (x - 5, y - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # FPS
            now = time.time()
            dt = now - prev_t
            prev_t = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else 0)

            cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 230, 20), 2, cv2.LINE_AA)

            cv2.imshow("Reconhecimento de Gestos - MediaPipe Hands", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC ou 'q'
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
