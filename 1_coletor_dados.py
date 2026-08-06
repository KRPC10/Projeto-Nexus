import cv2
import pandas as pd
import os
from cvzone.HandTrackingModule import HandDetector

# --- CONFIGURAÇÕES ---
ARQUIVO_DADOS = "memoria_jutsus.csv"

CLASSES = {
    '1': 'GOJO', 
    '2': 'SUKUNA', 
    '3': 'FLECHA_FOGO', 
    '4': 'SIX_SEVEN', 
    '5': 'MAHITO', 
    '0': 'NENHUM'
}

def extrair_features(hands):
    """ Extrai e normaliza os 126 pontos matemáticos das mãos """
    vetor = []
    for i in range(2):
        if i < len(hands):
            lmList = hands[i]['lmList']
            pulso_x, pulso_y, pulso_z = lmList[0] 
            for pt in lmList:
                vetor.extend([pt[0] - pulso_x, pt[1] - pulso_y, pt[2] - pulso_z])
        else:
            vetor.extend([0] * 63)
    return vetor

def salvar_dados(label, features):
    """ Salva o vetor matemático e o nome do jutsu no arquivo CSV """
    df = pd.DataFrame([features + [label]])
    if not os.path.isfile(ARQUIVO_DADOS):
        df.to_csv(ARQUIVO_DADOS, header=False, index=False)
    else:
        df.to_csv(ARQUIVO_DADOS, mode='a', header=False, index=False)
    print(f"✅ [+1 Pose Salva] -> {label}")

# --- INICIALIZAÇÃO ---
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.7, maxHands=2)

# Variável para guardar o Jutsu selecionado no momento
modo_selecionado = "NENHUM"

print("\n" + "="*50)
print("📸 MODO DE CAPTURA DE DADOS INICIADO")
print("="*50)
print(" PASSO 1: Aperte 0 a 5 para ESCOLHER o Jutsu.")
print(" PASSO 2: Aperte 9 para SALVAR a pose.")
print(" [Q] - SAIR")
print("="*50 + "\n")

while True:
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    
    # Esqueleto ativado para treino
    hands, img = detector.findHands(img, draw=True)

    # --- INTERFACE VISUAL (HUD) ---
    # Mostra qual Jutsu está selecionado
    cv2.putText(img, f"ALVO ATUAL: {modo_selecionado}", (10, 40), 
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)
    
    # Mostra os controles
    cv2.putText(img, "TECLAS [0 a 5]: Escolher  |  TECLA [9]: Gravar Pose  |  [Q]: Sair", (10, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    cv2.imshow("Scanner de Energia Amaldicoada", img)

    # --- LEITURA DO TECLADO ---
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("Saindo do modo de captura...")
        break
        
    # Se apertar de 0 a 5, apenas TROCA o modo selecionado
    elif chr(key) in CLASSES:
        modo_selecionado = CLASSES[chr(key)]
        print(f"🔄 Modo alterado para: {modo_selecionado}")
        
    # Se apertar 9, SALVA a pose no modo que estiver selecionado
    elif key == ord('9') and hands:
        features = extrair_features(hands)
        salvar_dados(modo_selecionado, features)
        
        # Pisca a tela de verde para dar feedback visual que salvou
        cv2.rectangle(img, (0,0), (1280,720), (0, 255, 0), 15)
        cv2.imshow("Scanner de Energia Amaldicoada", img)
        cv2.waitKey(50)

cap.release()
cv2.destroyAllWindows()