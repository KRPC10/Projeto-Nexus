import cv2
import numpy as np
import pygame
import os
import sys
import pandas as pd
from cvzone.HandTrackingModule import HandDetector
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from sklearn.neighbors import KNeighborsClassifier

# =============================================================
# 1. CONFIGURAÇÃO DE MACHINE LEARNING (O CÉREBRO)
# =============================================================
ARQUIVO_DADOS = "memoria_jutsus.csv"
CLASSES = {'1': 'GOJO', '2': 'SUKUNA', '3': 'FLECHA_FOGO', '4': 'SIX_SEVEN', '5': 'MAHITO', '0': 'NENHUM'}

modelo_knn = KNeighborsClassifier(n_neighbors=5)

if not os.path.isfile(ARQUIVO_DADOS):
    print("❌ ERRO: Arquivo 'memoria_jutsus.csv' não encontrado!")
    print("Rode o script '1_coletor_dados.py' primeiro para criar sua base de dados.")
    sys.exit()

def treinar_modelo():
    dados = pd.read_csv(ARQUIVO_DADOS, header=None)
    X = dados.iloc[:, :-1].values
    y = dados.iloc[:, -1].values
    
    # Ajusta a exigência da IA (K) dependendo de quantas fotos você tirou
    k_ideal = 5
    if len(dados) < 5:
        k_ideal = len(dados)
        print("⚠️ AVISO: Você tem poucas poses salvas. A precisão será baixa.")
        
    modelo_knn.n_neighbors = k_ideal
    modelo_knn.fit(X, y)
    print(f"🧠 [IA] Modelo treinado! ({len(dados)} posições salvas | Nível de Exigência: {k_ideal})")

def salvar_dados_e_retreinar(label, features):
    df = pd.DataFrame([features + [label]])
    df.to_csv(ARQUIVO_DADOS, mode='a', header=False, index=False)
    print(f"✅ [APRENDIZADO CONTÍNUO] Pose salva para: {label}")
    treinar_modelo()

def extrair_features(hands):
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

treinar_modelo()

# =============================================================
# 2. CONFIGURAÇÃO DE ÁUDIOS E IMAGENS 2D
# =============================================================
pygame.mixer.init()
def carregar_som(caminho):
    try: return pygame.mixer.Sound(caminho)
    except: return None

som_gojo = carregar_som("audio_gojo.mp3")
som_sukuna = carregar_som("audio_sukuna.mp3")
som_flecha = carregar_som("audio_flecha.mp3")
som_sixseven = carregar_som("audio_sixseven.mp3")
som_mahito = carregar_som("audio_mahito.mp3")

segmentor = SelfiSegmentation(model=1)

def carregar_fundo_2d(caminho_imagem, cor_fallback):
    if os.path.exists(caminho_imagem):
        fundo = cv2.imread(caminho_imagem)
        return cv2.resize(fundo, (1280, 720))
    else:
        fundo = np.zeros((720, 1280, 3), np.uint8)
        fundo[:] = cor_fallback
        return fundo

bg_gojo = carregar_fundo_2d("fundo_gojo.jpg", (50, 0, 20))
bg_sukuna = carregar_fundo_2d("fundo_sukuna.jpg", (0, 0, 160))
bg_flecha = carregar_fundo_2d("fundo_flecha.jpg", (0, 60, 220))
bg_sixseven = carregar_fundo_2d("fundo_sixseven.jpg", (255, 0, 255))
bg_mahito = carregar_fundo_2d("fundo_mahito.jpg", (90, 0, 90))
bg_inicio = carregar_fundo_2d("fundo_inicio.jpg", (15, 15, 15)) 

# =============================================================
# 3. FUNÇÃO DE PARALISIA E EXPANSÃO
# =============================================================
def ativar_dominio_congelado(img_frame_atual, cap_camera, gesto, som, fundo_escolhido, texto_dominio, cor_texto):
    print(f"\n💥 EXPANSÃO DE DOMÍNIO ATIVADA: {gesto}")
    if som: som.play()
    
    img_congelada = segmentor.removeBG(img_frame_atual, fundo_escolhido, cutThreshold=0.7)

    cv2.putText(img_congelada, f"DOMINIO: {texto_dominio}", (40, 80), 
                cv2.FONT_HERSHEY_DUPLEX, 1.2, cor_texto, 3, cv2.LINE_AA)
    cv2.putText(img_congelada, "[ PRESSIONE ESPACO PARA DESFAZER O DOMINIO ]", (40, 680), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    while True:
        sucesso, lixo = cap_camera.read() 
        cv2.imshow("Projeto Nexus - Batalha Cinematografica", img_congelada)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '): 
            if som: som.stop()
            break
        elif key == ord('q'):
            sys.exit()

# =============================================================
# 4. TELA INICIAL
# =============================================================
tela_inicio = bg_inicio.copy()
cv2.putText(tela_inicio, "SEJA BEM VINDO", (360, 300), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), 4, cv2.LINE_AA)
cv2.putText(tela_inicio, "APERTE A TECLA [ J ] PARA INICIAR", (330, 420), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 2, cv2.LINE_AA)

while True:
    cv2.imshow("Projeto Nexus - Batalha Cinematografica", tela_inicio)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('j') or key == ord('J'):
        break
    elif key == ord('q'):
        sys.exit()

# =============================================================
# 5. LOOP PRINCIPAL (ATIVAÇÃO AUTOMÁTICA)
# =============================================================
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.7, maxHands=2)

gesto_anterior = "NENHUM"
gesto_candidato = "NENHUM"
frames_confirmacao = 0
LIMITE_FRAMES = 10 
TAXA_CONFIANCA_MINIMA = 0.79 

while True:
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    
    hands, img = detector.findHands(img, draw=False) 

    features_atuais = []
    gesto_ia = "NENHUM" 
    porcentagem_confianca = 0.0

    if hands:
        features_atuais = extrair_features(hands)
        
        # Pega a certeza da IA
        probabilidades = modelo_knn.predict_proba([features_atuais])[0]
        indice_maior = np.argmax(probabilidades)
        porcentagem_confianca = probabilidades[indice_maior]
        gesto_ia_bruto = modelo_knn.classes_[indice_maior]
        
        # Se a confiança for alta, aceita o gesto
        if porcentagem_confianca >= TAXA_CONFIANCA_MINIMA:
            gesto_ia = gesto_ia_bruto
        else:
            gesto_ia = "NENHUM"
        
        # Lógica de confirmação (Barra de Casting)
        if gesto_ia == gesto_candidato and gesto_ia != "NENHUM":
            frames_confirmacao += 1
        else:
            gesto_candidato = gesto_ia
            frames_confirmacao = 0

        # Se a barra encheu, ativa na mesma hora!
        if frames_confirmacao >= LIMITE_FRAMES:
            gesto_validado = gesto_candidato

            if gesto_validado != "NENHUM" and gesto_validado != gesto_anterior:
                if gesto_validado == "GOJO": ativar_dominio_congelado(img, cap, gesto_validado, som_gojo, bg_gojo, "VAZIO INCOMENSURAVEL", (255, 255, 255))
                elif gesto_validado == "SUKUNA": ativar_dominio_congelado(img, cap, gesto_validado, som_sukuna, bg_sukuna, "SANTUARIO MALEVOLENTE", (0, 0, 255))
                elif gesto_validado == "FLECHA_FOGO": ativar_dominio_congelado(img, cap, gesto_validado, som_flecha, bg_flecha, "FLECHA DE FOGO (KAMINO)", (0, 165, 255))
                elif gesto_validado == "SIX_SEVEN": ativar_dominio_congelado(img, cap, gesto_validado, som_sixseven, bg_sixseven, "MODO: SIX SEVEN (67)", (255, 255, 0))
                elif gesto_validado == "MAHITO": ativar_dominio_congelado(img, cap, gesto_validado, som_mahito, bg_mahito, "AUTO-ENCARNACAO DA PERFEICAO", (255, 0, 255))
                
                # Reseta tudo depois de desfazer o Domínio (ao apertar Espaço)
                gesto_anterior = "NENHUM"
                gesto_candidato = "NENHUM"
                frames_confirmacao = 0
            else:
                gesto_anterior = gesto_validado

    else:
        gesto_candidato = "NENHUM"
        frames_confirmacao = 0
        gesto_anterior = "NENHUM"

    # --- HUD NA TELA ---
    texto_hud = f"LENDO: {gesto_ia} ({porcentagem_confianca*100:.0f}%) | Casting: [{frames_confirmacao}/{LIMITE_FRAMES}]"
    
    if porcentagem_confianca >= TAXA_CONFIANCA_MINIMA and gesto_ia != "NENHUM":
        cor_hud = (0, 255, 0) 
    else:
        cor_hud = (0, 150, 255) 

    cv2.putText(img, texto_hud, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, cor_hud, 2)
    cv2.putText(img, "Segure a pose. Errou? Aperte [0 a 5] para treinar a IA.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Projeto Nexus - Batalha Cinematografica", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif chr(key) in CLASSES and features_atuais:
        label = CLASSES[chr(key)]
        salvar_dados_e_retreinar(label, features_atuais)
        cv2.rectangle(img, (0,0), (1280,720), (255, 0, 255), 15) 
        cv2.imshow("Projeto Nexus - Batalha Cinematografica", img)
        cv2.waitKey(50)

cap.release()
cv2.destroyAllWindows()