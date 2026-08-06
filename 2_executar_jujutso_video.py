import cv2
import numpy as np
import pygame
import os
import sys
import pandas as pd
from cvzone.HandTrackingModule import HandDetector
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
    
    k_ideal = 5
    if len(dados) < 5:
        k_ideal = len(dados)
        
    modelo_knn.n_neighbors = k_ideal
    modelo_knn.fit(X, y)
    print(f"🧠 [IA] Modelo treinado! ({len(dados)} posições salvas)")

def salvar_dados_e_retreinar(label, features):
    df = pd.DataFrame([features + [label]])
    df.to_csv(ARQUIVO_DADOS, mode='a', header=False, index=False)
    print(f"✅ Pose salva para: {label}")
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
# 2. CONFIGURAÇÃO DE ÁUDIOS E CAMINHOS DOS VÍDEOS
# =============================================================
pygame.mixer.init()

def carregar_som(caminho):
    try: return pygame.mixer.Sound(caminho)
    except: return None

# Áudios dos Jutsus
som_gojo = carregar_som("audio_gojo.mp3")
som_sukuna = carregar_som("audio_sukuna.mp3")
som_flecha = carregar_som("audio_flecha.mp3")
som_sixseven = carregar_som("audio_sixseven.mp3")
som_mahito = carregar_som("audio_mahito.mp3")

# Vídeos dos Jutsus
video_gojo = "video_gojo.mp4"
video_sukuna = "video_sukuna.mp4"
video_flecha = "video_flecha.mp4"
video_sixseven = "video_sixseven.mp4"
video_mahito = "video_mahito.mp4"

# =============================================================
# 3. FUNÇÃO QUE EXIBE O VÍDEO DO DOMÍNIO EM LOOP
# =============================================================
def ativar_dominio_video(cap_camera, gesto, som, caminho_video, texto_dominio, cor_texto):
    print(f"\n💥 EXPANSÃO DE DOMÍNIO ATIVADA: {gesto}")
    
    # 1. Toca o áudio
    if som: som.play()
    
    # 2. Prepara o leitor do vídeo
    cap_video = cv2.VideoCapture(caminho_video)
    fallback_ativo = False
    
    if not cap_video.isOpened():
        print(f"⚠️ AVISO: Arquivo de vídeo '{caminho_video}' não encontrado! Exibindo tela preta.")
        fallback_ativo = True

    # 3. Loop de Exibição do Vídeo
    while True:
        # Lemos a câmera web no vazio para evitar travamentos
        _, _ = cap_camera.read() 
        
        if not fallback_ativo:
            sucesso_vid, frame_video = cap_video.read()
            
            # Se o vídeo chegou no final, volta para o frame 0 (Loop)
            if not sucesso_vid:
                cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            frame_final = cv2.resize(frame_video, (1280, 720))
        else:
            frame_final = np.zeros((720, 1280, 3), np.uint8)

        # Textos de saída
        cv2.putText(frame_final, f"DOMINIO: {texto_dominio}", (40, 80), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, cor_texto, 3, cv2.LINE_AA)
        cv2.putText(frame_final, "[ PRESSIONE ESPACO PARA VOLTAR ]", (40, 680), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Projeto Nexus - Batalha Cinematografica", frame_final)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord(' '): 
            print("🛑 Domínio desfeito. Voltando à realidade.")
            if som: som.stop() 
            break
        elif key == ord('q'):
            sys.exit()

    if not fallback_ativo:
        cap_video.release()

# =============================================================
# 4. TELA INICIAL (VÍDEO ANIMADO E TRILHA SONORA)
# =============================================================
print("\n🎵 Carregando Tela Inicial...")

# --- A LINHA DO SEU SOM FOI CRIADA AQUI ---
som_inicio = carregar_som("audio_inicio.mp3") 
caminho_video_inicio = "tela_inicial.mp4" 

# Toca a música de fundo em loop infinito (loops=-1)
if som_inicio:
    som_inicio.play(loops=-1)

cap_inicio = cv2.VideoCapture(caminho_video_inicio)

if not cap_inicio.isOpened():
    print(f"⚠️ AVISO: Vídeo animado '{caminho_video_inicio}' não encontrado. Tela preta ativada.")
    tela_preta = np.zeros((720, 1280, 3), np.uint8)
    tela_preta[:] = (15, 15, 15)
    
    while True:
        cv2.imshow("Projeto Nexus - Batalha Cinematografica", tela_preta)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('j') or key == ord('J'):
            if som_inicio: som_inicio.stop() # Para a música quando o jogo começar
            break
        elif key == ord('q'):
            sys.exit()
else:
    # O Vídeo está rodando!
    while True:
        sucesso_ini, frame_inicio = cap_inicio.read()
        
        # Se o vídeo acabar, ele reinicia (Loop do GIF/MP4)
        if not sucesso_ini:
            cap_inicio.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        frame_inicio = cv2.resize(frame_inicio, (1280, 720))
        cv2.imshow("Projeto Nexus - Batalha Cinematografica", frame_inicio)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('j') or key == ord('J'):
            if som_inicio: som_inicio.stop() # Desliga a música ao entrar na câmera
            break
        elif key == ord('q'):
            sys.exit()
            
    cap_inicio.release()

# =============================================================
# 5. LOOP PRINCIPAL (LEITURA DA CÂMERA E IA)
# =============================================================
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.7, maxHands=2)

gesto_anterior = "NENHUM"
gesto_candidato = "NENHUM"
frames_confirmacao = 0
LIMITE_FRAMES = 12 
TAXA_CONFIANCA_MINIMA = 0.80

print("\n🎥 SISTEMA PRONTO. Faça um sinal e SEGURE firme para ativar!")

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
        
        probabilidades = modelo_knn.predict_proba([features_atuais])[0]
        indice_maior = np.argmax(probabilidades)
        porcentagem_confianca = probabilidades[indice_maior]
        gesto_ia_bruto = modelo_knn.classes_[indice_maior]
        
        if porcentagem_confianca >= TAXA_CONFIANCA_MINIMA:
            gesto_ia = gesto_ia_bruto
        else:
            gesto_ia = "NENHUM"
        
        if gesto_ia == gesto_candidato and gesto_ia != "NENHUM":
            frames_confirmacao += 1
        else:
            gesto_candidato = gesto_ia
            frames_confirmacao = 0

        # --- O GATILHO DA ATIVAÇÃO ---
        if frames_confirmacao >= LIMITE_FRAMES:
            gesto_validado = gesto_candidato

            if gesto_validado != "NENHUM" and gesto_validado != gesto_anterior:
                if gesto_validado == "GOJO":
                    ativar_dominio_video(cap, gesto_validado, som_gojo, video_gojo, "VAZIO INCOMENSURAVEL", (255, 255, 255))
                elif gesto_validado == "SUKUNA":
                    ativar_dominio_video(cap, gesto_validado, som_sukuna, video_sukuna, "SANTUARIO MALEVOLENTE", (0, 0, 255))
                elif gesto_validado == "FLECHA_FOGO":
                    ativar_dominio_video(cap, gesto_validado, som_flecha, video_flecha, "FLECHA DE FOGO (KAMINO)", (0, 165, 255))
                elif gesto_validado == "SIX_SEVEN":
                    ativar_dominio_video(cap, gesto_validado, som_sixseven, video_sixseven, "MODO: SIX SEVEN (67)", (255, 255, 0))
                elif gesto_validado == "MAHITO":
                    ativar_dominio_video(cap, gesto_validado, som_mahito, video_mahito, "AUTO-ENCARNACAO DA PERFEICAO", (255, 0, 255))
                
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
    cor_hud = (0, 255, 0) if (porcentagem_confianca >= TAXA_CONFIANCA_MINIMA and gesto_ia != "NENHUM") else (0, 150, 255) 

    cv2.putText(img, texto_hud, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, cor_hud, 2)
    cv2.putText(img, "Segure a pose firme. Errou? Aperte [0-5] para treinar.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Projeto Nexus - Batalha Cinematografica", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    elif chr(key) in CLASSES and features_atuais:
        label = CLASSES[chr(key)]
        salvar_dados_e_retreinar(label, features_atuais)
        cv2.rectangle(img, (0,0), (1280,720), (255, 0, 255), 15) 
        cv2.imshow("Projeto Nexus - Batalha Cinematografica", img)
        cv2.waitKey(50)

cap.release()
cv2.destroyAllWindows()