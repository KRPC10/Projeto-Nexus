import cv2
import numpy as np
import pygame
import os
import sys
import pandas as pd
import time
from cvzone.HandTrackingModule import HandDetector
from sklearn.neighbors import KNeighborsClassifier

# --- NOVAS BIBLIOTECAS PARA FONTES CUSTOMIZADAS ---
from PIL import ImageFont, ImageDraw, Image

# =============================================================
# 0. CARREGADOR DE FONTES CUSTOMIZADAS (ESTÉTICA ANIME)
# =============================================================
# Tenta carregar a fonte que você baixou. Se não achar, não vai travar o código.
try:
    fonte_gigante = ImageFont.truetype("fonte_anime.ttf", 250) # Tamanho do relógio
    fonte_dominio = ImageFont.truetype("fonte_anime.ttf", 80)  # Tamanho do nome do Jutsu
    fonte_hud = ImageFont.truetype("fonte_anime.ttf", 35)      # Tamanho dos avisos
    print("🎨 Fonte customizada carregada com sucesso!")
except Exception as e:
    print("⚠️ AVISO: 'fonte_anime.ttf' não encontrada na pasta. Baixe uma fonte .ttf para ter o efeito!")
    fonte_gigante = None
    fonte_dominio = None
    fonte_hud = None

def desenhar_texto_estilizado(img, texto, y, fonte_pil, cor_bgr, espessura_contorno=3, alinhar_centro=True, pos_x=50):
    """ Desenha textos usando fontes .ttf com contorno grosso profissional """
    if fonte_pil is None:
        # Fundo de emergência se a pessoa esquecer de baixar a fonte .ttf
        cv2.putText(img, texto, (pos_x, y), cv2.FONT_HERSHEY_DUPLEX, 2.0, cor_bgr, 3)
        return img
        
    # Converte imagem do OpenCV (BGR) para formato que o Pillow entende (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    cor_rgb = (cor_bgr[2], cor_bgr[1], cor_bgr[0]) # Inverte cores
    
    # Calcula a largura do texto para centralizar perfeitamente na tela de 1280px
    bbox = draw.textbbox((0, 0), texto, font=fonte_pil)
    largura_texto = bbox[2] - bbox[0]
    
    if alinhar_centro:
        x = (1280 - largura_texto) // 2
    else:
        x = pos_x
        
    # Desenha o contorno escuro (Sombra outline)
    for dx in range(-espessura_contorno, espessura_contorno+1):
        for dy in range(-espessura_contorno, espessura_contorno+1):
            if dx != 0 or dy != 0:
                draw.text((x+dx, y+dy), texto, font=fonte_pil, fill=(0,0,0))
                
    # Desenha a cor principal por cima
    draw.text((x, y), texto, font=fonte_pil, fill=cor_rgb)
    
    # Devolve para o formato do OpenCV
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# =============================================================
# 1. CONFIGURAÇÃO DE MACHINE LEARNING (O CÉREBRO)
# =============================================================
ARQUIVO_DADOS = "memoria_jutsus.csv"
CLASSES = {'1': 'GOJO', '2': 'SUKUNA', '3': 'FLECHA_FOGO', '4': 'SIX_SEVEN', '5': 'MAHITO', '0': 'NENHUM'}

modelo_knn = KNeighborsClassifier(n_neighbors=5)

if not os.path.isfile(ARQUIVO_DADOS):
    print("❌ ERRO: Arquivo 'memoria_jutsus.csv' não encontrado!")
    sys.exit()

def treinar_modelo():
    dados = pd.read_csv(ARQUIVO_DADOS, header=None)
    X = dados.iloc[:, :-1].values
    y = dados.iloc[:, -1].values
    k_ideal = 5
    if len(dados) < 5: k_ideal = len(dados)
    modelo_knn.n_neighbors = k_ideal
    modelo_knn.fit(X, y)

def salvar_dados_e_retreinar(label, features):
    df = pd.DataFrame([features + [label]])
    df.to_csv(ARQUIVO_DADOS, mode='a', header=False, index=False)
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

som_gojo = carregar_som("audio_gojo.mp3")
som_sukuna = carregar_som("audio_sukuna.mp3")
som_flecha = carregar_som("audio_flecha.mp3") 
som_sixseven = carregar_som("audio_sixseven.mp3")
som_mahito = carregar_som("audio_mahito.mp3")

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
    if som: som.play()
    
    cap_video = cv2.VideoCapture(caminho_video)
    fallback_ativo = False
    
    if not cap_video.isOpened(): fallback_ativo = True

    while True:
        _, _ = cap_camera.read() 
        if not fallback_ativo:
            sucesso_vid, frame_video = cap_video.read()
            if not sucesso_vid:
                cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame_final = cv2.resize(frame_video, (1280, 720))
        else:
            frame_final = np.zeros((720, 1280, 3), np.uint8)

        # Usa nossa nova função para escrever o nome do Domínio com estilo!
        frame_final = desenhar_texto_estilizado(frame_final, f"DOMINIO: {texto_dominio}", 80, fonte_dominio, cor_texto, 4, False, 40)
        frame_final = desenhar_texto_estilizado(frame_final, "[ ESPACO PARA VOLTAR ]", 650, fonte_hud, (255, 255, 255), 2, False, 40)

        cv2.imshow("Projeto Nexus - Batalha Cinematografica", frame_final)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord(' '): 
            if som: som.stop() 
            break
        elif key == ord('q'): sys.exit()

    if not fallback_ativo: cap_video.release()

# =============================================================
# 4. TELA INICIAL (VÍDEO ANIMADO E TRILHA SONORA)
# =============================================================
som_inicio = carregar_som("audio_inicio.mp3") 
caminho_video_inicio = "tela_inicial.mp4" 

if som_inicio: som_inicio.play(loops=-1)
cap_inicio = cv2.VideoCapture(caminho_video_inicio)

if not cap_inicio.isOpened():
    tela_preta = np.zeros((720, 1280, 3), np.uint8)
    while True:
        cv2.imshow("Projeto Nexus - Batalha Cinematografica", tela_preta)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('j') or key == ord('J'):
            if som_inicio: som_inicio.stop()
            break
        elif key == ord('q'): sys.exit()
else:
    while True:
        sucesso_ini, frame_inicio = cap_inicio.read()
        if not sucesso_ini:
            cap_inicio.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame_inicio = cv2.resize(frame_inicio, (1280, 720))
        cv2.imshow("Projeto Nexus - Batalha Cinematografica", frame_inicio)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('j') or key == ord('J'):
            if som_inicio: som_inicio.stop()
            break
        elif key == ord('q'): sys.exit()
    cap_inicio.release()

# =============================================================
# 5. LOOP PRINCIPAL (LEITURA DA CÂMERA E IA COM CRONÔMETRO)
# =============================================================
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.7, maxHands=2)

gesto_anterior = "NENHUM"
gesto_candidato = "NENHUM"
TAXA_CONFIANCA_MINIMA = 0.80

tempo_inicio_gesto = None
TEMPO_NECESSARIO = 5.0  

while True:
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1) 
    
    hands, img = detector.findHands(img, draw=False) 

    features_atuais = []
    gesto_ia = "NENHUM" 
    porcentagem_confianca = 0.0
    tempo_decorrido = 0.0

    if hands:
        features_atuais = extrair_features(hands)
        probabilidades = modelo_knn.predict_proba([features_atuais])[0]
        indice_maior = np.argmax(probabilidades)
        porcentagem_confianca = probabilidades[indice_maior]
        
        if porcentagem_confianca >= TAXA_CONFIANCA_MINIMA:
            gesto_ia = modelo_knn.classes_[indice_maior]
        
        if gesto_ia != "NENHUM":
            if gesto_ia == gesto_candidato:
                if tempo_inicio_gesto is None: tempo_inicio_gesto = time.time()
                else: tempo_decorrido = time.time() - tempo_inicio_gesto
            else:
                gesto_candidato = gesto_ia
                tempo_inicio_gesto = time.time()
        else:
            gesto_candidato = "NENHUM"
            tempo_inicio_gesto = None

        if tempo_decorrido >= TEMPO_NECESSARIO:
            gesto_validado = gesto_candidato
            if gesto_validado != gesto_anterior:
                if gesto_validado == "GOJO": ativar_dominio_video(cap, gesto_validado, som_gojo, video_gojo, "VAZIO INCOMENSURAVEL", (255, 255, 255))
                elif gesto_validado == "SUKUNA": ativar_dominio_video(cap, gesto_validado, som_sukuna, video_sukuna, "SANTUARIO MALEVOLENTE", (0, 0, 255))
                elif gesto_validado == "FLECHA_FOGO": ativar_dominio_video(cap, gesto_validado, som_flecha, video_flecha, "FLECHA DE FOGO (KAMINO)", (0, 165, 255))
                elif gesto_validado == "SIX_SEVEN": ativar_dominio_video(cap, gesto_validado, som_sixseven, video_sixseven, "MODO: SIX SEVEN (67)", (255, 255, 0))
                elif gesto_validado == "MAHITO": ativar_dominio_video(cap, gesto_validado, som_mahito, video_mahito, "AUTO-ENCARNACAO DA PERFEICAO", (255, 0, 255))
                
                gesto_anterior = "NENHUM"
                gesto_candidato = "NENHUM"
                tempo_inicio_gesto = None
                tempo_decorrido = 0.0
            else:
                gesto_anterior = gesto_validado

    else:
        gesto_candidato = "NENHUM"
        tempo_inicio_gesto = None

    # Efeito de Escurecimento Cinematográfico
    if gesto_candidato != "NENHUM" and tempo_inicio_gesto is not None:
        progresso = min(tempo_decorrido / TEMPO_NECESSARIO, 1.0)
        tela_escura = np.zeros_like(img)
        opacidade = progresso * 0.85 
        img = cv2.addWeighted(img, 1.0 - opacidade, tela_escura, opacidade, 0)

    # --- RENDERIZAÇÃO DOS TEXTOS COM A FONTE .TTF ---
    
    # 1. Avisos pequenos no canto
    cor_hud = (0, 255, 0) if (porcentagem_confianca >= TAXA_CONFIANCA_MINIMA and gesto_ia != "NENHUM") else (0, 150, 255) 
    img = desenhar_texto_estilizado(img, f"LEITURA IA: {gesto_ia} ({porcentagem_confianca*100:.0f}%)", 30, fonte_hud, cor_hud, 2, False, 20)
    
    # 2. Relógio Gigante e Nome do Jutsu no Centro!
    if gesto_candidato != "NENHUM" and tempo_inicio_gesto is not None:
        tempo_restante = max(0.0, TEMPO_NECESSARIO - tempo_decorrido)
        proporcao = tempo_restante / TEMPO_NECESSARIO
        cor_contagem = (0, int(255 * proporcao), 255) # Do amarelo pro vermelho
        
        # Desenha o tempo no centro da tela (y=200)
        img = desenhar_texto_estilizado(img, f"{tempo_restante:.1f}s", 200, fonte_gigante, cor_contagem, 10, True)
        
        # Desenha o nome do Jutsu abaixo do relógio (y=450)
        img = desenhar_texto_estilizado(img, f"CARREGANDO: {gesto_candidato}", 450, fonte_dominio, (255, 255, 255), 6, True)

    cv2.imshow("Projeto Nexus - Batalha Cinematografica", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    elif chr(key) in CLASSES and features_atuais:
        label = CLASSES[chr(key)]
        salvar_dados_e_retreinar(label, features_atuais)
        cv2.waitKey(50)

cap.release()
cv2.destroyAllWindows()