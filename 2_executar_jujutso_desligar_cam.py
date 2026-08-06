import cv2
import numpy as np
import pygame
import os
import sys
import pandas as pd
from cvzone.HandTrackingModule import HandDetector
# REMOVIDO: SelfiSegmentationModule (não é mais necessário)
from sklearn.neighbors import KNeighborsClassifier

# =============================================================
# 1. CONFIGURAÇÃO DE MACHINE LEARNING (O CÉREBRO)
# =============================================================
ARQUIVO_DADOS = "memoria_jutsus.csv"
CLASSES = {'1': 'GOJO', '2': 'SUKUNA', '3': 'FLECHA_FOGO', '4': 'SIX_SEVEN', '5': 'MAHITO', '0': 'NENHUM'}

# A IA precisa de dados para treinar. O código de coleta deve ser rodado antes.
modelo_knn = KNeighborsClassifier(n_neighbors=5)

if not os.path.isfile(ARQUIVO_DADOS):
    print("❌ ERRO: Arquivo 'memoria_jutsus.csv' não encontrado!")
    print("Rode o script '1_coletor_dados.py' primeiro para criar sua base de dados.")
    sys.exit()

def treinar_modelo():
    dados = pd.read_csv(ARQUIVO_DADOS, header=None)
    X = dados.iloc[:, :-1].values
    y = dados.iloc[:, -1].values
    
    # Ajuste dinâmico do nível de exigência (K)
    k_ideal = 5
    if len(dados) < 5:
        k_ideal = len(dados)
        print("⚠️ AVISO: Poucas poses salvas. Precisão baixa.")
        
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
# 2. CONFIGURAÇÃO DE ÁUDIOS E IMAGENS DE FUNDO 2D
# =============================================================
# Prepara a biblioteca de áudio
pygame.mixer.init()

def carregar_som(caminho):
    try: return pygame.mixer.Sound(caminho)
    except: return None

# Caminhos dos sons (Coloque seus arquivos .mp3 na mesma pasta)
som_gojo = carregar_som("audio_gojo.mp3")
som_sukuna = carregar_som("audio_sukuna.mp3")
som_flecha = carregar_som("audio_flecha.mp3")
som_sixseven = carregar_som("audio_sixseven.mp3")
som_mahito = carregar_som("audio_mahito.mp3")

# Não precisamos mais do segmentor.

def carregar_imagem_dominio(caminho_imagem, cor_fallback):
    """ Tenta carregar a imagem 2D do domínio. Se não achar, cria cor sólida. """
    if os.path.exists(caminho_imagem):
        fundo = cv2.imread(caminho_imagem)
        return cv2.resize(fundo, (1280, 720)) # Redimensiona para o padrão
    else:
        # Cria um fundo colorido se a imagem faltar
        fundo = np.zeros((720, 1280, 3), np.uint8)
        fundo[:] = cor_fallback
        return fundo

# Caminhos das Imagens de Fundo (Coloque seus arquivos .jpg na mesma pasta)
img_dominio_gojo = carregar_imagem_dominio("gojo.gif", (50, 0, 20))    # Roxo Escuro
img_dominio_sukuna = carregar_imagem_dominio("fundo_sukuna.jpg", (0, 0, 160)) # Vermelho Sangue
img_dominio_flecha = carregar_imagem_dominio("fundo_flecha.jpg", (0, 60, 220)) # Laranja Fogo
img_dominio_sixseven = carregar_imagem_dominio("fundo_sixseven.jpg", (255, 0, 255)) # Neon
img_dominio_mahito = carregar_imagem_dominio("fundo_mahito.jpg", (90, 0, 90)) # Roxo/Cinza

# Imagem da tela inicial (Splash Screen)
bg_inicio = carregar_imagem_dominio("fundo_inicio.jpg", (15, 15, 15)) # Cinza Escuro

# =============================================================
# 3. FUNÇÃO QUE EXIBE APENAS O DOMÍNIO E TRAVA
# =============================================================
# Note que não recebemos mais o 'img_frame_atual', pois não vamos usar sua imagem.
def ativar_dominio_estatico(cap_camera, gesto, som, imagem_dominio_escolhida, texto_dominio, cor_texto):
    print(f"\n💥 EXPANSÃO DE DOMÍNIO ATIVADA: {gesto}")
    
    # 1. Toca o áudio
    if som: som.play()
    
    # 2. Prepara a imagem de fundo pura (sem você nela)
    # Criamos uma cópia para não riscar o original global com texto
    img_final = imagem_dominio_escolhida.copy()

    # 3. Adiciona Textos de Interface na imagem de fundo
    cv2.putText(img_final, f"DOMINIO: {texto_dominio}", (40, 80), 
                cv2.FONT_HERSHEY_DUPLEX, 1.2, cor_texto, 3, cv2.LINE_AA)
    cv2.putText(img_final, "[ PRESSIONE ESPACO PARA VOLTAR ]", (40, 680), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 4. Loop de Paralisia Temporal: Mostra apenas a imagem pura
    while True:
        # Lê a câmera "no vácuo" para não travar o hardware, mas descarta a imagem
        sucesso, _ = cap_camera.read() 
        
        # Exibe a imagem estática do domínio
        cv2.imshow("Projeto Nexus - Batalha Cinematografica", img_final)
        
        # Aguarda comando para sair
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '): # Barra de espaço
            print("🛑 Domínio desfeito. Voltando à realidade.")
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
# 5. LOOP PRINCIPAL (LEITURA DA CÂMERA E IA)
# =============================================================
cap = cv2.VideoCapture(0)
cap.set(3, 1280) # Largura
cap.set(4, 720)  # Altura

# Detector de mãos (sem desenhar esqueleto na tela final)
detector = HandDetector(detectionCon=0.7, maxHands=2)

gesto_anterior = "NENHUM"
gesto_candidato = "NENHUM"
frames_confirmacao = 0
LIMITE_FRAMES = 12 # Quantos frames seguidos a pose deve ser mantida
TAXA_CONFIANCA_MINIMA = 0.80 # 80% de certeza da IA

print("\n🎥 SISTEMA PRONTO. Faça um sinal e SEGURE firme para ativar!")

while True:
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1) # Espelha a câmera
    
    # Busca mãos sem desenhar linhas verdes ( draw=False )
    hands, img = detector.findHands(img, draw=False) 

    features_atuais = []
    gesto_ia = "NENHUM" 
    porcentagem_confianca = 0.0

    if hands:
        features_atuais = extrair_features(hands)
        
        # Pega as probabilidades da IA
        probabilidades = modelo_knn.predict_proba([features_atuais])[0]
        indice_maior = np.argmax(probabilidades)
        porcentagem_confianca = probabilidades[indice_maior]
        gesto_ia_bruto = modelo_knn.classes_[indice_maior]
        
        # Filtro de confiança
        if porcentagem_confianca >= TAXA_CONFIANCA_MINIMA:
            gesto_ia = gesto_ia_bruto
        else:
            gesto_ia = "NENHUM"
        
        # Lógica de confirmação (Barra de Carregamento)
        if gesto_ia == gesto_candidato and gesto_ia != "NENHUM":
            frames_confirmacao += 1
        else:
            gesto_candidato = gesto_ia
            frames_confirmacao = 0

        # --- O GATILHO DA ATIVAÇÃO ---
        if frames_confirmacao >= LIMITE_FRAMES:
            gesto_validado = gesto_candidato

            if gesto_validado != "NENHUM" and gesto_validado != gesto_anterior:
                # Chama a função estática (note que não passamos a 'img' da câmera)
                if gesto_validado == "GOJO":
                    ativar_dominio_estatico(cap, gesto_validado, som_gojo, img_dominio_gojo, "VAZIO INCOMENSURAVEL", (255, 255, 255))
                elif gesto_validado == "SUKUNA":
                    ativar_dominio_estatico(cap, gesto_validado, som_sukuna, img_dominio_sukuna, "SANTUARIO MALEVOLENTE", (0, 0, 255))
                elif gesto_validado == "FLECHA_FOGO":
                    ativar_dominio_estatico(cap, gesto_validado, som_flecha, img_dominio_flecha, "FLECHA DE FOGO (KAMINO)", (0, 165, 255))
                elif gesto_validado == "SIX_SEVEN":
                    ativar_dominio_estatico(cap, gesto_validado, som_sixseven, img_dominio_sixseven, "MODO: SIX SEVEN (67)", (255, 255, 0))
                elif gesto_validado == "MAHITO":
                    ativar_dominio_estatico(cap, gesto_validado, som_mahito, img_dominio_mahito, "AUTO-ENCARNACAO DA PERFEICAO", (255, 0, 255))
                
                # Zera variáveis após voltar do domínio
                gesto_anterior = "NENHUM"
                gesto_candidato = "NENHUM"
                frames_confirmacao = 0
            else:
                gesto_anterior = gesto_validado

    else:
        gesto_candidato = "NENHUM"
        frames_confirmacao = 0
        gesto_anterior = "NENHUM"

    # --- HUD NA TELA (Quando estiver em modo câmera normal) ---
    texto_hud = f"LENDO: {gesto_ia} ({porcentagem_confianca*100:.0f}%) | Casting: [{frames_confirmacao}/{LIMITE_FRAMES}]"
    cor_hud = (0, 255, 0) if (porcentagem_confianca >= TAXA_CONFIANCA_MINIMA and gesto_ia != "NENHUM") else (0, 150, 255) 

    cv2.putText(img, texto_hud, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, cor_hud, 2)
    cv2.putText(img, "Segure a pose firme. Errou? Aperte [0-5] para treinar.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Projeto Nexus - Batalha Cinematografica", img)

    # Contróis do teclado (Q para sair, 0-5 para treino contínuo)
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