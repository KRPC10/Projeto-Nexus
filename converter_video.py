import os
# Mudança aqui: Tiramos o '.editor' pois o MoviePy 2.0+ não usa mais isso!
from moviepy import VideoFileClip

print("==================================================")
print(" 🎬 CONVERSOR DE GIF/IMAGEM ANIMADA PARA MP4 🎬 ")
print("==================================================")

arquivo_entrada = input("Digite o nome do arquivo que você quer converter (ex: gojo.jpg, sukuna.gif): ")

if not os.path.exists(arquivo_entrada):
    print(f"\n❌ ERRO: O arquivo '{arquivo_entrada}' não foi encontrado nesta pasta!")
    print("Verifique se o nome está correto (incluindo o .jpg ou .gif no final).")
else:
    nome_base, _ = os.path.splitext(arquivo_entrada)
    arquivo_saida = f"video_{nome_base}.mp4"
    
    print(f"\n⏳ Preparando para converter: '{arquivo_entrada}' -> '{arquivo_saida}'...")
    
    try:
        clip = VideoFileClip(arquivo_entrada)
        clip.write_videofile(arquivo_saida, codec="libx264")
        print(f"\n✅ SUCESSO! O arquivo '{arquivo_saida}' foi criado e está pronto para o Domínio!")
    except Exception as erro:
        print(f"\n❌ Ocorreu um erro durante a conversão: {erro}")