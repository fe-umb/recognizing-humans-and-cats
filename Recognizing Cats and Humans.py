# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 11:04:59 2021

@authors:

Fernanda Pereira Umberto

"""


# Importações necessárias.
import cv2
import os

# Alterando o diretório para a pasta do trabalho.
os.chdir("C:/recognizing-humans-and-cats")

# Criando a Cascata de Classificação para gatos e humanos.
cat_cascade = cv2.CascadeClassifier('visionary.net_cat_cascade_web_LBP.xml') # Técnica: LBP
human_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Técnica: HAAR

n_images = 20 # Variável que guarda a quantidade de imagens processadas

# Método que obtém as imagens, processa e as salva com o nome "out + número da imagem".
def processarImagem(image_dir,image_filename):

    img = cv2.imread(image_dir+'/'+image_filename) # Leitura da imagem.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Definição dos tons de cinza.
    
    cats = cat_cascade.detectMultiScale(gray, 1.3, 5) # Função para a detecção de gatos.
    human = human_cascade.detectMultiScale(gray, 1.3, 5) # Função para a detecção de humanos.
    
    i = 0
    for (x,y,w,h) in cats: # For que realiza o mapeamento dos retângulos e legenda nos gatos.
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) # Set dos retângulos.
        cv2.putText(img, "Gato {}".format(i + 1), (x, y - 10), # Set da legenda.
		cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        i += 1
        
    j = 0
    for (x,y,w,h) in human: # For que realiza o mapeamento dos retângulos e legenda nos humanos.
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) # Set dos retângulos.
        cv2.putText(img, "Humano {}".format(j + 1), (x, y - 10), # Set da legenda.
		cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        j += 1
    
    cv2.imwrite('out'+image_filename,img) # Salvando a imagem no diretório com as classificações.
    

for idx in range(1,n_images+1): 
    processarImagem('cats/',str(idx)+'.jpg') # Chamada do método de processamento de imagens para cada uma que está na pasta denominada "gatinhos".


for idx in range(1,n_images+1): # For para realizar a criação de janelas com as classificações finalizadas. Para ir para a próxima imagem basta apertar qualquer botão do teclado.
    img = cv2.imread('out'+str(idx)+'.jpg')
    cv2.imshow('Imagem {}'.format(idx), img) # Mostrando as imagens por ordem de criação.
    cv2.waitKey() # Aguarda a tecla para continuar.

cv2.destroyAllWindows() # Desalocando memória.