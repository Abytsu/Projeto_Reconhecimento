# Importando as blibiotecas que serão utilizadas no código
import cv2
import numpy as np 
import pandas as pd

# Importando o TensorFlow
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Criando uma variável do local da imagem que será analisada
imagem = cv2.imread('C:/Users/Abytsu/Pictures/Icones/gustavo.jpg') 
# É necessário colocar o caminho da imagem, como por exemplo: 'C:/Users/Abytsu/Pictures/Icones/gustavo.jpg'

# Criando variáveis que indicam a localização dos recursos da inteligência artificial
cascade_faces = "C:/Users/Abytsu/Downloads/Material/Material/haarcascade_frontalface_default.xml" # Material para detectação de faces
caminho_modelo = "C:/Users/Abytsu/Downloads/Material/Material/modelo_01_expressoes.h5" # Rede neural já estabelecida
face_detection = cv2.CascadeClassifier(cascade_faces) # objeto para detecção de faces
classificador_emocoes = load_model(caminho_modelo, compile = False) # modelo do tensorflow 
expressoes = ['Raiva', 'Nojo', 'Medo', 'Feliz', 'Triste', 'Surpreso', 'Neutro']

original = imagem.copy() # copia a imagem original
faces = face_detection.detectMultiScale(original, scaleFactor = 1.2,minNeighbors = 5, minSize = (20,20))

print(faces) #valores da face
print(len(faces)) #quantas faces encontradas
print(faces.shape) #atributos da face encontrada

cinza = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) # transforma a imagem de colorido para cinza (um canal) para processamento mais rápido
# cv2_imshow(cinza) # mostra a imagem em escala cinza
cinza.shape # mostra o tamanho da imagem (agora sem os canais de cores)

original = imagem.copy()

# Realização do reconhecimento facial e da aplicação do texto na imagem
for (x, y, w, h) in faces:
    roi = cinza[y:y + h, x:x + w] # Coordenadas para a extração da face

    # Redimensionamento da imagem
    roi = cv2.resize(roi, (48, 48))
    cv2.imshow("image", roi)

    # Convertendo de Inteiro para Float
    roi = roi.astype('float')/255
    roi = img_to_array(roi)
    roi = np.expand_dims(roi,axis = 0)

    # Realizando a previsão
    preds = classificador_emocoes.predict(roi)[0]
    print(preds)

    # Detectando emoção com maior probabilidade
    emotion_probability = np.max(preds)
    print(emotion_probability)

    # Transformando os números da emoção em texto
    print(preds.argmax())
    label = expressoes[preds.argmax()]

    # Escrevendos os dados na imagem
    cv2.putText(original, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, 1)
    cv2.rectangle(original, (x, y), (x + w, y + h), (0,0,255), 2)

cv2.imshow("image", original) # Mostra o resultado final
cv2.waitKey(0) 
cv2.destroyAllWindows()
# Duas últimas linhas necessárias para a imagem permanecer ativa