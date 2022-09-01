from keras.models import load_model
import numpy as np
import cv2

# Carregamento da rede neural.
model = load_model("keras_model.h5")

# Define o formato de entrada dos dados da rede neural.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Carrega e define qual Web Cam será utilizada.
webCam = cv2.VideoCapture(0)

# Nomeia as "possibilidades" da rede neural.
classes = ['Felipe', 'Guilherme', 'Nada']

while True:
    # Captura as imagens.
    x, img = webCam.read()

    # Redimensiona a imagem capturada para a proporção e resolução definida em "data".
    imgMini = cv2.resize(img, (224, 224))

    # Transforma a imagem em um array Numpy.
    image_array = np.asarray(imgMini)

    # Ajusta o padrão da imagem para o mesmo da rede neural.
    normalizacao = (image_array.astype(np.float32)/127.0)-1

    # Armazena a imagem em um array python.
    data[0] = normalizacao

    # Função que calcula a probabilidade da imagem ser uma das "possibilidades" da rede neural.
    prediction = model.predict(data)

    # Define um resultado e armazena em forma de índice de array.
    indexValue = np.argmax(prediction)

    # Função para escrita na tela.
    cv2.putText(img, str(classes[indexValue]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

    # Printa no console os resultados.
    print(classes[indexValue])

    # Carrega a janela mostrando os resultados.
    cv2.imshow('Web Cam', img)

    # Função que define a continuidade da captura.
    cv2.waitKey(1)

    # Função para quebra do loop e finalização do programa.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break