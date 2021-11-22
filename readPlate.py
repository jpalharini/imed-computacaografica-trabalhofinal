import cv2 as cv
import numpy as np
import imutils
import sys
import pytesseract as pt

# O HAAR Cascade foi treinado a partir de:
# - 96 imagens positivas de acervo pessoal
# - 30 imagens negativas (fundos comuns como estradas e estacionamentos) coletadas no Google
#
# Após a seleção das imagens, realizei as seguintes etapas (todos arquivos disponíveis no diretório
# 'personal-cars-dataset'):
# 1. Realizei as anotações (opencv_annotations) sobre as imagens positivas (pos.txt);
# 2. Gerei um vetor de exemplos (opencv_createsamples), usando uma imagem-base e depois 
#    utilizando o conjunto completo (sample.vec);
# 3. Executei o treinamento (opencv_traincascade) usando os seguintes parâmetros:
#    - Estágios (nstages): 12
#    - Mínimo de acertos (minhitrate): 0.99
#    - Máximo de alertas falsos (maxfalsealarm): 0.5
#    - Número de positivos (npos): 80
#    - Número de negativos (nneg): 30
#    - Tamanho da imagem (w / h): 20 / 20
cascadeCar = cv.CascadeClassifier("personal-cars/cascade/cascade.xml")

def debug(img, title="Debug"):
    while True:
        cv.imshow(title, img)
        k = cv.waitKey(60)
        if k == 27:
            break

file = sys.argv[1]
car = cv.imread("test/" + file)
carResize = cv.resize(car, (1000,1000))
carGray = cv.cvtColor(carResize, cv.COLOR_BGR2GRAY)

def findPlateCandidates(x, y, w, h):
    relevantColorImage = carResize[y:y+h, x:x+w]
    debug(relevantColorImage, "Car")

    # Usando transformações morfológicas para melhorar a detecção de retângulos
    thresh = cv.threshold(carGray, 170, 255, cv.THRESH_BINARY)[1]
    relevantImage = thresh[y:y+h, x:x+w]
    
    # Detectando contornos
    contours = cv.findContours(relevantImage.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Selecionando os 10 contornos com maior área
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]

    # Inicializando uma lista para guardar as imagens "candidatas"
    plateCandidates = []
    for c in contours:
        # Identificando os contornos com um contêiner retângulo
        (cx, cy, cw, ch) = cv.boundingRect(c)
        ar = cw / ch
        # Verificando o aspect ratio do retângulo. Usual para placa, conforme testes, é entre 2 e 2.5.
        if 2 < ar < 2.5:
            print("Found plate candidate - Aspect Ratio: " + str(ar))
            plateCandidates.append(relevantImage[cy:cy+ch, cx:cx+cw])

    if not plateCandidates:
        print("No license plate detected")
        return None
    else:
        print("Found " + str(len(plateCandidates)) + " plate candidates")
        return plateCandidates

def readPlateCandidate(plateImage):
    pass
    #print(pt.image_to_string(plateImage, config='--psm 5'))

# Como trabalharemos com fotos onde carros são o sujeito principal, defini um tamanho mínimo 
# de 40% da imagem, o que melhorou a detecção.
# Também usei a imagem com histograma equalizado para melhorar o contraste
carContrast = cv.equalizeHist(carGray)
carDetection = cascadeCar.detectMultiScale(carContrast, 1.1, 5, minSize=(400, 400))

# Faremos a tentativa de leitura em mais de um retângulo, para garantir que acharemos a placa
for (x, y, w, h) in carDetection:
    print("Found car")
    plateCandidates = findPlateCandidates(x, y, w, h)
    if plateCandidates:
        for candidate in plateCandidates:
            debug(candidate)
            readPlateCandidate(candidate)


cv.destroyAllWindows()