from ast import And
from distutils.command.config import config
from operator import and_
from socket import MSG_PEEK
import cv2 # --- C:\Program Files\Tesseract-OCR ---
import numpy 
import pytesseract
from PIL import Image

CAP = cv2.VideoCapture ("video1.mp4")
CTEXTO = ''

while True:
    print("Inico video")
    ret, marco = CAP.read()
    print("Continua, ret: ", str(ret))
    if ret == False:
        break 

    cv2.rectangle(marco, (870, 750),(1070,850), (0,0,0),cv2.FILLED)
    cv2.putText(marco,CTEXTO[0:7],(900, 810), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255 ,0), 2)

    al, an, c = marco.shape
    x1 = int (an/3)
    x2 = int (x1*2)
    y1 = int (al/3)
    y2 = int (y1*2)

    cv2.rectangle(marco, (x1 + 160, y1 + 500), (1120,940), (0,0,0), cv2.FILLED)
    cv2.putText(marco, 'Resultado Analisis', (x1 + 180, y1 + 550), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)

    cv2.rectangle(marco, (x1,y1), (x2,y2), (0,255,0),2)
    recorte = marco[y1:y2, x1:x2]

    MB = numpy.matrix(recorte[:,:,0])
    MG = numpy.matrix(recorte[:,:,1])
    MR = numpy.matrix(recorte[:,:,2])
    COLOR = cv2.absdiff (MG, MB)


    _, UMBRAL = cv2.threshold(COLOR, 40, 255, cv2.THRESH_BINARY)
    Contornos, _ = cv2.findContours(UMBRAL, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Contornos = sorted(Contornos, key=lambda x: cv2.contourArea(x), reverse=True)

    for Contorno in Contornos:
        Area = cv2.contourArea(Contorno)
        print("Area: ", Area)
        if Area > 50 and Area < 500:
            x, y, ancho, alto = cv2.boundingRect(Contorno)
            xpi = x + x1
            ypi = y + y1
            xpf = x + ancho + x1
            ypf = y + alto + y1
            cv2.rectangle(marco, (xpi,ypi), (xpf,ypf), (255,255,0),2)

            placa = marco[ypi:ypf, xpi:xpf]
            alp, anp, cp = placa.shape
            
            MVA = numpy.zeros((alp, anp))

            MBP = numpy.matrix(placa[:, :, 0])
            MGP = numpy.matrix(placa[:, :, 1])
            MRP = numpy.matrix(placa[:, :, 2])

            for col in range (0, alp):
                for fil  in range (0, anp):
                    MAX = max(MRP[col,fil], MBP[col, fil])
                    MVA [col, fil] = 255 - MAX

            _, binariza = cv2.threshold(MVA, 150, 255, cv2.THRESH_BINARY)
            binariza = binariza.reshape (alp, anp)
            binariza = Image.fromarray(binariza)
            binariza = binariza.convert('L')

            print("binariza: ", binariza)

            if alp >= 1 and anp >= 10:
                print("Entro al if")
                
                pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
                texto = pytesseract.image_to_string(binariza, config="--psm 11")

                print("texto: ", texto)

                if len (texto) >=7:
                    CTEXTO = texto
                    print(texto)
            break
        cv2.imshow('Deteccion Placa en Vehiculo', marco)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
         #   break
        t = cv2.waitKey(1)
        
        if t == 27:
            break
CAP.release()
cv2.destroyAllWindows()

























