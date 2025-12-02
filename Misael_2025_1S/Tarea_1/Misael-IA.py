import matplotlib.pyplot as plt
import numpy as np
import random
import math
import time

#crea Lista tipo list para guardar los datos del documento
lista=[]
with open('IA\Misael_2025_1S\Tarea_1\data1.txt') as archivo:
    for i in archivo:
        # agrega a la Lista los datos del txt y los separa segun , ademas elimna el \n y espacios
        lista.append(i.strip(" ").strip("\n").split(","))

rx=[]
ry=[]
for i in lista:
    #el i[0] contiene todos los datos antes de la "," del txt, es decir, los datos del eje X
    rx.append(i[0])
    #el i[i] contiene todos los datos despues de la "," del txt, es decir, los datos del eje Y
    ry.append(i[1])

X=list(map(float,rx))
Y=list(map(float,ry))

# Generar una lista de colores aleatorios en formato RGB
colores = [(random.random(), random.random(), random.random()) for _ in range(len(X))]

# crea el grafico de dispercion de los datos X ,Y, ademas agrega un color
plt.scatter(X, Y, c=colores)

#cordenada 1
auxX1=[28,15]
auxY1=[-5,28]
#cordenada 2
auxX2=[random.randint(0,25),random.randint(0,25)]
auxY2=[random.randint(0,25),random.randint(0,25)]
#cordenda 3
auxX3=[random.randint(0,25),random.randint(0,25)]
auxY3=[random.randint(0,25),random.randint(0,25)]
#cordenada 4
auxX4=[random.randint(0,25),random.randint(0,25)]
auxY4=[random.randint(0,25),random.randint(0,25)]
#cordenada 5
auxX5=[5,22]
auxY5=[1,24]
cont=0
    
plt.scatter(X, Y, c=colores)
plt.plot(auxX1,auxY1)
plt.plot(auxX2,auxY2)
plt.plot(auxX3,auxY3)
plt.plot(auxX4,auxY4)
plt.plot(auxX5,auxY5)
plt.title('Gráfico de dato1.txt')
plt.xlabel('Eje x')
plt.ylabel('Eje y')
plt.savefig('grafica_dispersion.png')

plt.show()