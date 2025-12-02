import matplotlib.pyplot as plt
import numpy as np
import random
import time

# Leer datos desde archivo
lista = []
with open('IA\Misael_2025_1S\Tarea_2\data1.txt') as archivo:
    for i in archivo:
        lista.append(i.strip(" ").strip("\n").split(","))

# Separar datos en X e Y
rx = [i[0] for i in lista]
ry = [i[1] for i in lista]

X = list(map(float, rx))
Y = list(map(float, ry))

# Colores aleatorios
colores = [(random.random(), random.random(), random.random()) for _ in range(len(X))]

# Coordenadas de las líneas
#se debe cambiar las cordenadas ramdon a cordendas a traves de la gradiente desendente
lineas = [
    ([28, 15], [-5, 28]),
    ([random.randint(5, 25), random.randint(5, 25)], [random.randint(5, 25), random.randint(5, 25)]),
    ([random.randint(5, 25), random.randint(5, 25)], [random.randint(5, 25), random.randint(5, 25)]),
    ([random.randint(5, 25), random.randint(5, 25)], [random.randint(5, 25), random.randint(5, 25)]),
    ([5, 22], [1, 24]),
]

# Crear figura y ejes
fig, ax = plt.subplots()
ax.scatter(X, Y, c=colores)
ax.set_title('Gráfico de dato1.txt')
ax.set_xlabel('Eje x')
ax.set_ylabel('Eje y')

# Variable para guardar la línea actual y poder borrarla
linea_actual = None
linea_final = None
marca_x=0
marca_y=0
nose_x=[]
nose_y=[]
# Mostrar una línea a la vez
for auxX, auxY in lineas:
    # Si ya hay una línea dibujada, la eliminamos
    if linea_actual:
        linea_actual.remove()

    # Dibujar nueva línea y guardar la referencia
    linea_actual, = ax.plot(auxX, auxY, color='black')
    marca_x=auxX[1]
    marca_y=auxY[1]
    nose_x=auxX
    nose_y=auxY
    # Pausa para mostrar
    plt.pause(1)

# Mostrar el gráfico final (última línea)
plt.show()

# X ingresado por profesor
nuevo_x=int(input("ingresa el valor de X (entre 5 y 25): "))

# calculo de la pendiente
m= (nose_y[1]-nose_y[0])
m= m /(nose_x[1]-nose_x[0])

# calculko de punto posicion
A=nose_y[1]-(m*nose_x[1])

#calculo del nuevo Y segun el X ingresado
nuevo_y=A+(m*nuevo_x)

# linea del Eje Y (linea recta horizontal)
linea_Y_x = [5 , nuevo_x]
linea_Y_y = [nuevo_y , nuevo_y]

# linea del eje X (linea recdta vertical)
linea_X_x = [nuevo_x , nuevo_x]
linea_X_y = [0 , nuevo_y]

fig, ax = plt.subplots()
ax.scatter(X, Y, c=colores)
ax.set_title('Gráfico de dato1.txt')
ax.set_xlabel('Eje x')
ax.set_ylabel('Eje y')
# imprimr la racta horizontal
linea_y = ax.plot(linea_Y_x , linea_Y_y , color="RED" , linestyle='--')
# impreme la recta vertical
linea_x = ax.plot(linea_X_x , linea_X_y , color="RED" , linestyle='--')
# imprime la recta pendiente original
linea_final = ax.plot(nose_x , nose_y , color="BLACK")
# funcion imprimir
plt.show()