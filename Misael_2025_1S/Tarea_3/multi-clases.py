from mpl_toolkits.mplot3d import Axes3D  # Necesario para proyecciones 3D
import matplotlib.pyplot as plt
import numpy as np
import random
import time

def hteta(beta0: float, beta1: float, x: np.array):
     teta0=beta0
     teta1=beta1
     X=x
     resultado = (teta1*X)+teta0
     return resultado

def sumatori(beta0: float, beta1: float, y: np.array, x: np.array):
    teta0 = beta0
    teta1 = beta1
    Y = y
    X = x
    resultado = (hteta(teta0,teta1,X) - Y)
    return resultado

# Leer datos desde archivo
lista = []
with open('Tarea_3\data1.txt') as archivo:
    for i in archivo:
        #lista se transforma en una matriz tipo string
        lista.append(i.strip(" ").strip("\n").split(","))

#transforma la lista en una matriz tipo float
matriz = np.array(lista, dtype=float)
#print(matriz[0,0])
#X=matriz[0,0]*2
#print(X)

# Separar datos en X e Y
X = np.array([row[0] for row in matriz])
Y = np.array([row[1] for row in matriz])

#se consigue la pendiente M y la intersecion B
#m, b = np.polyfit(X, Y, 1)
#print(f"pendiente :  {m}")
#print(f"interseccion : {b}")

# Crear figura 1,2,3 y 4 junto con ejes (gradiente, costo, curvas, 3D) 
#fig = plt.figure(figsize=(12, 10))
#ax = fig.add_subplot(2, 2, 1)
#ax2 = fig.add_subplot(2, 2, 2)
#ax3 = fig.add_subplot(2, 2, 3)
#ax4 = fig.add_subplot(2, 2, 4, projection='3d')


#variables a usar
#cantidad de filas en la matriz
N=len(X)
teta0 = 0
teta1 = 0
iteraciones=50
resto=0
alpha = 0.02

# Lista para guardar el valor de costo J en cada iteración
costos = []

# Crear mallas para theta0 y theta1
teta0_malla = np.linspace(-2, 2, 100)
teta1_malla = np.linspace(-2, 2, 100)
T0, T1 = np.meshgrid(teta0_malla, teta1_malla)

#Nube de datos
#ax.scatter(X, Y, c="red", label='Datos originales', marker="x")
# Variable para guardar la línea actual y poder borrarla
linea_actual = None

#if (iteraciones<=20):
#    resto=2
#elif(iteraciones<=50):
#    resto=5
#elif (iteraciones<=100):
#    resto=10
#elif (iteraciones<=200):
#    resto=20
#elif (iteraciones<=500):
#    resto=50
#else:
#    resto=100

# Calcular J(teta_0, teta_1) para cada combinación
Z = np.zeros(T0.shape)
for i in range(T0.shape[0]):
    for j in range(T0.shape[1]):
        errores = hteta(T0[i, j], T1[i, j], X) - Y
        Z[i, j] = (1 / (2 * N)) * np.sum(errores ** 2)

# Usamos Z que ya calculaste como los valores del costo J teta_1 y teta_2 para hacer el 3D
#surf = ax4.plot_surface(T1, T0, Z, cmap='viridis', alpha=0.7, edgecolor='none')   

# Graficar curvas de nivel
#contornos = ax3.contour(T0, T1, Z, levels=50, cmap='viridis')
#ax3.clabel(contornos, inline=True, fontsize=8)

# Opcional: dibujar trayectoria del descenso de gradiente
teta0_hist = []
teta1_hist = []
# Inicializar punto en la curva
#punto, = ax3.plot([teta0], [teta1], 'ro', label='Descenso de gradiente', markersize=5)

# Descenso de gradiente (100 iteraciones) con J(teta0: teta1)
for i in range(iteraciones):
#para gradeinte, costo y curva de nivel
    error = sumatori(teta0, teta1, Y, X)
    costo = (1 / (2 * N)) * np.sum(error ** 2)
    costos.append(costo)

    teta0 -= alpha * (1/N) * np.sum(error)
    teta1 -= alpha * (1/N) * np.sum(error * X)
#------------------------

#para curva de nivel
    teta0_hist.append(teta0)
    teta1_hist.append(teta1)
#------------------------

    #if(i%resto==0):
        #if linea_actual:
            #linea_actual.remove()

    # Solo agregar label en la última iteración para evitar el warning
    #if i == (iteraciones-1):
        #if linea_actual:
            #linea_actual.remove()
            
    # imprime grafico 1
        #linea_actual, = ax.plot(X, teta1 * X + teta0, color="blue", label=f'Recta final con Alpha ={alpha}')
    # imprime grafico 2
        #ax2.plot(costos, color='green', label='Costo J(teta_0;teta_1)')
    # imprime grafico 3
        #ax3.plot(teta0_hist[-2:], teta1_hist[-2:], 'r-', alpha=0.6)  # traza línea entre dos puntos
        # Actualizar punto en el gráfico
        #punto.set_data([teta0], [teta1])
        #ax3.plot(teta0, teta1, 'rx', markersize=10, label='Mínimo encontrado')
        #plt.pause(0.05)
    # Traza la trayectoria del descenso de gradiente en la superficie
        #ax4.plot(teta0_hist, teta1_hist, costos, color='Orange', marker='o', label='Descenso de gradiente')

    #elif(i%resto==0):
    # imprime grafico 1
        #linea_actual, = ax.plot(X, teta1 * X + teta0, color="blue")
    # imprime grafico 2
        #ax2.plot(costos, color='green')
    # Traza la trayectoria del descenso de gradiente en la superficie
        #ax4.plot(teta0_hist, teta1_hist, costos, color='Orange', marker='o')
        #plt.pause(0.05)

#imprime grafico 3
    #ax3.plot(teta0_hist[-2:], teta1_hist[-2:], 'r-', alpha=0.6)  # traza línea entre dos puntos
    #Actualizar punto en el gráfico
    #punto.set_data([teta0], [teta1])
 

# Dibujar el punto final del descenso de gradiente
# Mostrar el gráfico final (última línea)
#ax.set_title('Ajuste por descenso de gradiente')
#ax.set_xlabel('Eje X')
#ax.set_ylabel('Eje Y')
#ax.legend()

# Mostrar gráfico del costo
#ax2.set_title('Evolución del costo J')
#ax2.set_xlabel('Iteración')
#ax2.set_ylabel('Costo')
#ax2.legend()

# Mostrar grafico de curva de nivel
#ax3.set_title('Curvas de nivel')
#ax3.set_xlabel('teta0')
#ax3.set_ylabel('teta1')
#ax3.legend()

# Mostrar grafico en 3D
#ax4.set_title('Superficie del Costo J(teta 0, teta 1)')
#ax4.set_xlabel('teta 1')
#ax4.set_ylabel('teta 0')
#ax4.set_zlabel('J(teta_0, teta_1)')
#ax4.legend()

#--------------------------
#       Emprimir graficos
#--------------------------
#plt.tight_layout()
#plt.show()

#-----------------------
#       Prediccion
#-----------------------
fig, ax5 = plt.subplots()

minoX = X.min(axis=0)
minoY = Y.min(axis=0)
maxiX = X.max(axis=0)
maxiY = Y.max(axis=0)

# X ingresado por profesor
nuevo_x=int(input("ingresa el valor de X (entre 5 y 25): "))

#calculo del nuevo Y segun el X ingresado
nuevo_y=teta0+(teta1*nuevo_x)

if nuevo_x not in X:
    X = np.append(X, nuevo_x)
    Y = np.append(Y, nuevo_y)

# linea del Eje Y (linea recta horizontal)
linea_Y_x = [minoX , nuevo_x]
linea_Y_y = [nuevo_y , nuevo_y]

# linea del eje X (linea recdta vertical)
linea_X_x = [nuevo_x , nuevo_x]
linea_X_y = [minoY , nuevo_y]

ax5.scatter(X, Y, c="red", label='Datos originales', marker="x")

# imprimr la racta horizontal
linea_y = ax5.plot(linea_Y_x , linea_Y_y , color="RED" , linestyle='--')
# impreme la recta vertical
linea_x = ax5.plot(linea_X_x , linea_X_y , color="RED" , linestyle='--')
# imprime la recta pendiente original
linea_actual, = ax5.plot(X, teta1 * X + teta0, color="blue", label=f'Recta final con Alpha ={alpha}')
# Dibujar el punto nuevo
ax5.scatter(nuevo_x, nuevo_y, color='black', zorder=5, label='Nuevo punto')

# Agregar texto con el valor de Y
ax5.text(nuevo_x + 0.5, nuevo_y, f'Y={nuevo_y:.2f}', color='black', fontsize=10)

ax5.set_title('Gráfico de dato1.txt')
ax5.set_xlabel('Eje x')
ax5.set_ylabel('Eje y')
ax5.legend()

#--------------------------
#       Emprimir graficos
#--------------------------
plt.show()