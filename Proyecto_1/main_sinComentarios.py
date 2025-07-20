from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import TextBox
import matplotlib.pyplot as plt
import numpy as np
import random
import time

def hteta(theta: np.array, x: np.array):
    return x @ theta

def computeCost(X: np.array, y: np.array, theta: np.array):
    m = len(y)
    error = hteta(theta, X) - y
    return (1 / (2 * m)) * np.sum(error ** 2)

def gradientDescent(X: np.array, y: np.array, theta: np.array, alpha: float, num_iters: int):
    m = len(y)
    J_history = np.zeros(num_iters)
    theta_history = []
    for i in range(num_iters):
        error = hteta(theta, X) - y
        theta = theta - (alpha / m) * (X.T @ error)
        J_history[i] = computeCost(X, y, theta)
        theta_history.append(theta.copy())
    return theta, J_history, theta_history

lista = []
with open('C:/Users/DEATH/Desktop/IA/data1.txt') as archivo:
    for i in archivo:
        lista.append(i.strip().split(","))
        
matriz = np.array(lista, dtype=float)
X = np.array([row[0] for row in matriz])
Y = np.array([row[1] for row in matriz])
X_b = np.c_[np.ones((len(X), 1)), X]
alpha = 0.02
N = 100
theta_inicial = np.array([0.0, 0.0])
fig = plt.figure(figsize=(12, 10))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
ax1.scatter(X, Y, c="red", label='Datos originales', marker="x")
theta_malla = np.linspace(-2, 2, 100)
T0, T1 = np.meshgrid(theta_malla, theta_malla)
Z = np.zeros(T0.shape)
for i in range(T0.shape[0]):
    for j in range(T0.shape[1]):
        t = np.array([T0[i, j], T1[i, j]])
        Z[i, j] = computeCost(X_b, Y, t)

contornos = ax3.contour(T0, T1, Z, levels=50, cmap='viridis')
ax3.clabel(contornos, inline=True, fontsize=8)
ax4.plot_surface(T1, T0, Z, cmap='viridis', alpha=0.7, edgecolor='none')
theta_final, J_history, theta_history = gradientDescent(X_b, Y, theta_inicial, alpha, N)
x_line = np.linspace(X.min(), X.max(), 100)
linea_plot, = ax1.plot([], [], color="blue", label=f"Ajuste con Alpha={alpha}")
linea_costo, = ax2.plot([], [], color='green', label='Costo J')
ax2.set_xlim(0, N)
ax2.set_ylim(0, max(J_history) * 1.1)
punto, = ax3.plot([], [], 'rx', markersize=10)
ax3.set_xlabel("θ0")
ax3.set_ylabel("θ1")
theta_0_malla = []
theta_1_malla = []

for i in range(N):
    theta = theta_history[i]
    y_line = theta[0] + theta[1] * x_line
    linea_plot.set_data(x_line, y_line)
    linea_costo.set_data(range(i + 1), J_history[:i + 1])
    theta_0_malla.append(theta[0])
    theta_1_malla.append(theta[1])
    ax3.plot(theta_0_malla, theta_1_malla, 'r-', alpha=0.6)
    punto.set_data([theta[0]], [theta[1]])
    ax4.plot(theta_1_malla, theta_0_malla, J_history[:i+1], color='Orange', marker='o')
    plt.pause(0.05)

ax1.set_title('Ajuste por descenso de gradiente')
ax1.set_xlabel('Pobllacion en 1 mil')
ax1.set_ylabel('Valor en 10 mil')
ax1.legend()
ax2.set_title('Evolución del costo J')
ax2.set_xlabel('Iteración')
ax2.set_ylabel('Costo')
ax2.legend()
ax3.set_title('Curvas de nivel')
ax3.set_xlabel('theta0')
ax3.set_ylabel('theta1')
ax4.set_title('Superficie del Costo J(theta 0, theta 1)')
ax4.set_xlabel('theta 0')
ax4.set_ylabel('theta 1')
ax4.set_zlabel('J(theta_0, theta_1)')
plt.tight_layout()
plt.figure(fig.number)  
plt.show()

# Crear la segunda figura
fig2, (ax5, ax6) = plt.subplots(1, 2)

#Norma de la regrecion lineal
B=[]
B= (X**-1)*Y

# Puntos originales y recta final
ax5.scatter(X, Y, c="red", label='Datos originales', marker="x")
linea_actual, = ax5.plot(x_line, theta_final[0] + theta_final[1] * x_line, color="blue", label=f'Recta final con Alpha={alpha}')

ax6.scatter(X, Y, c="red", label='Datos originales', marker="x")
linea_actual_2, = ax6.plot(x_line, B[0] + B[1] * x_line, color="blue", label=f'Recta final con Alpha={alpha}')

# Líneas auxiliares y punto nuevo (inicialmente vacíos)
linea_x, = ax5.plot([], [], 'r--')
linea_y, = ax5.plot([], [], 'r--')
punto_prediccion = ax5.scatter([], [], color='black', zorder=5, label='Nuevo punto')
texto_y = ax5.text(0, 0, '', color='black', fontsize=10)

linea_x_2, = ax6.plot([], [], 'r--')
linea_y_2, = ax6.plot([], [], 'r--')
punto_prediccion_2 = ax6.scatter([], [], color='black', zorder=5, label='Nuevo punto')
texto_y_2 = ax6.text(0, 0, '', color='black', fontsize=10)

# Configurar la caja de texto interactiva
axbox = plt.axes([0.25, 0.05, 0.5, 0.075])  # Posición [x, y, ancho, alto]
text_box = TextBox(axbox, 'Ingresar x:', initial="0")

# Función que se ejecuta al enviar un valor
def update(val):
    try:
        nuevo_x = float(text_box.text)
        nuevo_y = theta_final[0] + theta_final[1] * nuevo_x

        nuevo_x_2 = float(text_box.text)
        nuevo_y_2 = B[0] + B[1] * nuevo_x_2

        # Recalcular rango de x para la línea de regresión si es necesario
        x_min = min(X.min(), nuevo_x) - 1
        x_max = max(X.max(), nuevo_x) + 1
        x_linea_nueva = np.linspace(x_min, x_max, 100)

        x_min_2 = min(X.min(), nuevo_x_2) - 1
        x_max_2 = max(X.max(), nuevo_x_2) + 1
        x_linea_nueva_2 = np.linspace(x_min_2, x_max_2, 100)

        # Actualizar la recta de regresión
        linea_actual.set_data(x_linea_nueva, theta_final[0] + theta_final[1] * x_linea_nueva)

        linea_actual_2.set_data(x_linea_nueva_2, B[0] + B[1] * x_linea_nueva_2)

        # Actualizar líneas auxiliares
        linea_y.set_data([x_min, nuevo_x], [nuevo_y, nuevo_y])
        linea_x.set_data([nuevo_x, nuevo_x], [Y.min() - 1, nuevo_y])

        linea_y_2.set_data([x_min_2, nuevo_x_2], [nuevo_y_2, nuevo_y_2])
        linea_x_2.set_data([nuevo_x_2, nuevo_x_2], [Y.min() - 1, nuevo_y_2])

        # Actualizar punto y texto
        punto_prediccion.set_offsets([[nuevo_x, nuevo_y]])
        texto_y.set_position((nuevo_x + 0.5, nuevo_y))
        texto_y.set_text(f'Y = {nuevo_y:.2f}')

        punto_prediccion_2.set_offsets([[nuevo_x_2, nuevo_y_2]])
        texto_y.set_position((nuevo_x_2 + 0.5, nuevo_y_2))
        texto_y.set_text(f'Y = {nuevo_y_2:.2f}')

        # Ajustar límites del gráfico dinámicamente
        ax5.set_xlim(x_min, x_max)
        ax5.set_ylim(min(Y.min(), nuevo_y) - 1, max(Y.max(), nuevo_y) + 1)

        ax6.set_xlim(x_min_2, x_max_2)
        ax6.set_ylim(min(Y.min(), nuevo_y_2) - 1, max(Y.max(), nuevo_y_2) + 1)
        
        # imprir graficos
        fig2.canvas.draw_idle()

    except ValueError:
        print("¡Ingresa un número válido!")

# Conectar la función al evento 'submit' del TextBox
text_box.on_submit(update)

# Configuraciones adicionales del gráfico
ax5.set_title('Predicción con Regresión Lineal')
ax5.set_xlabel('Población (en 10,000s)')
ax5.set_ylabel('Ganancia (en $10,000s)')
ax5.legend()

ax6.set_title('Predicción con Regresión Lineal')
ax6.set_xlabel('Población (en 10,000s)')
ax6.set_ylabel('Ganancia (en $10,000s)')
ax6.legend()

plt.show()