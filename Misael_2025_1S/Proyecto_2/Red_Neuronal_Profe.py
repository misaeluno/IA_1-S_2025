import numpy as np
import matplotlib.pyplot as plt
# Resultados
#============
from IPython.display import Math

# Función de activación
#=======================
# Sigmoidal
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivada de la sigmoidal
def s_prime(z):
    return np.multiply(z, 1.0-z)

# Inicialización de los pesos
#=============================
def init_weights(layers, epsilon):
    weights = []
    for i in range(len(layers)-1):
        w = np.random.rand(layers[i+1], layers[i]+1)
        w = w * 2*epsilon - epsilon
        weights.append(np.mat(w))
    return weights

# Red Neuronal
#==============
def fit(X, Y, w):
    # Inicialización de cada parámetro con gradiente igual a 0
    w_grad = ([np.mat(np.zeros(np.shape(w[i])))
              for i in range(len(w))])  # len(w) es igual al número de capas
    m, n = X.shape
    h_total = np.zeros((m, 1))  # Valor predecido de todas las muestras, m*1, probabilidad
    for i in range(m):
        x = X[i]
        y = Y[0,i]
        # Propagación hacia adelante
        #============================
        a = x
        a_s = []
        for j in range(len(w)):
            a = np.mat(np.append(1, a)).T
            a_s.append(a)  # Aquí se guarda el valor a de la capa L-1 anterior.
            z = w[j] * a
            a = sigmoid(z)
        h_total[i, 0] = a
        # Propagación hacia atras (backpropagation)
        #===========================================
        delta = a - y.T
        w_grad[-1] += delta * a_s[-1].T  # Cradiente de la capa L-1
        # Reverso, desde la penúltima capa hasta el final de la segunda capa, excluyendo la primera y la última capa
        for j in reversed(range(1, len(w))):
            delta = np.multiply(w[j].T*delta, s_prime(a_s[j]))  # El parámetro pasado aquí es a, No z
            w_grad[j-1] += (delta[1:] * a_s[j-1].T)
    w_grad = [w_grad[i]/m for i in range(len(w))]
    J = (1.0 / m) * np.sum(-Y * np.log(h_total) - (np.array([[1]]) - Y) * np.log(1 - h_total))
    return {'w_grad': w_grad, 'J': J, 'h': h_total}

# Definición de los parámetros de la red
#========================================
# Conjunto de entrenamiento
X = np.mat([[0,0],
            [0,1],
            [1,0],
            [1,1]])
Y = np.mat([0,1,1,0])

# Configuración de red
layers = [2,2,1]
epochs = 5000 #número de iteraciones
alpha = 0.5 #tasa de aprendizaje
epsilon = 1 #para inicializar los pesos

# Entrenamiento de la red
#========================
w = init_weights(layers, epsilon)
result = {'J': [], 'h': []}
w_s = {}
for i in range(epochs):
    fit_result = fit(X, Y, w)
    w_grad = fit_result.get('w_grad')
    J = fit_result.get('J')
    h_current = fit_result.get('h')
    result['J'].append(J)
    result['h'].append(h_current)
    for j in range(len(w)):
        w[j] -= alpha * w_grad[j]
    if i == 0 or i == (epochs - 1):
        # print('w_grad', w_grad)
        w_s['w_' + str(i)] = w_grad[:]

# Gráfico de la Función de Costo, J
#===================================
# Resultados
# ============

# Si usas NumPy, asegúrate de haberlo importado al principio de tu script
# import numpy as np

# Asegúrate de que las variables 'w_s', 'epochs', y 'result'
# estén definidas y contengan los datos correctos de tu entrenamiento.
# Por ejemplo:
# w_s = {'w_0': [np.array([[...]]), np.array([[...]])], 'w_4999': [np.array([[...]]), np.array([[...]])]}
# epochs = 5000
# result = {'J': [...], 'h': [np.array([[...]]), ..., np.array([[...]])]}


print("---")
print("Matrices de pesos inicial (aleatorias) y final (después de iteraciones):")
print("---")

for i in [0, epochs - 1]:
    if i == 0:
        print('\nPESOS INICIALES ALEATORIOS')
        print('==========================')
    elif i == (epochs - 1): # Usamos elif porque solo puede ser uno de los dos
        print('\nPESOS FINALES DE RED NEURONAL ENTRENADA')
        print('=======================================')

    # Itera sobre los elementos guardados en w_s para la época actual
    # w_s['w_' + str(i)] es una lista de matrices de gradientes
    for j_idx, weight_matrix in enumerate(w_s['w_' + str(i)]):
        # Usamos j_idx + 1 para que el contador de capa sea 1-basado
        print(f'Theta^({j_idx + 1}) = ')
        print(weight_matrix) # Imprime la matriz de pesos
    print('') # Salto de línea para separar las secciones

print('---')
print('PREDICCIÓN INICIAL')
print('==================')
print('h_Theta(x) = ')
print(result.get('h')[0])

print('')

print('---')
print('PREDICCIÓN FINAL ESTIMADA')
print('=========================')
print('h_Theta(x) = ')
print(result.get('h')[-1])