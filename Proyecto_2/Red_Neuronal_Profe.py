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
plt.plot(result.get('J'))
plt.xlabel('Número de iteraciones (epocas)')
plt.ylabel('Función de Costo, J')
plt.show()

#print("Matrices de pesos inicial (aleatorias) y final (despues de iteraciones):")
#print(w_s)

for i in [0, epochs-1]:
    if i==0:
        print('PESOS INICIALES ALEATORIOS')
        print('==========================')
    if i==(epochs-1):
        print('PESOS FINALES DE RED NEURONAL ENTRENADA')
        print('=======================================')        
    j = len(w_s)
    for j in range(1,j+1):
        display(Math(r'\Theta^{(%d)} = ' % (j)))
        print(w_s['w_' + str(i)][j-1]) #w_s['w_0'][0],w_s['w_0'][1], w_s['w_4999'][0], w_s['w_4999'][1]
    print('')

print('PREDICCIÓN INICIAL')
print('==================')
display(Math(r'h_\Theta(x) = ' ))
print(result.get('h')[0])

print('')

print('PREDICCIÓN FINAL ESTIMADA')
print('=========================')
display(Math(r'h_\Theta(x) = ' ))
print(result.get('h')[-1])