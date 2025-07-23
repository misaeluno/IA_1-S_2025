import numpy as np
import matplotlib.pyplot as plt

# --- Funciones de la Red Neuronal ---

# Función de activación Sigmoidal
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivada de la sigmoidal
def s_prime(z):
    return np.multiply(z, 1.0-z)

# Inicialización de los pesos de la red
def init_weights(layers, epsilon):
    weights = []
    # Itera sobre las conexiones entre capas
    for i in range(len(layers)-1):
        # Crea una matriz de pesos con dimensiones (neuronas_siguientes, neuronas_actuales + bias)
        w = np.random.rand(layers[i+1], layers[i]+1)
        # Reajusta los pesos al rango [-epsilon, epsilon]
        w = w * 2*epsilon - epsilon
        # Añade la matriz de pesos a la lista (como objeto np.mat para operaciones de álgebra lineal)
        weights.append(np.mat(w))
    return weights

# Función para realizar un paso de entrenamiento (forward y backpropagation)
def fit(X, Y, w):
    # Inicializa las matrices de gradientes a cero para cada capa de pesos
    w_grad = ([np.mat(np.zeros(np.shape(w[i]))) for i in range(len(w))])
    m, _ = X.shape # m es el número de muestras (filas en X)
    h_total = np.zeros((m, Y.shape[1])) # Matriz para guardar las predicciones de todas las muestras

    for i in range(m): # Itera sobre cada muestra individual en el conjunto de entrenamiento
        x = X[i] # La fila de características de la muestra actual
        y = Y[i].T # La etiqueta/salida real de la muestra actual (transpuesta a columna)

        # --- Propagación hacia adelante (Forward Propagation) ---
        a = x # 'a' representa las activaciones de la capa actual, comienza con la entrada
        a_s = [] # Lista para guardar las activaciones de cada capa (incluyendo el término de bias)
        
        # Itera a través de las capas de pesos para calcular la salida de la red
        for j in range(len(w)):
            a = np.mat(np.append(1, a)).T # Añade el término de bias (1) y transpone a vector columna
            a_s.append(a) # Guarda la activación 'a' (con bias) antes de pasar a la siguiente capa
            z = w[j] * a # Calcula la entrada ponderada 'z' para la siguiente capa (pesos * activaciones)
            a = sigmoid(z) # Aplica la función de activación sigmoide para obtener las activaciones 'a' de la siguiente capa
        h_total[i, :] = a.T # Guarda la salida final 'a' de la red para la muestra actual (transpuesta a fila)

        # --- Propagación hacia atrás (Backpropagation) ---
        delta = a - y # Calcula el error en la capa de salida (diferencia entre predicción y real)
        w_grad[-1] += delta * a_s[-1].T # Actualiza el gradiente para la última capa de pesos

        # Itera hacia atrás desde la penúltima capa hasta la segunda capa (excluyendo la de entrada)
        for j in reversed(range(1, len(w))):
            # Propaga el error 'delta' hacia atrás a través de los pesos y multiplica por la derivada de la activación
            # delta[1:] excluye el término de bias al propagar el error para calcular el gradiente de los pesos
            delta = np.multiply(w[j].T * delta, s_prime(a_s[j]))
            w_grad[j-1] += (delta[1:] * a_s[j-1].T) # Acumula el gradiente para la matriz de pesos de la capa actual

    # Promedia los gradientes sobre todas las muestras para obtener el gradiente promedio del costo
    w_grad = [w_grad_layer / m for w_grad_layer in w_grad]

    # Calcula la función de costo (Entropía Cruzada Binaria para múltiples salidas)
    # np.log(0) resultaría en advertencias o errores, por lo que se añaden pequeños valores para evitarlo
    epsilon_log = 1e-10 # Pequeño valor para estabilidad numérica
    J = (1.0 / m) * np.sum(-np.multiply(Y, np.log(h_total + epsilon_log)) - np.multiply((1 - Y), np.log(1 - h_total + epsilon_log)))

    return {'w_grad': w_grad, 'J': J, 'h': h_total}

# --- Definición de los parámetros de la red y los datos ---

# Conjunto de entrenamiento X: ¡IMPORTANTE: DEBE TENER 225 COLUMNAS!
# Cada fila es una muestra, cada columna es una característica.
# Este es un ejemplo con 4 muestras y 225 características aleatorias.
# REEMPLAZA ESTO CON TUS DATOS REALES.
X = np.random.rand(4, 225)
X = np.mat(X) # Asegúrate de que sea una matriz de NumPy

# Conjunto de entrenamiento Y: Con 6 salidas.
# Esto es un EJEMPLO de cómo Y debería verse para 4 muestras y 6 salidas (one-hot encoding).
# ADAPTA ESTO A LA LÓGICA DE TUS DATOS REALES.
Y = np.mat([[1,0,0,0,0,0], # Salida deseada para la 1ra muestra de X
            [0,1,0,0,0,0], # Salida deseada para la 2da muestra de X
            [0,0,1,0,0,0], # Salida deseada para la 3ra muestra de X
            [0,0,0,1,0,0]]) # Salida deseada para la 4ta muestra de X

# Configuración de la red neuronal:
# [Número de entradas, Capa Oculta 1, Capa Oculta 2, Capa Oculta 3, Número de salidas]
layers = [225, 45, 45, 45, 6]
epochs = 5000 # Número de iteraciones (cuántas veces se entrena la red con todos los datos)
alpha = 0.5 # Tasa de aprendizaje (qué tan grande es el paso de ajuste de los pesos)
epsilon = 1 # Rango para la inicialización aleatoria de los pesos (entre -1 y 1)

# --- Entrenamiento de la red ---

# Inicializa los pesos aleatoriamente
w = init_weights(layers, epsilon)
# Prepara el diccionario para guardar el historial de costo y predicciones
result = {'J': [], 'h': []}
# Diccionario para guardar gradientes de la primera y última época (para análisis)
w_s = {}

# Bucle principal de entrenamiento
for i in range(epochs):
    # Realiza un paso de forward y backpropagation
    fit_result = fit(X, Y, w)
    w_grad = fit_result.get('w_grad') # Obtiene los gradientes calculados
    J = fit_result.get('J')         # Obtiene el valor de la función de costo
    h_current = fit_result.get('h') # Obtiene las predicciones actuales

    # Guarda el costo y las predicciones en el historial
    result['J'].append(J)
    result['h'].append(h_current)

    # Actualiza los pesos de la red usando el Descenso de Gradiente
    # w[j] = w[j] - alpha * w_grad[j]
    for j in range(len(w)):
        w[j] -= alpha * w_grad[j]

    # Guarda una "instantánea" de los gradientes en la primera y última época
    if i == 0 or i == (epochs - 1):
        w_s['w_' + str(i)] = w_grad[:] # Usa [:] para crear una copia y evitar que se modifique

# --- Gráfico de la Función de Costo ---
# Esencial para visualizar el progreso del entrenamiento
plt.plot(result.get('J'))
plt.xlabel('Número de iteraciones (Épocas)')
plt.ylabel('Función de Costo (J)')
plt.title('Función de Costo a lo largo de las Épocas de Entrenamiento')
plt.grid(True) # Añade una cuadrícula para facilitar la lectura del gráfico
plt.show()

# --- Resultados del Entrenamiento (Para Consola de Visual Studio / Terminal) ---

print("\n" + "="*60)
print("             *** RESULTADOS DEL ENTRENAMIENTO DE LA RED NEURONAL *** ")
print("="*60 + "\n")

# Mostrar los gradientes de los pesos (iniciales y finales)
# Esto es útil para ver cómo los gradientes cambian del inicio al final del aprendizaje.
print("--- Gradientes de los pesos (Iniciales y Finales) ---")
for i_epoch_snap in [0, epochs - 1]:
    if i_epoch_snap == 0:
        print('\n> PESOS INICIALES (Gradientes después de la 1ª época)')
        print('------------------------------------------------------')
    elif i_epoch_snap == (epochs - 1):
        print('\n> GRADIENTES FINALES (de la última época)')
        print('-----------------------------------------')

    for layer_idx, grad_matrix in enumerate(w_s['w_' + str(i_epoch_snap)]):
        print(f'   Gradiente para la Capa Theta^({layer_idx + 1}):')
        print(grad_matrix)
    print('')

# Mostrar los PESOS finales reales de la red (lo que la red aprendió)
print("\n--- PESOS FINALES DE LA RED NEURONAL ENTRENADA ---")
print("Estos son los valores que la red ha aprendido para hacer sus predicciones.")
for layer_idx, final_w_matrix in enumerate(w):
    print(f'   Capa Theta^({layer_idx + 1}):')
    print(final_w_matrix)
print('')

# Mostrar la predicción de la red al inicio (cuando apenas estaba aprendiendo)
print("\n--- PREDICCIÓN INICIAL (después de la 1ª época) ---")
print('   Predicción h_Theta(x) = ')
print(result.get('h')[0])
print('')

# Mostrar la predicción final de la red (el resultado del aprendizaje)
print("\n--- PREDICCIÓN FINAL ESTIMADA (después de la última época) ---")
print('   Predicción h_Theta(x) = ')
print(result.get('h')[-1])
print("\n" + "="*60 + "\n")