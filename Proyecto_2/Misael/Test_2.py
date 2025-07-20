from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton,
                             QFileDialog, QLabel, QScrollArea, QWidget,
                             QVBoxLayout, QHBoxLayout, QMessageBox, QInputDialog) # Añadido QMessageBox, QInputDialog
from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap, QImage

import numpy as np
import matplotlib.pyplot as plt
import pickle # Para guardar y cargar objetos Python (como los pesos de la red)

# --- Nueva subclase para el QLabel de la imagen ---
class ZoomableImageLabel(QLabel):
    # Señal para emitir la posición del ratón
    mouseMoved = pyqtSignal(QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True) # Habilitar seguimiento del ratón sin presionar botón
        self._show_frame = False
        self._frame_pos = QPoint()
        self._frame_size = 15 # Tamaño del marco
        self._frame_half_size = self._frame_size // 2

    def set_show_frame(self, show):
        self._show_frame = show
        self.update() # Forzar redibujo

    def set_frame_position(self, pos):
        self._frame_pos = pos
        self.update() # Forzar redibujo

    def mouseMoveEvent(self, event):
        # Emitir la posición del ratón relativa a este widget
        self.mouseMoved.emit(event.pos())
        super().mouseMoveEvent(event) # Llama al método base para otros comportamientos

    def paintEvent(self, event):
        super().paintEvent(event) # Dibuja el pixmap primero

        if self._show_frame and not self.pixmap().isNull():
            painter = QPainter(self)
            pen = QPen(QColor(255, 0, 0), 2, Qt.SolidLine)
            painter.setPen(pen)

            # Calcular la posición del marco para que esté centrado en _frame_pos
            x = self._frame_pos.x() - self._frame_half_size
            y = self._frame_pos.y() - self._frame_half_size

            # Asegurarse de que el marco esté dentro de los límites del pixmap visible
            pixmap_rect = self.pixmap().rect()
            
            # Si el pixmap no ocupa todo el label, necesitamos la compensación
            px_offset_x = (self.width() - pixmap_rect.width()) // 2 if self.width() > pixmap_rect.width() else 0
            px_offset_y = (self.height() - pixmap_rect.height()) // 2 if self.height() > pixmap_rect.height() else 0

            # Ajustar la posición del marco para que esté relativa al top-left del pixmap dibujado
            # dentro del label
            frame_draw_x = x + px_offset_x
            frame_draw_y = y + px_offset_y

            # Dibujar el rectángulo
            painter.drawRect(frame_draw_x, frame_draw_y, self._frame_size, self._frame_size)
            painter.end()

# --- Funciones para la Red Neuronal (las mismas de tu código original) ---
# Función de activación
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivada de la sigmoidal
def s_prime(z):
    return np.multiply(z, 1.0-z)

# Inicialización de los pesos
def init_weights(layers, epsilon):
    weights = []
    for i in range(len(layers)-1):
        w = np.random.rand(layers[i+1], layers[i]+1)
        w = w * 2*epsilon - epsilon
        weights.append(np.mat(w))
    return weights

# Red Neuronal (modificada para permitir solo forward pass si no se entrena)
def fit_or_predict(X, Y, w, train_mode=True):
    w_grad = ([np.mat(np.zeros(np.shape(w[i])))
               for i in range(len(w))])
    m, n = X.shape
    h_total = np.zeros((m, 1))

    for i in range(m):
        x = X[i]
        a = x
        a_s = []
        for j in range(len(w)):
            a = np.mat(np.append(1, a)).T
            a_s.append(a)
            z = w[j] * a
            a = sigmoid(z)
        h_total[i, 0] = a

        if train_mode: # Solo hace backpropagation si está en modo entrenamiento
            y = Y[0,i]
            delta = a - y.T
            w_grad[-1] += delta * a_s[-1].T
            for j in reversed(range(1, len(w))):
                delta = np.multiply(w[j].T*delta, s_prime(a_s[j]))
                w_grad[j-1] += (delta[1:] * a_s[j-1].T)
                
    if train_mode:
        w_grad = [w_grad[i]/m for i in range(len(w))]
        J = (1.0 / m) * np.sum(-Y * np.log(h_total) - (np.array([[1]]) - Y) * np.log(1 - h_total))
        return {'w_grad': w_grad, 'J': J, 'h': h_total}
    else: # Modo predicción, solo devuelve las activaciones finales
        return h_total

# --- Clase MiVentana ---
class MiVentana(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hola, PyQt!")
        self.setGeometry(100, 100, 640, 480)

        # Variables para el marco y zoom
        self.marco_activo = False
        self.imagen_cargada = False
        self.zoom_label = QLabel()  # Label para mostrar el zoom
        self.current_zoom_pixmap = None # Guardará el último pixmap de zoom para "Guardar" y "Comprobar"

        # --- Base de datos simulada en memoria ---
        # Almacenará diccionarios como: {'vector': numpy_array, 'tipo': 'rio'}
        self.image_database = [] 
        self.image_types = {
            'rio': 0, 
            'desierto': 1, 
            'mar': 2, 
            'ciudad': 3, 
            'montaña': 4
        }
        self.reverse_image_types = {v: k for k, v in self.image_types.items()}

        # --- Configuración y entrenamiento de la Red Neuronal ---
        self.nn_layers = [300, 50, len(self.image_types)] # Ejemplo: 300 entradas (para imagen 100x200 RGB), 50 capa oculta, 5 salidas
        self.nn_epochs = 1000 # Unas 1000 épocas para este ejemplo, puede necesitar más
        self.nn_alpha = 0.1 # Tasa de aprendizaje
        self.nn_epsilon = 0.1 # Para inicializar pesos

        self.nn_weights = init_weights(self.nn_layers, self.nn_epsilon)
        self.load_nn_weights() # Intenta cargar pesos existentes al inicio

        # Widget central y layout principal
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout_principal = QHBoxLayout()
        central_widget.setLayout(layout_principal)

        # Contenedor para la imagen (400x350) con QScrollArea
        self.scroll_area = QScrollArea()
        self.scroll_area.setFixedSize(400, 350)
        self.scroll_area.setWidgetResizable(True)

        # Usar la nueva clase ZoomableImageLabel
        self.label_imagen = ZoomableImageLabel()
        self.label_imagen.mouseMoved.connect(self.on_image_mouse_move) # Conectar la señal
        self.scroll_area.setWidget(self.label_imagen)
        layout_principal.addWidget(self.scroll_area)

        # Layout para los botones y el zoom (derecha)
        layout_derecha = QVBoxLayout()

        # Botón "Abrir imagen"
        boton_imagen = QPushButton("Abrir imagen")
        boton_imagen.clicked.connect(self.abrir_imagen)
        layout_derecha.addWidget(boton_imagen)

        # Botón "Marco 15x15"
        boton_marco = QPushButton("Marco 15x15") # Cambiado el texto para más claridad
        boton_marco.clicked.connect(self.al_marco15x15)
        layout_derecha.addWidget(boton_marco)

        # Label para el zoom (200x100)
        self.zoom_label.setFixedSize(200, 100)
        self.zoom_label.setStyleSheet("border: 1px solid black;")
        layout_derecha.addWidget(self.zoom_label)

        # Botón para guardar el zoom en la base de datos de manera vectorial
        boton_guardar = QPushButton("Guardar Zoom")
        boton_guardar.clicked.connect(self.guardar_zoom_vectorial) # Conectar nueva función
        layout_derecha.addWidget(boton_guardar)

        # Botón para saber que tipo de imagen es (rio, desierto, mar, ciudad o montaña)
        boton_comprobar = QPushButton("Comprobar Tipo")
        boton_comprobar.clicked.connect(self.comprobar_tipo_imagen) # Conectar nueva función
        layout_derecha.addWidget(boton_comprobar)
        
        # Botón para entrenar la red (útil para generar datos de entrenamiento)
        boton_entrenar_nn = QPushButton("Entrenar Red Neuronal")
        boton_entrenar_nn.clicked.connect(self.entrenar_red_neuronal)
        layout_derecha.addWidget(boton_entrenar_nn)


        # Botón "Salir"
        boton_salir = QPushButton("Salir")
        boton_salir.clicked.connect(self.al_salir)
        layout_derecha.addStretch()
        layout_derecha.addWidget(boton_salir)

        layout_principal.addLayout(layout_derecha)

    def abrir_imagen(self):
        ruta_imagen, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar imagen", "",
            "Imágenes (*.png *.jpg *.jpeg *.bmp *.gif)"
        )

        if ruta_imagen:
            self.pixmap_original = QPixmap(ruta_imagen) # Guardar el pixmap original
            self.label_imagen.setPixmap(self.pixmap_original.scaled(
                self.scroll_area.width(), self.scroll_area.height(), 
                Qt.KeepAspectRatio, Qt.SmoothTransformation)) # Escalar para que se vea bien en el scroll area
            self.imagen_cargada = True
            self.current_zoom_pixmap = None # Resetear el zoom actual

    def al_marco15x15(self):
        self.marco_activo = not self.marco_activo
        self.label_imagen.set_show_frame(self.marco_activo) # Indicar al label si mostrar el marco
        print("Marco 15x15:", "Activado" if self.marco_activo else "Desactivado")
        if not self.marco_activo: # Si se desactiva el marco, limpiar el zoom
            self.zoom_label.clear()
            self.current_zoom_pixmap = None

    def al_salir(self):
        self.save_nn_weights() # Guardar pesos al salir
        QApplication.quit()

    def on_image_mouse_move(self, pos):
        # Esta es la posición del ratón dentro del ZoomableImageLabel
        if self.marco_activo and self.imagen_cargada:
            # Pasa la posición al label para que dibuje el marco
            self.label_imagen.set_frame_position(pos)
            self.actualizar_zoom(pos)

    def actualizar_zoom(self, mouse_pos_in_label):
        """Extrae la región 15x15 bajo el marco y la amplía a 200x100."""
        if not self.imagen_cargada or self.pixmap_original.isNull():
            return

        original_size = self.pixmap_original.size()
        label_size = self.label_imagen.size()
        
        scale_factor = min(label_size.width() / original_size.width(), label_size.height() / original_size.height())
        
        drawn_pixmap_w = original_size.width() * scale_factor
        drawn_pixmap_h = original_size.height() * scale_factor
        
        drawn_offset_x = (label_size.width() - drawn_pixmap_w) / 2
        drawn_offset_y = (label_size.height() - drawn_pixmap_h) / 2

        x_on_original = (mouse_pos_in_label.x() - drawn_offset_x) / scale_factor
        y_on_original = (mouse_pos_in_label.y() - drawn_offset_y) / scale_factor
        
        half_frame = self.label_imagen._frame_half_size
        
        x_region_orig = max(0, int(x_on_original - half_frame))
        y_region_orig = max(0, int(y_on_original - half_frame))

        width_region = self.label_imagen._frame_size
        height_region = self.label_imagen._frame_size
        
        if x_region_orig + width_region > original_size.width():
            width_region = original_size.width() - x_region_orig
        if y_region_orig + height_region > original_size.height():
            height_region = original_size.height() - y_region_orig
            
        if width_region <= 0 or height_region <= 0:
            self.zoom_label.clear()
            self.current_zoom_pixmap = None
            return
            
        region = self.pixmap_original.copy(QRect(x_region_orig, y_region_orig, width_region, height_region))

        zoom_pixmap = region.scaled(200, 100, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.zoom_label.setPixmap(zoom_pixmap)
        self.current_zoom_pixmap = zoom_pixmap # Guardar el pixmap actual del zoom

    # --- NUEVA FUNCIÓN: guardar_zoom_vectorial ---
    def guardar_zoom_vectorial(self):
        if self.current_zoom_pixmap is None:
            QMessageBox.warning(self, "Error", "No hay una región de zoom activa para guardar.")
            return

        # 1. Convertir QPixmap a QImage
        image = self.current_zoom_pixmap.toImage()
        if image.isNull():
            QMessageBox.warning(self, "Error", "No se pudo convertir el zoom a QImage.")
            return

        # 2. Extraer características (vectorizar)
        # Convertir QImage a numpy array (RGB) y aplanarlo
        # Las QImage de PyQt5 son a menudo 32-bit ARGB. Necesitamos convertir a un formato que NumPy pueda usar.
        # Aquí asumimos 3 canales (RGB) o 4 (RGBA) y aplanamos.
        
        # Convertir a formato RGB888 para consistencia
        if image.format() != QImage.Format_RGB32:
            image = image.convertToFormat(QImage.Format_RGB32)

        # Crear un array numpy desde los bytes de la imagen
        ptr = image.bits()
        # image.bits() returns a sip.voidptr, which does not support buffer interface for Python 3.
        # Need to convert to an actual Python bytes object.
        # Ensure the bytes object is correctly sized for RGB32 (width * height * 4 bytes per pixel)
        # Note: Qt's RGB32 is usually ARGB, so 4 bytes per pixel.
        arr = np.array(ptr).reshape((image.height(), image.width(), 4)) # RGBA
        
        # Si quieres solo RGB, puedes descartar el canal alfa (arr[:,:,:3])
        features = arr[:,:,:3].flatten() # Aplanar a un vector 1D (200 * 100 * 3 = 60000 elementos)
        
        # Normalizar las características para que estén en un rango de 0 a 1 (o similar)
        # Esto es importante para el entrenamiento de la red neuronal.
        features = features.astype(np.float32) / 255.0 # Normalizar a [0, 1]

        # 3. Pedir al usuario el tipo de imagen
        items = list(self.image_types.keys())
        item, ok = QInputDialog.getItem(self, "Guardar Imagen", 
                                         "Selecciona el tipo de imagen:", items, 0, False)
        
        if ok and item:
            label_numeric = self.image_types[item]
            self.image_database.append({'vector': features, 'tipo': item, 'label_numeric': label_numeric})
            QMessageBox.information(self, "Guardado", f"Imagen de tipo '{item}' guardada con éxito. Total: {len(self.image_database)} entradas.")
            print(f"Vector guardado para tipo '{item}'. Dimensión: {features.shape}. Valor de etiqueta: {label_numeric}")
        else:
            QMessageBox.warning(self, "Cancelado", "Operación de guardado cancelada.")

    # --- NUEVA FUNCIÓN: comprobar_tipo_imagen ---
    def comprobar_tipo_imagen(self):
        if self.current_zoom_pixmap is None:
            QMessageBox.warning(self, "Error", "No hay una región de zoom activa para comprobar.")
            return
        
        if not self.nn_weights:
            QMessageBox.warning(self, "Error", "La red neuronal no ha sido inicializada. Entrénala primero.")
            return

        # Extraer características del pixmap de zoom actual
        image = self.current_zoom_pixmap.toImage()
        if image.format() != QImage.Format_RGB32:
            image = image.convertToFormat(QImage.Format_RGB32)
        ptr = image.bits()
        arr = np.array(ptr).reshape((image.height(), image.width(), 4))
        features = arr[:,:,:3].flatten()
        features = features.astype(np.float32) / 255.0 # Normalizar

        # Asegurarse de que el vector de entrada tenga la dimensión correcta para la red neuronal
        # Si la red espera una sola entrada (1, num_features), formatea X_predict
        input_vector = np.mat(features).T # Transponer a una columna
        # El método fit_or_predict espera un array 2D donde cada fila es una muestra
        # Así que, si es una sola muestra, debe ser (1, num_features)
        X_predict = np.mat(features.reshape(1, -1))
        
        # Realizar la predicción usando la red neuronal (modo sin entrenamiento)
        # La función fit_or_predict se usa para esto, pasando train_mode=False
        predictions = fit_or_predict(X_predict, None, self.nn_weights, train_mode=False)
        
        # La salida de la red es un vector de probabilidades (una para cada clase)
        # La clase predicha es la que tiene la mayor probabilidad
        predicted_class_index = np.argmax(predictions)
        predicted_type = self.reverse_image_types.get(predicted_class_index, "Desconocido")
        
        # Opcional: mostrar las probabilidades
        probabilities_str = ", ".join([f"{self.reverse_image_types[i]}: {p[0]:.2f}" for i, p in enumerate(predictions)])

        QMessageBox.information(self, "Resultado de Comprobación", 
                                f"Tipo de imagen predicho: {predicted_type}\n"
                                f"Probabilidades: {probabilities_str}")
        print(f"Predicción: {predicted_type} (Índice: {predicted_class_index}, Probabilidades: {predictions.T})")

    # --- Función para entrenar la red neuronal con los datos guardados ---
    def entrenar_red_neuronal(self):
        if not self.image_database:
            QMessageBox.warning(self, "Advertencia", "No hay datos guardados para entrenar la red neuronal. Usa 'Guardar Zoom' primero.")
            return

        QMessageBox.information(self, "Entrenamiento", "Iniciando el entrenamiento de la red neuronal. Esto puede tardar unos segundos...")
        
        # Preparar los datos para el entrenamiento
        X_train_list = [entry['vector'] for entry in self.image_database]
        Y_train_list = [entry['label_numeric'] for entry in self.image_database]

        # Convertir a formato de NumPy matrices
        X_train = np.mat(X_train_list) # Cada fila es una muestra (vector de características)
        
        # Y_train debe ser one-hot encoded para clasificación multi-clase
        # Si tienes 5 tipos (0-4), Y_train debe ser una matriz (num_samples, 5)
        num_classes = len(self.image_types)
        Y_train_one_hot = np.zeros((len(Y_train_list), num_classes))
        for i, label_idx in enumerate(Y_train_list):
            Y_train_one_hot[i, label_idx] = 1
        Y_train = np.mat(Y_train_one_hot).T # Transponer para que sea (num_classes, num_samples) como espera `fit`

        # Re-inicializar los pesos o usar los actuales
        self.nn_weights = init_weights(self.nn_layers, self.nn_epsilon) # Opcional: reiniciar pesos
        
        # Entrenar la red
        for i in range(self.nn_epochs):
            fit_result = fit_or_predict(X_train, Y_train, self.nn_weights, train_mode=True)
            w_grad = fit_result.get('w_grad')
            J = fit_result.get('J')
            
            for j in range(len(self.nn_weights)):
                self.nn_weights[j] -= self.nn_alpha * w_grad[j]
            
            if i % 100 == 0: # Imprimir el costo cada 100 épocas
                print(f"Época {i}/{self.nn_epochs}, Costo: {J:.4f}")
        
        self.save_nn_weights() # Guardar los pesos entrenados
        QMessageBox.information(self, "Entrenamiento Completado", "La red neuronal ha sido entrenada con los datos guardados.")

    def save_nn_weights(self):
        """Guarda los pesos de la red neuronal en un archivo."""
        try:
            with open("nn_weights.pkl", "wb") as f:
                pickle.dump(self.nn_weights, f)
            print("Pesos de la red neuronal guardados exitosamente.")
        except Exception as e:
            print(f"Error al guardar los pesos de la red neuronal: {e}")

    def load_nn_weights(self):
        """Carga los pesos de la red neuronal desde un archivo."""
        try:
            with open("nn_weights.pkl", "rb") as f:
                self.nn_weights = pickle.load(f)
            print("Pesos de la red neuronal cargados exitosamente.")
        except FileNotFoundError:
            print("No se encontraron pesos de red neuronal guardados. Se usarán pesos iniciales aleatorios.")
        except Exception as e:
            print(f"Error al cargar los pesos de la red neuronal: {e}")


if __name__ == "__main__":
    app = QApplication([])
    ventana = MiVentana()
    ventana.show()
    app.exec_()