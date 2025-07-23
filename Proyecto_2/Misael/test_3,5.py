from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton,
                             QFileDialog, QLabel, QScrollArea, QWidget,
                             QVBoxLayout, QHBoxLayout, QMessageBox)
from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap, QImage

import numpy as np
from scipy.fft import fft2, fftshift # Importar para FFT
import os # Importar para manejo de rutas de archivos
from PIL import Image # Importar para cargar imágenes (requiere 'pip install Pillow')

# --- Nueva subclase para el QLabel de la imagen ---
# (Esta clase es necesaria para manejar correctamente el dibujado del marco
# y la emisión de señales de movimiento del ratón solo dentro del label de la imagen)
class ZoomableImageLabel(QLabel):
    # Señal para emitir la posición del ratón
    mouseMoved = pyqtSignal(QPoint)
    # Nueva señal para emitir la posición al hacer clic
    mouseClicked = pyqtSignal(QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True) # Habilitar seguimiento del ratón sin presionar botón
        self._show_frame = False
        self._frame_pos = QPoint()
        self._frame_size = 15 # Tamaño del marco
        self._frame_half_size = self._frame_size // 2
        self._pixmap = QPixmap() # Almacena el pixmap para su uso interno

    def setPixmap(self, pixmap):
        super().setPixmap(pixmap)
        self._pixmap = pixmap # Guardar la referencia del pixmap
        self.update()

    def pixmap(self):
        # Sobreescribir el método pixmap para devolver nuestra referencia interna
        return self._pixmap

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

    def mousePressEvent(self, event):
        # Emitir la posición del ratón al hacer clic
        self.mouseClicked.emit(event.pos())
        super().mousePressEvent(event)

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


class MiVentana(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visor de Imágenes con Zoom")
        self.setGeometry(100, 100, 640, 480)
        
        # Variables para el marco y zoom
        self.marco_activo = False
        self.imagen_cargada = False
        self.pixmap_original = QPixmap() # Almacena el pixmap original sin escalar
        
        # Nuevas variables para fijar el zoom
        self.zoom_is_fixed = False
        self.fixed_zoom_position = QPoint()

        # Variables para los datos de ML
        self.X_data = None
        self.y_labels = None
        
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
        # Conectar la señal mouseMoved de ZoomableImageLabel a nuestro manejador
        self.label_imagen.mouseMoved.connect(self.on_image_mouse_move)
        # Conectar la nueva señal mouseClicked a nuestro manejador
        self.label_imagen.mouseClicked.connect(self.on_image_clicked)
        self.scroll_area.setWidget(self.label_imagen)
        layout_principal.addWidget(self.scroll_area)
        
        # Layout para los botones y el zoom (derecha)
        layout_derecha = QVBoxLayout()
        
        # Botón "Abrir imagen"
        boton_imagen = QPushButton("Abrir imagen")
        boton_imagen.clicked.connect(self.abrir_imagen)
        layout_derecha.addWidget(boton_imagen)
        
        # Botón "Marco 15x15" (se hace atributo para control de visibilidad)
        self.boton_marco = QPushButton("Marco 15x15")
        self.boton_marco.clicked.connect(self.al_marco15x15)
        layout_derecha.addWidget(self.boton_marco)
        
        # Label para el zoom (200x100) (se hace atributo para control de visibilidad)
        self.zoom_label = QLabel()
        self.zoom_label.setFixedSize(200, 100)
        self.zoom_label.setStyleSheet("border: 1px solid black;")
        layout_derecha.addWidget(self.zoom_label)

        # --- NUEVOS BOTONES ---
        # Botón "Guardar"
        self.boton_guardar = QPushButton("Guardar")
        self.boton_guardar.clicked.connect(self.guardar_zoom_bn)
        layout_derecha.addWidget(self.boton_guardar)

        # Botón "Comprobación"
        self.boton_comprobacion = QPushButton("Comprobación")
        self.boton_comprobacion.clicked.connect(self.realizar_comprobacion)
        layout_derecha.addWidget(self.boton_comprobacion)

        # Botón para cargar datos de ML
        self.boton_cargar_ml = QPushButton("Cargar Datos ML")
        self.boton_cargar_ml.clicked.connect(self.cargar_datos_ml)
        layout_derecha.addWidget(self.boton_cargar_ml)

        # NUEVO: Botón para guardar datos de ML
        self.boton_guardar_ml = QPushButton("Guardar Datos ML")
        self.boton_guardar_ml.clicked.connect(self.guardar_datos_ml)
        layout_derecha.addWidget(self.boton_guardar_ml)
        # --- FIN NUEVOS BOTONES ---
        
        # Ocultar inicialmente los botones de marco, zoom, guardar y comprobación
        self.boton_marco.setVisible(False)
        self.zoom_label.setVisible(False)
        self.boton_guardar.setVisible(False)
        self.boton_comprobacion.setVisible(False)
        self.boton_guardar_ml.setVisible(False) # Ocultar inicialmente el botón de guardar ML

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
            self.pixmap_original = QPixmap(ruta_imagen) # Guardar el pixmap original sin escalar
            # Escalar el pixmap para que quepa en el scroll area y se vea bien
            self.label_imagen.setPixmap(self.pixmap_original.scaled(
                self.scroll_area.width(), self.scroll_area.height(), 
                Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.imagen_cargada = True
            
            # Mostrar los botones y el label de zoom
            self.boton_marco.setVisible(True)
            self.zoom_label.setVisible(True)
            self.boton_guardar.setVisible(True) # Mostrar el botón "Guardar"
            self.boton_comprobacion.setVisible(True) # Mostrar el botón "Comprobación"
            
            # Asegurarse de que el marco esté desactivado y el zoom limpio al cargar nueva imagen
            self.marco_activo = False
            self.zoom_is_fixed = False # Reiniciar el estado de fijado del zoom
            self.label_imagen.set_show_frame(False)
            self.zoom_label.clear()


    def al_marco15x15(self):
        self.marco_activo = not self.marco_activo
        # Indicar al label si debe dibujar el marco
        self.label_imagen.set_show_frame(self.marco_activo) 
        print("Marco 15x15:", "Activado" if self.marco_activo else "Desactivado")
        
        # Si se desactiva el marco, limpiar el zoom y resetear el estado de fijado
        if not self.marco_activo: 
            self.zoom_label.clear()
            self.zoom_is_fixed = False
            # Mover el marco fuera de la vista cuando se desactiva
            self.label_imagen.set_frame_position(QPoint(-1000, -1000))

    def al_salir(self):
        QApplication.quit()

    def on_image_clicked(self, pos):
        """Maneja el evento de clic del ratón para fijar el zoom."""
        if self.marco_activo and self.imagen_cargada:
            self.zoom_is_fixed = True
            self.fixed_zoom_position = pos
            self.label_imagen.set_frame_position(pos) # Fija la posición del marco
            self.actualizar_zoom(pos) # Actualiza el zoom a la posición fijada

    def on_image_mouse_move(self, pos):
        # Esta es la posición del ratón dentro del ZoomableImageLabel
        # Solo actualiza el zoom si el marco está activo y el zoom no está fijado
        if self.marco_activo and self.imagen_cargada and not self.zoom_is_fixed:
            # Pasa la posición al label para que dibuje el marco
            self.label_imagen.set_frame_position(pos)
            self.actualizar_zoom(pos)
        elif self.marco_activo and self.imagen_cargada and self.zoom_is_fixed:
            # Si el zoom está fijado, asegúrate de que el marco se mantenga en la posición fijada
            self.label_imagen.set_frame_position(self.fixed_zoom_position)


    def actualizar_zoom(self, mouse_pos_in_label):
        """Extrae la región 15x15 bajo el marco y la amplía a 200x100."""
        if not self.imagen_cargada or self.pixmap_original.isNull():
            return
        
        original_size = self.pixmap_original.size()
        label_size = self.label_imagen.size()
        
        # Calcular el factor de escala y los offsets del pixmap dibujado dentro del label
        # Esto es crucial porque el pixmap se escala para ajustarse al label,
        # y la posición del ratón está en las coordenadas del label, no del pixmap original.
        scale_factor = min(label_size.width() / original_size.width(), label_size.height() / original_size.height())
        
        drawn_pixmap_w = original_size.width() * scale_factor
        drawn_pixmap_h = original_size.height() * scale_factor
        
        drawn_offset_x = (label_size.width() - drawn_pixmap_w) / 2
        drawn_offset_y = (label_size.height() - drawn_pixmap_h) / 2

        # Convertir la posición del ratón (en el label) a la posición correspondiente en el pixmap original
        x_on_original = (mouse_pos_in_label.x() - drawn_offset_x) / scale_factor
        y_on_original = (mouse_pos_in_label.y() - drawn_offset_y) / scale_factor
        
        half_frame = self.label_imagen._frame_half_size # Usar el atributo del tamaño del marco

        # Calcular la región a copiar, asegurándose de no ir más allá de los bordes de la imagen original
        x_region_orig = max(0, int(x_on_original - half_frame))
        y_region_orig = max(0, int(y_on_original - half_frame))

        width_region = self.label_imagen._frame_size
        height_region = self.label_imagen._frame_size
        
        # Ajustar el tamaño de la región si se acerca a los bordes de la imagen original
        if x_region_orig + width_region > original_size.width():
            width_region = original_size.width() - x_region_orig
        if y_region_orig + height_region > original_size.height():
            height_region = original_size.height() - y_region_orig
            
        # Si la región es inválida (por ejemplo, se sale completamente de la imagen), limpiar el zoom
        if width_region <= 0 or height_region <= 0:
            self.zoom_label.clear()
            return
            
        # Copiar la región del pixmap original
        region = self.pixmap_original.copy(QRect(x_region_orig, y_region_orig, width_region, height_region))
        
        # Escalar a 200x100 (zoom de ~13x)
        zoom_pixmap = region.scaled(200, 100, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.zoom_label.setPixmap(zoom_pixmap)

    def guardar_zoom_bn(self):
        """Guarda la imagen del zoom en blanco y negro en formato .png."""
        if not self.imagen_cargada or self.pixmap_original.isNull():
            QMessageBox.warning(self, "Error al guardar", "No hay imagen cargada para guardar.")
            return
        
        if not self.marco_activo:
            QMessageBox.warning(self, "Error al guardar", "El marco 15x15 debe estar activo para guardar el zoom.")
            return

        # Determinar la posición del marco para extraer la región 15x15
        if self.zoom_is_fixed:
            current_zoom_pos = self.fixed_zoom_position
        else:
            # Si no está fijado, usamos la última posición del ratón que actualizó el marco
            current_zoom_pos = self.label_imagen._frame_pos

        original_size = self.pixmap_original.size()
        label_size = self.label_imagen.size()
        
        # Calcular el factor de escala y los offsets del pixmap dibujado dentro del label
        scale_factor = min(label_size.width() / original_size.width(), label_size.height() / original_size.height())
        
        drawn_pixmap_w = original_size.width() * scale_factor
        drawn_pixmap_h = original_size.height() * scale_factor
        
        drawn_offset_x = (label_size.width() - drawn_pixmap_w) / 2
        drawn_offset_y = (label_size.height() - drawn_pixmap_h) / 2

        # Convertir la posición del marco (en el label) a la posición correspondiente en el pixmap original
        x_on_original = (current_zoom_pos.x() - drawn_offset_x) / scale_factor
        y_on_original = (current_zoom_pos.y() - drawn_offset_y) / scale_factor
        
        half_frame = self.label_imagen._frame_half_size

        # Calcular la región a copiar, asegurándose de no ir más allá de los bordes de la imagen original
        x_region_orig = max(0, int(x_on_original - half_frame))
        y_region_orig = max(0, int(y_on_original - half_frame))

        width_region = self.label_imagen._frame_size
        height_region = self.label_imagen._frame_size
        
        # Ajustar el tamaño de la región si se acerca a los bordes de la imagen original
        if x_region_orig + width_region > original_size.width():
            width_region = original_size.width() - x_region_orig
        if y_region_orig + height_region > original_size.height():
            height_region = original_size.height() - y_region_orig
            
        # Si la región es inválida, no se puede guardar
        if width_region <= 0 or height_region <= 0:
            QMessageBox.warning(self, "Error al guardar", "La región de 15x15 es inválida o está fuera de la imagen.")
            return
            
        # Copiar la región de 15x15 píxeles del pixmap original
        region_15x15 = self.pixmap_original.copy(QRect(x_region_orig, y_region_orig, width_region, height_region))
        
        # Convertir QPixmap a QImage para poder manipular los píxeles
        image_to_save = region_15x15.toImage()

        # Convertir la imagen a blanco y negro (escala de grises)
        if image_to_save.format() != QImage.Format_Grayscale8:
            image_to_save = image_to_save.convertToFormat(QImage.Format_Grayscale8)

        # Abrir diálogo para guardar el archivo
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar imagen de zoom en B/N (15x15)", "zoom_15x15_bn.png",
            "Imágenes PNG (*.png);;Todos los archivos (*)"
        )

        if file_path:
            if image_to_save.save(file_path, "PNG"):
                QMessageBox.information(self, "Guardado exitoso", f"Imagen 15x15 B/N guardada en: {file_path}")
            else:
                QMessageBox.critical(self, "Error al guardar", "No se pudo guardar la imagen.")

    def realizar_comprobacion(self):
        """Método para el botón 'Comprobación' (actualmente no hace nada significativo)."""
        print("Botón 'Comprobación' presionado. (Funcionalidad pendiente)")
        QMessageBox.information(self, "Comprobación", "El botón 'Comprobación' fue presionado. Por ahora, no hace nada.")

    def guardar_datos_ml(self):
        """
        Guarda las matrices X_data y y_labels en archivos de texto para su uso futuro.
        """
        if self.X_data is None or self.y_labels is None:
            QMessageBox.warning(self, "Guardar Datos ML", "No hay datos de ML cargados para guardar.")
            return

        x_file_path = os.path.join(os.path.dirname(__file__), "entrenamiento_X.txt")
        y_file_path = os.path.join(os.path.dirname(__file__), "entrenamiento_y.txt")

        try:
            # Guardar X_data. Convertir a array si es necesario para savetxt, aunque asmatrix funciona.
            # Usar un formato de alta precisión para evitar pérdida de datos con FFT.
            np.savetxt(x_file_path, self.X_data, fmt='%.18e', delimiter=',')
            # Guardar y_labels. Asegurarse de que sea 1D para savetxt si es necesario, aunque asmatrix funciona.
            np.savetxt(y_file_path, self.y_labels, fmt='%d', delimiter=',') # Etiquetas son enteros

            QMessageBox.information(self, "Guardar Datos ML", 
                                    f"Datos de entrenamiento guardados exitosamente en:\n"
                                    f"{x_file_path}\n{y_file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error al Guardar Datos ML", f"No se pudieron guardar los datos: {e}")
            print(f"Error al guardar datos ML: {e}")


    def cargar_datos_ml(self):
        """
        Carga y preprocesa imágenes de 15x15 píxeles desde la carpeta 'Data_set'
        (con subcarpetas C1-C6) con FFT para crear las matrices X y y.
        Si los archivos de entrenamiento ya existen, los carga directamente.
        """
        x_file_path = os.path.join(os.path.dirname(__file__), "entrenamiento_X.txt")
        y_file_path = os.path.join(os.path.dirname(__file__), "entrenamiento_y.txt")

        # Intentar cargar desde archivos si existen
        if os.path.exists(x_file_path) and os.path.exists(y_file_path):
            try:
                QMessageBox.information(self, "Cargando Datos ML", 
                                        "Cargando datos de entrenamiento desde archivos guardados. Esto es más rápido...")
                self.X_data = np.asmatrix(np.loadtxt(x_file_path, delimiter=','))
                self.y_labels = np.asmatrix(np.loadtxt(y_file_path, delimiter=','))
                
                # Asegurarse de que y_labels sea una matriz fila (1xN) si se cargó como 1D
                if self.y_labels.ndim == 1:
                    self.y_labels = self.y_labels.reshape(1, -1)

                QMessageBox.information(self, "Datos ML Cargados", 
                                        f"Datos de entrenamiento cargados desde archivos.\n"
                                        f"Dimensión de X: {self.X_data.shape}\n"
                                        f"Dimensión de y: {self.y_labels.shape}")
                self.boton_guardar_ml.setVisible(True) # Mostrar el botón de guardar
                print(f"X (primeras 5 filas):\n{self.X_data[:min(5, self.X_data.shape[0]), :min(5, self.X_data.shape[1])]}...")
                print(f"y (primeros 10 elementos):\n{self.y_labels[:, :min(10, self.y_labels.shape[1])]}")
                return # Salir de la función si los datos se cargaron con éxito
            except Exception as e:
                QMessageBox.warning(self, "Error al Cargar Datos ML", 
                                    f"Error al cargar datos desde archivos, se procederá a procesar las imágenes: {e}")
                print(f"Error al cargar datos desde archivos: {e}")
                # Si hay un error al cargar, continuar para procesar desde imágenes

        # Si los archivos no existen o hubo un error al cargarlos, procesar desde imágenes
        image_size = 15
        vector_dim = image_size * image_size # 225

        X_list = []
        y_list = []

        base_path = os.path.join(os.path.dirname(__file__), "Data_set")

        class_labels = {
            "C1": 1, # agua
            "C2": 2, # vegetación
            "C3": 3, # montañas
            "C4": 4, # desiertos
            "C5": 5, # ríos
            "C6": 6  # ciudades
        }

        QMessageBox.information(self, "Cargando Datos ML", 
                                "Cargando y procesando imágenes desde 'Data_set'. Esto puede tomar un momento...")

        total_images_loaded = 0
        
        for class_folder_name, label in class_labels.items():
            folder_path = os.path.join(base_path, class_folder_name)
            
            if not os.path.isdir(folder_path):
                print(f"Advertencia: La carpeta '{folder_path}' no existe. Saltando.")
                QMessageBox.warning(self, "Error de Ruta", f"La carpeta '{folder_path}' no se encontró. Asegúrate de que 'Data_set' y sus subcarpetas estén en el mismo directorio que el script.")
                continue

            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    image_path = os.path.join(folder_path, filename)
                    try:
                        img = Image.open(image_path)
                        
                        if img is None:
                            print(f"Error: No se pudo abrir la imagen {image_path}. Puede estar corrupta o no ser un formato válido.")
                            continue

                        img = img.convert('L')
                        img = img.resize((image_size, image_size))
                        
                        image_array = np.array(img)
                        
                        if image_array.ndim != 2 or image_array.shape != (image_size, image_size):
                             print(f"Error: La imagen {image_path} no tiene el formato esperado (15x15 en escala de grises) después de la conversión. Dimensiones: {image_array.shape}")
                             continue

                        fft_image = fft2(image_array)
                        fft_shifted = fftshift(fft_image)
                        fft_magnitude = np.abs(fft_shifted)

                        vectorized_image = fft_magnitude.flatten()
                        X_list.append(vectorized_image)

                        y_list.append(label)
                        total_images_loaded += 1

                    except Exception as e:
                        print(f"Error al procesar la imagen {image_path}: {e}")
                        continue

        if X_list:
            self.X_data = np.asmatrix(X_list)
            self.y_labels = np.asmatrix(y_list)

            QMessageBox.information(self, "Datos ML Cargados", 
                                    f"Datos de entrenamiento cargados y preprocesados.\n"
                                    f"Total de imágenes cargadas: {total_images_loaded}\n"
                                    f"Dimensión de X: {self.X_data.shape}\n"
                                    f"Dimensión de y: {self.y_labels.shape}")
            
            print(f"X (primeras 5 filas):\n{self.X_data[:min(5, self.X_data.shape[0]), :min(5, self.X_data.shape[1])]}...")
            print(f"y (primeros 10 elementos):\n{self.y_labels[:, :min(10, self.y_labels.shape[1])]}")
            
            # NUEVO: Guardar los datos recién procesados para futuras ejecuciones
            self.guardar_datos_ml()
            self.boton_guardar_ml.setVisible(True) # Asegurarse de que el botón de guardar esté visible
        else:
            self.X_data = None
            self.y_labels = None
            QMessageBox.warning(self, "Carga de Datos ML", "No se encontraron imágenes válidas para cargar en el conjunto de datos.")
            print("No se encontraron imágenes válidas para cargar en el conjunto de datos.")
            self.boton_guardar_ml.setVisible(False) # Ocultar el botón si no hay datos para guardar


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
        weights.append(np.asmatrix(w))
    return weights

# Red Neuronal
#==============
def fit(X, Y, w):
    # Inicialización de cada parámetro con gradiente igual a 0
    w_grad = ([np.asmatrix(np.zeros(np.shape(w[i])))
              for i in range(len(w))])  # len(w) es igual al número de capas
    m, n = X.shape
    h_total = np.zeros((m, 1))  # Valor predecido de todas las muestras, m*1, probabilidad
    for i in range(m):
        x = X[i].T # Transponer x para que sea una columna
        y = Y[0,i]
        # Propagación hacia adelante
        #============================
        a = x
        a_s = []
        for j in range(len(w)):
            a = np.asmatrix(np.append(1, a.T)).T # Añadir bias y transponer
            a_s.append(a)  # Aquí se guarda el valor a de la capa L-1 anterior.
            z = w[j] * a
            a = sigmoid(z)
        h_total[i, 0] = a
        # Propagación hacia atras (backpropagation)
        #===========================================
        delta = a - y.T
        w_grad[-1] += delta * a_s[-1].T  # Gradiente de la capa L-1
        # Reverso, desde la penúltima capa hasta el final de la segunda capa, excluyendo la primera y la última capa
        for j in reversed(range(1, len(w))):
            # Excluir el término de bias de delta al multiplicar por w[j].T
            delta = np.multiply(w[j].T[1:]*delta, s_prime(a_s[j][1:])) # El parámetro pasado aquí es a, No z
            w_grad[j-1] += (delta * a_s[j-1].T)
    
    # Ajustar el cálculo del costo para múltiples clases si es necesario,
    # pero para una salida sigmoidal simple (0 o 1) el costo de regresión logística es:
    J = (1.0 / m) * np.sum(-Y * np.log(h_total) - (np.array([[1]]) - Y) * np.log(1 - h_total))
    return {'w_grad': w_grad, 'J': J, 'h': h_total}


if __name__ == "__main__":
    app = QApplication([])
    ventana = MiVentana()
    ventana.show()
    app.exec_()