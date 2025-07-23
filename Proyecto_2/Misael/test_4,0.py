from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton,
                             QFileDialog, QLabel, QScrollArea, QWidget,
                             QVBoxLayout, QHBoxLayout, QMessageBox, QProgressDialog)
from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap, QImage

import numpy as np
from scipy.fft import fft2, fftshift
import os
from PIL import Image

# --- Clases de la Interfaz Gráfica (PyQt5) ---
class ZoomableImageLabel(QLabel):
    mouseMoved = pyqtSignal(QPoint)
    mouseClicked = pyqtSignal(QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self._show_frame = False
        self._frame_pos = QPoint()
        self._frame_size = 15
        self._frame_half_size = self._frame_size // 2
        self._pixmap = QPixmap()

    def setPixmap(self, pixmap):
        super().setPixmap(pixmap)
        self._pixmap = pixmap
        self.update()

    def pixmap(self):
        return self._pixmap

    def set_show_frame(self, show):
        self._show_frame = show
        self.update()

    def set_frame_position(self, pos):
        self._frame_pos = pos
        self.update()

    def mouseMoveEvent(self, event):
        self.mouseMoved.emit(event.pos())
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        self.mouseClicked.emit(event.pos())
        super().mousePressEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)

        if self._show_frame and not self.pixmap().isNull():
            painter = QPainter(self)
            pen = QPen(QColor(255, 0, 0), 2, Qt.SolidLine)
            painter.setPen(pen)

            x = self._frame_pos.x() - self._frame_half_size
            y = self._frame_pos.y() - self._frame_half_size

            pixmap_rect = self.pixmap().rect()
            
            px_offset_x = (self.width() - pixmap_rect.width()) // 2 if self.width() > pixmap_rect.width() else 0
            px_offset_y = (self.height() - pixmap_rect.height()) // 2 if self.height() > pixmap_rect.height() else 0

            frame_draw_x = x + px_offset_x
            frame_draw_y = y + px_offset_y

            painter.drawRect(frame_draw_x, frame_draw_y, self._frame_size, self._frame_size)
            painter.end()


class MiVentana(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visor de Imágenes con Zoom y Clasificación ML")
        self.setGeometry(100, 100, 640, 480)
        
        self.marco_activo = False
        self.imagen_cargada = False
        self.pixmap_original = QPixmap()
        
        self.zoom_is_fixed = False
        self.fixed_zoom_position = QPoint()

        self.X_data = None
        self.y_labels = None
        self.trained_weights = None # Para almacenar los pesos entrenados de la red neuronal

        # Mapeo de etiquetas numéricas a nombres de clase
        self.class_names = {
            1: "agua (mares, lagos, océanos)",
            2: "vegetación (selva, bosques)",
            3: "montañas (cordillera, cerros)",
            4: "desiertos",
            5: "ríos",
            6: "ciudades"
        }
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout_principal = QHBoxLayout()
        central_widget.setLayout(layout_principal)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setFixedSize(400, 350)
        self.scroll_area.setWidgetResizable(True)
        
        self.label_imagen = ZoomableImageLabel()
        self.label_imagen.mouseMoved.connect(self.on_image_mouse_move)
        self.label_imagen.mouseClicked.connect(self.on_image_clicked)
        self.scroll_area.setWidget(self.label_imagen)
        layout_principal.addWidget(self.scroll_area)
        
        layout_derecha = QVBoxLayout()
        
        boton_imagen = QPushButton("Abrir imagen")
        boton_imagen.clicked.connect(self.abrir_imagen)
        layout_derecha.addWidget(boton_imagen)
        
        self.boton_marco = QPushButton("Marco 15x15")
        self.boton_marco.clicked.connect(self.al_marco15x15)
        layout_derecha.addWidget(self.boton_marco)
        
        self.zoom_label = QLabel()
        self.zoom_label.setFixedSize(200, 100)
        self.zoom_label.setStyleSheet("border: 1px solid black;")
        layout_derecha.addWidget(self.zoom_label)

        self.boton_guardar = QPushButton("Guardar")
        self.boton_guardar.clicked.connect(self.guardar_zoom_bn)
        layout_derecha.addWidget(self.boton_guardar)

        self.boton_comprobacion = QPushButton("Comprobación")
        self.boton_comprobacion.clicked.connect(self.realizar_comprobacion)
        layout_derecha.addWidget(self.boton_comprobacion)

        self.boton_cargar_ml = QPushButton("Cargar Datos ML")
        self.boton_cargar_ml.clicked.connect(self.cargar_datos_ml)
        layout_derecha.addWidget(self.boton_cargar_ml)

        self.boton_guardar_ml = QPushButton("Guardar Datos ML")
        self.boton_guardar_ml.clicked.connect(self.guardar_datos_ml)
        layout_derecha.addWidget(self.boton_guardar_ml)

        # Botón para entrenar el modelo ML
        self.boton_entrenar_ml = QPushButton("Entrenar Modelo ML")
        self.boton_entrenar_ml.clicked.connect(self.train_ml_model)
        layout_derecha.addWidget(self.boton_entrenar_ml)
        
        self.boton_marco.setVisible(False)
        self.zoom_label.setVisible(False)
        self.boton_guardar.setVisible(False)
        self.boton_comprobacion.setVisible(False)
        self.boton_guardar_ml.setVisible(False)
        self.boton_entrenar_ml.setVisible(False) 

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
            self.pixmap_original = QPixmap(ruta_imagen)
            self.label_imagen.setPixmap(self.pixmap_original.scaled(
                self.scroll_area.width(), self.scroll_area.height(), 
                Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.imagen_cargada = True
            
            self.boton_marco.setVisible(True)
            self.zoom_label.setVisible(True)
            self.boton_guardar.setVisible(True)
            self.boton_comprobacion.setVisible(True)
            
            self.marco_activo = False
            self.zoom_is_fixed = False
            self.label_imagen.set_show_frame(False)
            self.zoom_label.clear()

    def al_marco15x15(self):
        self.marco_activo = not self.marco_activo
        self.label_imagen.set_show_frame(self.marco_activo) 
        print("Marco 15x15:", "Activado" if self.marco_activo else "Desactivado")
        
        if not self.marco_activo: 
            self.zoom_label.clear()
            self.zoom_is_fixed = False
            self.label_imagen.set_frame_position(QPoint(-1000, -1000))

    def al_salir(self):
        QApplication.quit()

    def on_image_clicked(self, pos):
        if self.marco_activo and self.imagen_cargada:
            self.zoom_is_fixed = True
            self.fixed_zoom_position = pos
            self.label_imagen.set_frame_position(pos)
            self.actualizar_zoom(pos)

    def on_image_mouse_move(self, pos):
        if self.marco_activo and self.imagen_cargada and not self.zoom_is_fixed:
            self.label_imagen.set_frame_position(pos)
            self.actualizar_zoom(pos)
        elif self.marco_activo and self.imagen_cargada and self.zoom_is_fixed:
            self.label_imagen.set_frame_position(self.fixed_zoom_position)

    def actualizar_zoom(self, mouse_pos_in_label):
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
            return
            
        region = self.pixmap_original.copy(QRect(x_region_orig, y_region_orig, width_region, height_region))
        
        zoom_pixmap = region.scaled(200, 100, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.zoom_label.setPixmap(zoom_pixmap)

    def guardar_zoom_bn(self):
        if not self.imagen_cargada or self.pixmap_original.isNull():
            QMessageBox.warning(self, "Error al guardar", "No hay imagen cargada para guardar.")
            return
        
        if not self.marco_activo:
            QMessageBox.warning(self, "Error al guardar", "El marco 15x15 debe estar activo para guardar el zoom.")
            return

        if self.zoom_is_fixed:
            current_zoom_pos = self.fixed_zoom_position
        else:
            current_zoom_pos = self.label_imagen._frame_pos

        original_size = self.pixmap_original.size()
        label_size = self.label_imagen.size()
        
        scale_factor = min(label_size.width() / original_size.width(), label_size.height() / original_size.height())
        
        drawn_pixmap_w = original_size.width() * scale_factor
        drawn_pixmap_h = original_size.height() * scale_factor
        
        drawn_offset_x = (label_size.width() - drawn_pixmap_w) / 2
        drawn_offset_y = (label_size.height() - drawn_pixmap_h) / 2

        x_on_original = (current_zoom_pos.x() - drawn_offset_x) / scale_factor
        y_on_original = (current_zoom_pos.y() - drawn_offset_y) / scale_factor
        
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
            QMessageBox.warning(self, "Error al guardar", "La región de 15x15 es inválida o está fuera de la imagen.")
            return
            
        region_15x15 = self.pixmap_original.copy(QRect(x_region_orig, y_region_orig, width_region, height_region))
        
        image_to_save = region_15x15.toImage()

        if image_to_save.format() != QImage.Format_Grayscale8:
            image_to_save = image_to_save.convertToFormat(QImage.Format_Grayscale8)

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
        """
        Método para el botón 'Comprobación'.
        Predice la clase de la imagen bajo el zoom.
        """
        if not self.imagen_cargada or self.pixmap_original.isNull():
            QMessageBox.warning(self, "Comprobación", "Primero, abre una imagen.")
            return
        
        if not self.marco_activo:
            QMessageBox.warning(self, "Comprobación", "Activa el 'Marco 15x15' para seleccionar una región.")
            return

        if self.trained_weights is None:
            QMessageBox.warning(self, "Comprobación", "El modelo ML no ha sido entrenado. Por favor, haz clic en 'Cargar Datos ML' y luego en 'Entrenar Modelo ML'.")
            return

        # 1. Obtener la región de 15x15 píxeles bajo el marco
        if self.zoom_is_fixed:
            current_zoom_pos = self.fixed_zoom_position
        else:
            current_zoom_pos = self.label_imagen._frame_pos

        original_size = self.pixmap_original.size()
        label_size = self.label_imagen.size()
        
        scale_factor = min(label_size.width() / original_size.width(), label_size.height() / original_size.height())
        
        drawn_offset_x = (label_size.width() - original_size.width() * scale_factor) / 2
        drawn_offset_y = (label_size.height() - original_size.height() * scale_factor) / 2

        x_on_original = (current_zoom_pos.x() - drawn_offset_x) / scale_factor
        y_on_original = (current_zoom_pos.y() - drawn_offset_y) / scale_factor
        
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
            QMessageBox.warning(self, "Comprobación", "La región de 15x15 es inválida o está fuera de la imagen.")
            return
            
        region_15x15_pixmap = self.pixmap_original.copy(QRect(x_region_orig, y_region_orig, width_region, height_region))
        
        # 2. Preprocesar la región para la predicción
        image_size = 15
        
        # Asegurarse de que la imagen sea de 15x15 antes de convertir a array
        region_15x15_pil = region_15x15_pixmap.toImage().convertToFormat(QImage.Format_Grayscale8)
        region_15x15_pil = Image.fromqimage(region_15x15_pil).resize((image_size, image_size))
        
        image_array = np.array(region_15x15_pil)

        fft_image = fft2(image_array)
        fft_shifted = fftshift(fft_image)
        fft_magnitude = np.abs(fft_shifted)
        input_vector = fft_magnitude.flatten()
        input_vector = np.asmatrix(input_vector) # Convertir a matriz de NumPy

        # 3. Realizar la predicción
        predicted_class_num = predict_single_image(input_vector, self.trained_weights, len(self.class_names))

        if predicted_class_num is not None:
            class_name = self.class_names.get(predicted_class_num, "Clase Desconocida")
            QMessageBox.information(self, "Comprobación de Clase", f"La clase predicha para la región es: {class_name}")
        else:
            QMessageBox.critical(self, "Error de Predicción", "No se pudo realizar la predicción. Asegúrate de que el modelo esté entrenado.")


    def guardar_datos_ml(self):
        if self.X_data is None or self.y_labels is None:
            QMessageBox.warning(self, "Guardar Datos ML", "No hay datos de ML cargados para guardar.")
            return

        x_file_path = os.path.join(os.path.dirname(__file__), "entrenamiento_X.txt")
        y_file_path = os.path.join(os.path.dirname(__file__), "entrenamiento_y.txt")

        try:
            np.savetxt(x_file_path, self.X_data, fmt='%.18e', delimiter=',')
            np.savetxt(y_file_path, self.y_labels, fmt='%d', delimiter=',')

            QMessageBox.information(self, "Guardar Datos ML", 
                                    f"Datos de entrenamiento guardados exitosamente en:\n"
                                    f"{x_file_path}\n{y_file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error al Guardar Datos ML", f"No se pudieron guardar los datos: {e}")
            print(f"Error al guardar datos ML: {e}")


    def cargar_datos_ml(self):
        x_file_path = os.path.join(os.path.dirname(__file__), "entrenamiento_X.txt")
        y_file_path = os.path.join(os.path.dirname(__file__), "entrenamiento_y.txt")

        QMessageBox.information(self, "Cargando Datos ML", 
                                "Intentando cargar datos de entrenamiento desde archivos guardados.\n"
                                "Si no se encuentran o hay un error, se procesarán las imágenes desde 'Data_set'.")

        if os.path.exists(x_file_path) and os.path.exists(y_file_path):
            try:
                self.X_data = np.asmatrix(np.loadtxt(x_file_path, delimiter=','))
                self.y_labels = np.asmatrix(np.loadtxt(y_file_path, delimiter=','))
                
                # Asegurarse de que y_labels sea una matriz con la forma correcta (N, num_classes)
                # Si se cargó como 1D, y debe ser (N, 1) o (N, num_classes)
                if self.y_labels.ndim == 1:
                    # Si y_labels es 1D, asumimos que son las etiquetas originales (1 a 6)
                    # y las convertimos a one-hot encoding
                    num_classes = len(self.class_names)
                    y_one_hot = np.zeros((self.y_labels.shape[1], num_classes))
                    for idx, val in enumerate(np.array(self.y_labels).flatten()):
                        y_one_hot[idx, int(val) - 1] = 1
                    self.y_labels = np.asmatrix(y_one_hot)
                elif self.y_labels.ndim == 2 and self.y_labels.shape[0] == 1: # Si es 1xN
                     # Si es 1xN, y_labels es una fila de etiquetas numéricas, transponer y convertir a one-hot
                    num_classes = len(self.class_names)
                    y_one_hot = np.zeros((self.y_labels.shape[1], num_classes))
                    for idx, val in enumerate(np.array(self.y_labels).flatten()):
                        y_one_hot[idx, int(val) - 1] = 1
                    self.y_labels = np.asmatrix(y_one_hot)

                QMessageBox.information(self, "Datos ML Cargados", 
                                        f"Datos de entrenamiento cargados desde archivos.\n"
                                        f"Dimensión de X: {self.X_data.shape}\n"
                                        f"Dimensión de y: {self.y_labels.shape}")
                self.boton_guardar_ml.setVisible(True)
                self.boton_entrenar_ml.setVisible(True) # Mostrar el botón de entrenar
                print(f"X (primeras 5 filas):\n{self.X_data[:min(5, self.X_data.shape[0]), :min(5, self.X_data.shape[1])]}...")
                print(f"y (primeras 5 filas, one-hot):\n{self.y_labels[:min(5, self.y_labels.shape[0]), :min(6, self.y_labels.shape[1])]}")
                return
            except Exception as e:
                QMessageBox.warning(self, "Error al Cargar Datos ML", 
                                    f"Error al cargar datos desde archivos, se procederá a procesar las imágenes: {e}")
                print(f"Error al cargar datos desde archivos: {e}")
        else:
            QMessageBox.information(self, "Cargando Datos ML", 
                                    "Archivos de entrenamiento no encontrados. Procesando imágenes desde 'Data_set'...")

        image_size = 15
        vector_dim = image_size * image_size

        X_list = []
        y_list = []

        base_path = os.path.join(os.path.dirname(__file__), "Data_set")

        class_labels = {
            "C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5, "C6": 6
        }

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

                        y_list.append(label) # Guardar la etiqueta numérica para luego convertir a one-hot
                        total_images_loaded += 1

                    except Exception as e:
                        print(f"Error al procesar la imagen {image_path}: {e}")
                        continue

        if X_list:
            self.X_data = np.asmatrix(X_list)
            
            # Convertir y_list a one-hot encoding
            num_classes = len(class_labels)
            y_one_hot = np.zeros((len(y_list), num_classes))
            for idx, val in enumerate(y_list):
                y_one_hot[idx, val - 1] = 1 # Restar 1 porque los índices son 0-basados
            self.y_labels = np.asmatrix(y_one_hot)

            QMessageBox.information(self, "Datos ML Cargados", 
                                    f"Datos de entrenamiento cargados y preprocesados.\n"
                                    f"Total de imágenes cargadas: {total_images_loaded}\n"
                                    f"Dimensión de X: {self.X_data.shape}\n"
                                    f"Dimensión de y (one-hot): {self.y_labels.shape}")
            
            print(f"X (primeras 5 filas):\n{self.X_data[:min(5, self.X_data.shape[0]), :min(5, self.X_data.shape[1])]}...")
            print(f"y (primeras 5 filas, one-hot):\n{self.y_labels[:min(5, self.y_labels.shape[0]), :min(num_classes, self.y_labels.shape[1])]}")
            
            self.guardar_datos_ml() # Guardar los datos recién procesados
            self.boton_guardar_ml.setVisible(True)
            self.boton_entrenar_ml.setVisible(True) # Mostrar el botón de entrenar
        else:
            self.X_data = None
            self.y_labels = None
            QMessageBox.warning(self, "Carga de Datos ML", "No se encontraron imágenes válidas para cargar en el conjunto de datos.")
            print("No se encontraron imágenes válidas para cargar en el conjunto de datos.")
            self.boton_guardar_ml.setVisible(False)
            self.boton_entrenar_ml.setVisible(False)

    def train_ml_model(self):
        """
        Entrena el modelo de Machine Learning (Red Neuronal) con los datos cargados.
        """
        if self.X_data is None or self.y_labels is None:
            QMessageBox.warning(self, "Entrenar Modelo ML", "Primero, carga los datos de entrenamiento haciendo clic en 'Cargar Datos ML'.")
            return
        
        QMessageBox.information(self, "Entrenar Modelo ML", "Iniciando el entrenamiento del modelo ML. Esto puede tomar un tiempo...")

        # Configuración de la red
        input_layer_size = self.X_data.shape[1] # 225
        hidden_layer_size = 45 # Ajustado a 45 como en tu ejemplo
        output_layer_size = self.y_labels.shape[1] # 6 clases
        layers = [input_layer_size, hidden_layer_size, hidden_layer_size, hidden_layer_size, output_layer_size] # 3 capas ocultas
        
        epochs = 5000 # número de iteraciones (puedes ajustar)
        alpha = 0.5 # tasa de aprendizaje (puedes ajustar)
        epsilon = 0.1 # para inicializar los pesos (rango de pesos aleatorios)

        # Inicializar pesos
        self.trained_weights = init_weights(layers, epsilon)

        # Entrenar la red
        progress_dialog = QProgressDialog("Entrenando Modelo ML...", "Cancelar", 0, epochs, self)
        progress_dialog.setWindowTitle("Progreso del Entrenamiento")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setAutoClose(True)
        progress_dialog.show()

        for epoch in range(epochs):
            if progress_dialog.wasCanceled():
                QMessageBox.information(self, "Entrenamiento Cancelado", "El entrenamiento del modelo ML fue cancelado.")
                self.trained_weights = None # Resetear pesos si se cancela
                progress_dialog.close()
                return

            fit_result = fit_single_epoch(self.X_data, self.y_labels, self.trained_weights, alpha)
            self.trained_weights = fit_result['weights']
            J = fit_result['J']
            
            progress_dialog.setValue(epoch)
            progress_dialog.setLabelText(f"Epoch {epoch}/{epochs}, Costo: {J:.6f}")
            QApplication.processEvents() # Permite que la GUI se actualice

        progress_dialog.close()
        QMessageBox.information(self, "Entrenamiento Completado", "El modelo ML ha sido entrenado exitosamente.")
        print("Entrenamiento completado. Pesos guardados en self.trained_weights.")


# --- Funciones de la Red Neuronal (colocadas al final del script) ---

# Función de activación Sigmoidal
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivada de la sigmoidal
def s_prime(z):
    return np.multiply(z, 1.0-z)

# Función de activación Softmax para la capa de salida multiclase
def softmax(x):
    # Convertir a ndarray para usar np.max con keepdims
    x_array = np.asarray(x)
    e_x = np.exp(x_array - np.max(x_array, axis=0, keepdims=True))
    return e_x / np.sum(e_x, axis=0, keepdims=True)

# Inicialización de los pesos de la red
def init_weights(layers, epsilon):
    weights = []
    for i in range(len(layers)-1):
        w = np.random.rand(layers[i+1], layers[i]+1)
        w = w * 2*epsilon - epsilon
        weights.append(np.asmatrix(w)) # Usar np.asmatrix
    return weights

# Función para realizar un paso de entrenamiento (forward y backpropagation)
def fit_single_epoch(X, Y, w, alpha):
    m, n = X.shape
    num_output_classes = Y.shape[1]

    current_w_grad = [np.asmatrix(np.zeros(np.shape(w[i]))) for i in range(len(w))]
    h_total = np.zeros((m, num_output_classes))

    for i in range(m):
        a_s = [] 
        a_biased_s = []

        a_prev_biased = np.asmatrix(np.append(1, X[i].T)).T 
        a_biased_s.append(a_prev_biased)

        for j in range(len(w)):
            z = w[j] * a_prev_biased 

            if j == len(w) - 1:
                a_curr = softmax(z)
            else:
                a_curr = sigmoid(z)

            a_s.append(a_curr)

            if j < len(w) - 1:
                a_prev_biased = np.asmatrix(np.append(1, a_curr.T)).T 
                a_biased_s.append(a_prev_biased)

        h_total[i, :] = a_s[-1].T

        delta_L = a_s[-1] - Y[i].T 

        current_w_grad[-1] += delta_L * a_biased_s[-1].T 

        for j in reversed(range(len(w) - 1)): 
            delta_next = w[j+1].T[1:] * delta_L
            delta_L = np.multiply(delta_next, s_prime(a_s[j])) 
            current_w_grad[j] += delta_L * a_biased_s[j].T 

    for j in range(len(w)):
        w[j] -= alpha * (current_w_grad[j] / m)

    h_total = np.clip(h_total, 1e-10, 1 - 1e-10) 
    J = -(1.0 / m) * np.sum(np.multiply(Y, np.log(h_total)))
    
    return {'weights': w, 'J': J}

# Función para realizar una predicción en una sola imagen
def predict_single_image(input_vector, trained_weights, num_output_classes):
    if trained_weights is None:
        return None

    a = np.asmatrix(np.append(1, input_vector.T)).T 
    
    for j in range(len(trained_weights)):
        z = trained_weights[j] * a
        if j == len(trained_weights) - 1:
            a = softmax(z)
        else:
            a = sigmoid(z)
        
        if j < len(trained_weights) - 1:
            a = np.asmatrix(np.append(1, a.T)).T

    predicted_class_index = np.argmax(a) + 1 
    return predicted_class_index


if __name__ == "__main__":
    app = QApplication([])
    ventana = MiVentana()
    ventana.show()
    app.exec_()
