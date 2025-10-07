from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton,
                             QFileDialog, QLabel, QScrollArea, QWidget,
                             QVBoxLayout, QHBoxLayout, QMessageBox, QProgressDialog)
from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap, QImage

import numpy as np
from scipy.fft import fft2, fftshift
import os
from PIL import Image
import matplotlib.pyplot as plt # Importar matplotlib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # Importar para matriz de confusión

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

        self.X_data = None # This will hold all loaded FFT data before splitting
        self.y_labels = None # This will hold all loaded labels before splitting
        self.original_image_arrays_for_vis = None # New: Stores original grayscale image arrays for visualization

        self.X_train = None # New: Training features (FFT)
        self.y_train = None # New: Training labels
        self.X_test = None  # New: Test features (FFT)
        self.y_test = None  # New: Test labels
        self.original_train_for_vis = None # New: Original image arrays for training set visualization
        self.original_test_for_vis = None # New: Original image arrays for test set visualization

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
        # Lista de nombres de clase ordenados para la matriz de confusión
        self.class_names_ordered = [self.class_names[i] for i in sorted(self.class_names.keys())]
        
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
        
        self.boton_marco = QPushButton("Marco 15x15") # Defined as self.boton_marco
        self.boton_marco.clicked.connect(self.al_marco15x15)
        layout_derecha.addWidget(self.boton_marco) # Corrected: used self.boton_marco
        
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

        # New button for visualizing subset
        self.boton_visualizar_muestras = QPushButton("Visualizar Muestras")
        self.boton_visualizar_muestras.clicked.connect(self.visualizar_muestras)
        layout_derecha.addWidget(self.boton_visualizar_muestras)

        # New button for Confusion Matrix
        self.boton_matriz_confusion = QPushButton("Matriz de Confusión")
        self.boton_matriz_confusion.clicked.connect(self.mostrar_matriz_confusion)
        layout_derecha.addWidget(self.boton_matriz_confusion)
        
        self.boton_marco.setVisible(False)
        self.zoom_label.setVisible(False)
        self.boton_guardar.setVisible(False)
        self.boton_comprobacion.setVisible(False)
        self.boton_guardar_ml.setVisible(False)
        self.boton_entrenar_ml.setVisible(False) 
        self.boton_visualizar_muestras.setVisible(False) # Hide initially
        self.boton_matriz_confusion.setVisible(False) # Hide initially

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
            
        region_15x15 = self.pixmap_original.copy(QRect(x_region_orig, y_region_orig, width_region, height_region))
        
        zoom_pixmap = region_15x15.scaled(200, 100, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
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
        
        # Convertir QPixmap a QImage
        qimage = region_15x15_pixmap.toImage()
        
        # Convertir QImage a escala de grises si no lo está ya
        if qimage.format() != QImage.Format_Grayscale8:
            qimage = qimage.convertToFormat(QImage.Format_Grayscale8)
        
        # Asegurarse de que sea 15x15. La región ya debería ser de este tamaño, pero escalamos por seguridad.
        if qimage.width() != image_size or qimage.height() != image_size:
            qimage = qimage.scaled(image_size, image_size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

        # Convertir QImage a NumPy array de forma robusta (fila por fila)
        try:
            image_array_list = []
            for row in range(qimage.height()):
                # Get a pointer to the start of the current scanline (row)
                ptr = qimage.constScanLine(row)
                # Convert the scanline to a NumPy array
                # The size of the scanline is width * bytes_per_pixel. For Grayscale8, bytes_per_pixel is 1.
                # Use np.frombuffer with ptr.asstring() and the expected row size
                row_data = np.frombuffer(ptr.asstring(qimage.width()), dtype=np.uint8)
                image_array_list.append(row_data)
            
            image_array = np.array(image_array_list)
            print(f"DEBUG: image_array shape after conversion: {image_array.shape}")

        except Exception as e:
            QMessageBox.critical(self, "Error de Imagen", f"Error al convertir la imagen a array NumPy: {e}")
            print(f"ERROR: Fallo al convertir QImage a NumPy array: {e}")
            return


        fft_image = fft2(image_array)
        fft_shifted = fftshift(fft_image)
        fft_magnitude = np.abs(fft_shifted)
        input_vector = fft_magnitude.flatten()
        input_vector = np.asmatrix(input_vector) # Convertir a matriz de NumPy

        # APLICAR NORMALIZACIÓN AL VECTOR DE ENTRADA PARA PREDICCIÓN
        # Usar los valores min/max del conjunto de entrenamiento para normalizar
        # Esto es crucial para que la predicción sea consistente con el entrenamiento
        if self.X_train is not None and self.X_train.shape[0] > 0:
            min_val_train = np.min(self.X_train)
            max_val_train = np.max(self.X_train)
            if (max_val_train - min_val_train) > 1e-8:
                input_vector = (input_vector - min_val_train) / (max_val_train - min_val_train)
            else:
                input_vector = np.zeros_like(input_vector)
            print(f"DEBUG: Input vector normalized to range [{np.min(input_vector):.4f}, {np.max(input_vector):.4f}] for prediction.")
        else:
            print("WARNING: Training data not available for consistent normalization during prediction. Normalizing to own min/max.")
            min_val_single = np.min(input_vector)
            max_val_single = np.max(input_vector)
            if (max_val_single - min_val_single) > 1e-8:
                input_vector = (input_vector - min_val_single) / (max_val_single - min_val_single)
            else:
                input_vector = np.zeros_like(input_vector)


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
        # No saving original_image_arrays_for_vis to file, as it's only for visualization and can be re-generated.

        try:
            # When saving, save the original numerical labels, not one-hot encoded,
            # as loadtxt expects a single value per row for y.
            # Convert one-hot back to single numerical label
            y_labels_numerical = np.argmax(self.y_labels, axis=1) + 1 
            np.savetxt(x_file_path, self.X_data, fmt='%.18e', delimiter=',')
            np.savetxt(y_file_path, y_labels_numerical, fmt='%d', delimiter=',')

            QMessageBox.information(self, "Guardar Datos ML", 
                                    f"Datos de entrenamiento guardados exitosamente en:\n"
                                    f"{x_file_path}\n{y_file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error al Guardar Datos ML", f"No se pudieron guardar los datos: {e}")
            print(f"Error al guardar datos ML: {e}")


    def normalize_features(self):
        """
        Normalizes the X_data (FFT magnitudes) to a [0, 1] range.
        This should be called after X_data is fully loaded.
        """
        if self.X_data is None or self.X_data.shape[0] == 0:
            print("WARNING: No X_data to normalize.")
            return

        # Calculate min and max across all samples and all features
        min_val = np.min(self.X_data)
        max_val = np.max(self.X_data)

        if (max_val - min_val) > 1e-8: # Avoid division by zero for constant features
            self.X_data = (self.X_data - min_val) / (max_val - min_val)
            print(f"DEBUG: X_data normalized to range [{np.min(self.X_data):.4f}, {np.max(self.X_data):.4f}]")
        else:
            # If all values are the same, set to 0 (or a small constant if preferred)
            self.X_data = np.zeros_like(self.X_data)
            print("WARNING: X_data has constant values, set to zeros after normalization attempt.")


    def cargar_datos_ml(self):
        # Reinicializar datos al inicio para asegurar un estado limpio
        self.X_data = None
        self.y_labels = None
        self.original_image_arrays_for_vis = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        # Always process from Data_set to ensure original_image_arrays_for_vis is populated
        QMessageBox.information(self, "Cargando Datos ML", 
                                "Procesando imágenes desde 'Data_set' para cargar datos de entrenamiento.")
        print("DEBUG: Entering cargar_datos_ml (processing from Data_set)...")

        image_size = 15
        vector_dim = image_size * image_size

        X_list = [] # For FFT magnitudes
        y_list = [] # For numerical labels
        original_image_arrays_list = [] # For original grayscale image arrays (for visualization)

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

                        img_grayscale = img.convert('L') # Convert to grayscale (luminance)
                        img_resized = img_grayscale.resize((image_size, image_size))
                        
                        image_array = np.array(img_resized) # This is the 15x15 grayscale array
                        
                        if image_array.ndim != 2 or image_array.shape != (image_size, image_size):
                             print(f"Error: La imagen {image_path} no tiene el formato esperado (15x15 en escala de grises) después de la conversión. Dimensiones: {image_array.shape}")
                             continue

                        # Store original grayscale array for visualization
                        original_image_arrays_list.append(image_array)

                        # Calculate FFT magnitude for network input
                        fft_image = fft2(image_array)
                        fft_shifted = fftshift(fft_image)
                        fft_magnitude = np.abs(fft_shifted)

                        vectorized_image = fft_magnitude.flatten()
                        X_list.append(vectorized_image)

                        y_list.append(label) # Store numerical label for one-hot conversion later
                        total_images_loaded += 1

                    except Exception as e:
                        print(f"Error al procesar la imagen {image_path}: {e}")
                        continue

        if X_list:
            self.X_data = np.asmatrix(X_list)
            self.original_image_arrays_for_vis = np.array(original_image_arrays_list) # Store as NumPy array
            
            # Convert y_list to one-hot encoding
            num_classes = len(class_labels)
            y_one_hot = np.zeros((len(y_list), num_classes))
            for idx, val in enumerate(y_list):
                y_one_hot[idx, val - 1] = 1 # Subtract 1 because indices are 0-based
            self.y_labels = np.asmatrix(y_one_hot)

            QMessageBox.information(self, "Datos ML Cargados", 
                                    f"Datos de entrenamiento cargados y preprocesados.\n"
                                    f"Total de imágenes cargadas: {total_images_loaded}\n"
                                    f"Dimensión de X (FFT): {self.X_data.shape}\n"
                                    f"Dimensión de y (one-hot): {self.y_labels.shape}")
            
            print(f"DEBUG: X (primeras 5 filas, FFT):\n{self.X_data[:min(5, self.X_data.shape[0]), :min(5, self.X_data.shape[1])]}...")
            print(f"DEBUG: y (primeras 5 filas, one-hot):\n{self.y_labels[:min(5, self.y_labels.shape[0]), :min(num_classes, self.y_labels.shape[1])]}")
            
        else:
            self.X_data = None
            self.y_labels = None
            self.original_image_arrays_for_vis = None
            QMessageBox.warning(self, "Carga de Datos ML", "No se encontraron imágenes válidas para cargar en el conjunto de datos.")
            print("No se encontraron imágenes válidas para cargar en el conjunto de datos.")
        
        # After loading/processing, perform normalization, shuffling and splitting
        if self.X_data is not None and self.y_labels is not None and self.X_data.shape[0] > 0: # Ensure there's data before splitting
            self.normalize_features() # Call normalization here
            print(f"DEBUG: Estado antes de split_and_shuffle_data: X_data.shape={self.X_data.shape}, y_labels.shape={self.y_labels.shape}, original_image_arrays_for_vis is None: {self.original_image_arrays_for_vis is None}")
            self.split_and_shuffle_data()
            print(f"DEBUG: Estado después de split_and_shuffle_data: X_train.shape={self.X_train.shape if self.X_train is not None else 'None'}, y_train.shape={self.y_train.shape if self.y_train is not None else 'None'}")
            self.boton_guardar_ml.setVisible(True)
            self.boton_entrenar_ml.setVisible(True)
            self.boton_visualizar_muestras.setVisible(True) # Show visualize button
            self.boton_matriz_confusion.setVisible(True) # Show confusion matrix button
        else:
            print("DEBUG: No se pudo cargar o procesar datos ML válidos. Botones de ML ocultos.")
            self.boton_guardar_ml.setVisible(False)
            self.boton_entrenar_ml.setVisible(False)
            self.boton_visualizar_muestras.setVisible(False) # Hide visualize button
            self.boton_matriz_confusion.setVisible(False) # Hide confusion matrix button


    def split_and_shuffle_data(self):
        """
        Randomizes X_data and y_labels, then splits them into 80% training and 20% testing sets.
        Also shuffles and splits original_image_arrays_for_vis if available.
        """
        print(f"DEBUG: Entering split_and_shuffle_data. X_data is None: {self.X_data is None}, y_labels is None: {self.y_labels is None}, original_image_arrays_for_vis is None: {self.original_image_arrays_for_vis is None}")

        if self.X_data is None or self.y_labels is None: # Only these two are strictly required for ML training/testing
            QMessageBox.warning(self, "División de Datos", "No hay datos de características (X) o etiquetas (y) para aleatorizar y dividir. Carga los datos ML primero.")
            print("Error: Datos X o Y insuficientes para split_and_shuffle_data.")
            return

        m = self.X_data.shape[0] # Number of samples
        if m == 0:
            QMessageBox.warning(self, "División de Datos", "No hay muestras de datos para aleatorizar y dividir. Asegúrate de que la carpeta 'Data_set' contenga imágenes válidas.")
            print("Error: 0 muestras en X_data. No se puede dividir.")
            return
        if m < 2: # Need at least 2 samples for a meaningful split
            QMessageBox.warning(self, "División de Datos", "Se necesitan al menos 2 muestras para dividir los datos en conjuntos de entrenamiento y prueba.")
            print(f"Error: Solo {m} muestras. Se necesitan al menos 2.")
            return

        # Generate random permutation of indices
        permutation_indices = np.random.permutation(m)

        # Apply permutation to X_data and y_labels
        self.X_data = self.X_data[permutation_indices, :]
        self.y_labels = self.y_labels[permutation_indices, :]

        # Apply permutation to original_image_arrays_for_vis ONLY if it exists
        if self.original_image_arrays_for_vis is not None:
            self.original_image_arrays_for_vis = self.original_image_arrays_for_vis[permutation_indices, :, :] # For 3D array
        else:
            print("DEBUG: original_image_arrays_for_vis is None. Skipping shuffling for visualization data.")


        # Determine split point (80% for training)
        split_idx = int(m * 0.8)

        # Split data
        self.X_train = self.X_data[:split_idx, :]
        self.y_train = self.y_labels[:split_idx, :]
        if self.original_image_arrays_for_vis is not None:
            self.original_train_for_vis = self.original_image_arrays_for_vis[:split_idx, :, :]
        else:
            self.original_train_for_vis = None # Ensure it's None if original data wasn't available

        self.X_test = self.X_data[split_idx:, :]
        self.y_test = self.y_labels[split_idx:, :]
        if self.original_image_arrays_for_vis is not None:
            self.original_test_for_vis = self.original_image_arrays_for_vis[split_idx:, :, :]
        else:
            self.original_test_for_vis = None # Ensure it's None if original data wasn't available


        QMessageBox.information(self, "División de Datos", 
                                f"Datos aleatorizados y divididos:\n"
                                f"Conjunto de Entrenamiento (X_train, y_train): {self.X_train.shape}, {self.y_train.shape}\n"
                                f"Conjunto de Prueba (X_test, y_test): {self.X_test.shape}, {self.y_test.shape}")
        
        print(f"DEBUG: X_train (primeras 5 filas, FFT):\n{self.X_train[:min(5, self.X_train.shape[0]), :min(5, self.X_train.shape[1])]}...")
        print(f"DEBUG: y_train (primeras 5 filas, one-hot):\n{self.y_train[:min(5, self.y_train.shape[0]), :min(self.y_train.shape[1], self.y_train.shape[1])]}...")
        print(f"DEBUG: X_test (primeras 5 filas, FFT):\n{self.X_test[:min(5, self.X_test.shape[0]), :min(5, self.X_test.shape[1])]}...")
        print(f"DEBUG: y_test (primeros 5 filas, one-hot):\n{self.y_test[:min(5, self.y_test.shape[0]), :min(self.y_test.shape[1], self.y_test.shape[1])]}...")


    def visualizar_muestras(self):
        """
        Visualizes a subset of original grayscale images from the training data using matplotlib.
        """
        if self.original_train_for_vis is None:
            QMessageBox.warning(self, "Visualizar Muestras", "No hay datos de entrenamiento cargados para visualizar. Carga los datos ML primero.")
            return

        num_images_to_show = min(9, self.original_train_for_vis.shape[0]) # Show up to 9 images
        if num_images_to_show == 0:
            QMessageBox.information(self, "Visualizar Muestras", "No hay suficientes imágenes en el conjunto de entrenamiento para visualizar.")
            return

        fig, axes = plt.subplots(3, 3, figsize=(6, 6)) # Create a 3x3 grid of plots
        axes = axes.flatten() # Flatten the 2D array of axes for easy iteration

        image_size = 15 # Images are 15x15 pixels
        
        # Select random indices from the training set to display
        random_indices = np.random.choice(self.original_train_for_vis.shape[0], num_images_to_show, replace=False)

        for i, idx in enumerate(random_indices):
            image_2d = self.original_train_for_vis[idx, :, :] # Get the 15x15 grayscale image array

            # Get the original class label (before one-hot encoding) for display
            # Find the index of the '1' in the one-hot encoded label
            original_label_idx = np.argmax(np.array(self.y_train[idx, :]).flatten())
            original_label_num = original_label_idx + 1 # Convert back to 1-indexed class number
            class_name = self.class_names.get(original_label_num, "Desconocida")

            ax = axes[i]
            ax.imshow(image_2d, cmap='gray') # Display as grayscale image
            ax.set_title(f"Clase: {class_name}")
            ax.axis('off') # Hide axes

        # Hide any unused subplots
        for i in range(num_images_to_show, 9):
            fig.delaxes(axes[i])

        plt.tight_layout() # Adjust layout to prevent overlapping titles
        plt.show() # Display the plot


    def train_ml_model(self):
        """
        Entrena el modelo de Machine Learning (Red Neuronal) con los datos cargados.
        """
        if self.X_train is None or self.y_train is None: # Use training data
            QMessageBox.warning(self, "Entrenar Modelo ML", "Primero, carga y divide los datos de entrenamiento haciendo clic en 'Cargar Datos ML'.")
            print("Error: X_train o y_train es None. No se puede entrenar.")
            return
        
        QMessageBox.information(self, "Entrenar Modelo ML", "Iniciando el entrenamiento del modelo ML. Esto puede tomar un tiempo...")

        # Configuración de la red
        input_layer_size = self.X_train.shape[1] # 225
        hidden_layer_size = 25 # Changed to 25 units as per request
        output_layer_size = self.y_train.shape[1] # 6 clases
        layers = [input_layer_size, hidden_layer_size, output_layer_size] # One hidden layer as per request
        
        epochs = 5000 # número de iteraciones (puedes ajustar)
        alpha = 0.5 # tasa de aprendizaje (puedes ajustar)
        epsilon = 0.05 # Reducido para mitigar overflow en sigmoid

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

            fit_result = fit_single_epoch(self.X_train, self.y_train, self.trained_weights, alpha) # Use training data
            self.trained_weights = fit_result['weights']
            J = fit_result['J']
            
            progress_dialog.setValue(epoch)
            progress_dialog.setLabelText(f"Epoch {epoch}/{epochs}, Costo: {J:.6f}")
            QApplication.processEvents() # Permite que la GUI se actualice

        progress_dialog.close()
        QMessageBox.information(self, "Entrenamiento Completado", "El modelo ML ha sido entrenado exitosamente.")
        print("Entrenamiento completado. Pesos guardados en self.trained_weights.")
        
        # Evaluate on test set after training (optional, but good practice)
        if self.X_test is not None and self.y_test is not None:
            self.evaluate_model()

    def evaluate_model(self):
        """
        Evaluates the trained model on the test set.
        """
        if self.trained_weights is None:
            QMessageBox.warning(self, "Evaluación del Modelo", "El modelo no ha sido entrenado para evaluar.")
            return
        if self.X_test is None or self.y_test is None:
            QMessageBox.warning(self, "Evaluación del Modelo", "No hay datos de prueba para evaluar.")
            return

        correct_predictions = 0
        total_predictions = self.X_test.shape[0]

        for i in range(total_predictions):
            input_vector = np.asmatrix(self.X_test[i, :])
            true_label_one_hot = np.array(self.y_test[i, :]).flatten()
            true_label_num = np.argmax(true_label_one_hot) + 1

            predicted_class_num = predict_single_image(input_vector, self.trained_weights, len(self.class_names))

            if predicted_class_num == true_label_num:
                correct_predictions += 1
        
        accuracy = (correct_predictions / total_predictions) * 100
        QMessageBox.information(self, "Evaluación del Modelo", 
                                f"Evaluación del conjunto de prueba:\n"
                                f"Precisión: {accuracy:.2f}% ({correct_predictions}/{total_predictions} correctas)")
        print(f"Evaluación del conjunto de prueba: Precisión: {accuracy:.2f}%")

    def mostrar_matriz_confusion(self):
        """
        Calcula y muestra la matriz de confusión del modelo en el conjunto de prueba.
        """
        if self.trained_weights is None:
            QMessageBox.warning(self, "Matriz de Confusión", "El modelo ML no ha sido entrenado. Por favor, entrena el modelo primero.")
            return
        if self.X_test is None or self.y_test is None:
            QMessageBox.warning(self, "Matriz de Confusión", "No hay datos de prueba disponibles para generar la matriz de confusión. Carga los datos ML primero.")
            return

        print(f"DEBUG: X_test shape at start of mostrar_matriz_confusion: {self.X_test.shape}")
        print(f"DEBUG: y_test shape at start of mostrar_matriz_confusion: {self.y_test.shape}")

        # 1. Obtener las predicciones del modelo para el conjunto de prueba
        y_pred_nums = []
        for i in range(self.X_test.shape[0]):
            input_vector = np.asmatrix(self.X_test[i, :])
            predicted_class_num = predict_single_image(input_vector, self.trained_weights, len(self.class_names))
            
            # Debug print for each prediction
            print(f"DEBUG: Prediction {i}: {predicted_class_num}") 
            
            # Handle cases where predicted_class_num might be None or invalid
            if predicted_class_num is not None:
                y_pred_nums.append(predicted_class_num)
            else:
                # If a prediction is None, it indicates an issue in the neural network's forward pass.
                # For confusion matrix, we need a valid prediction. Assign a placeholder or skip.
                # For debugging the shape issue, we must ensure y_pred_nums has the same length as y_true_nums.
                # Let's append a placeholder for now to keep the length consistent.
                print(f"WARNING: Predicted class for sample {i} is None. Appending placeholder (1).")
                y_pred_nums.append(1) # Appending a default valid class (e.g., class 1) to maintain length

        y_pred_nums = np.array(y_pred_nums)

        # 2. Obtener las etiquetas verdaderas del conjunto de prueba (convertir de one-hot a numérico)
        # Convert self.y_test to a numpy array first to ensure argmax returns a 1D array
        y_true_nums = np.array(self.y_test).argmax(axis=1).flatten() + 1 # Convertir de 0-indexed a 1-indexed

        print(f"DEBUG: y_true_nums shape before confusion_matrix: {y_true_nums.shape}")
        print(f"DEBUG: y_pred_nums shape before confusion_matrix: {y_pred_nums.shape}")

        # 3. Calcular la matriz de confusión
        # Ensure labels are integers for confusion_matrix
        cm = confusion_matrix(y_true_nums.astype(int), y_pred_nums.astype(int), labels=sorted(self.class_names.keys()))

        # 4. Visualizar la matriz de confusión
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names_ordered)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        ax.set_title("Matriz de Confusión")
        plt.xticks(rotation=45, ha='right') # Rotar etiquetas para mejor lectura
        plt.tight_layout()
        plt.show()


# --- Funciones de la Red Neuronal (colocadas al final del script) ---

# Función de activación Sigmoidal
def sigmoid(x):
    # Clip input to prevent overflow/underflow in exp for numerical stability
    x_clipped = np.clip(x, -10, 10) # Tighter clipping range
    return 1/(1+np.exp(-x_clipped))

# Derivada de la sigmoidal
def s_prime(z):
    return np.multiply(z, 1.0-z)

# Función de activación Softmax para la capa de salida multiclase
def softmax(x):
    # Convertir a ndarray para usar np.max con keepdims
    x_array = np.asarray(x)
    # Subtract max for numerical stability (already implemented)
    e_x = np.exp(x_array - np.max(x_array, axis=0, keepdims=True))
    # Evitar división por cero si la suma es 0 (aunque softmax debería evitarlo para valores reales)
    sum_e_x = np.sum(e_x, axis=0, keepdims=True)
    return e_x / (sum_e_x + 1e-10) # Añadir un pequeño valor para estabilidad

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
    # h_total is not strictly needed inside fit_single_epoch if only cost is returned.
    # But keeping it for consistency if needed for debugging/inspection.
    h_total = np.zeros((m, num_output_classes)) 

    for i in range(m):
        a_s = [] # Stores activations *after* activation function (without bias for next layer)
        a_biased_s = [] # Stores activations *with* bias (inputs to weight matrices)

        # Input layer (Layer 0)
        # X[i] is (1, n) -> X[i].T is (n, 1)
        # np.append(1, X[i].T) makes it (n+1, 1) then .T makes it (1, n+1)
        # .T again makes it (n+1, 1)
        a_prev_biased = np.asmatrix(np.append(1, X[i].T)).T 
        a_biased_s.append(a_prev_biased)

        # Forward propagation
        for j in range(len(w)):
            z = w[j] * a_prev_biased 

            if j == len(w) - 1: # Output layer
                a_curr = softmax(z)
            else: # Hidden layers
                a_curr = sigmoid(z)

            a_s.append(a_curr)

            if j < len(w) - 1: # If not the last layer, add bias for next layer's input
                a_prev_biased = np.asmatrix(np.append(1, a_curr.T)).T 
                a_biased_s.append(a_prev_biased)
            # No need for else here, as a_prev_biased is only needed if there's a next layer.

        h_total[i, :] = a_s[-1].T # Store final activation for cost calculation

        # Backpropagation
        delta_L = a_s[-1] - Y[i].T 

        # Gradient for the last layer's weights (output layer)
        current_w_grad[-1] += delta_L * a_biased_s[-1].T 

        # Backpropagate through hidden layers
        for j in reversed(range(len(w) - 1)): 
            # delta for previous layer
            # w[j+1].T[1:] excludes bias row from next layer's weights
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

    a = np.asmatrix(np.append(1, input_vector.T)).T # Input to first layer with bias
    
    for j in range(len(trained_weights)):
        z = trained_weights[j] * a
        if j == len(trained_weights) - 1: # Output layer
            a = softmax(z)
        else: # Hidden layers
            a = sigmoid(z)
        
        if j < len(trained_weights) - 1: # If not the last layer, add bias for next layer's input
            a = np.asmatrix(np.append(1, a.T)).T

    predicted_class_index = np.argmax(a) + 1 
    return predicted_class_index


if __name__ == "__main__":
    app = QApplication([])
    ventana = MiVentana()
    ventana.show()
    app.exec_()
