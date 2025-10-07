from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton,
                             QFileDialog, QLabel, QScrollArea, QWidget,
                             QVBoxLayout, QHBoxLayout, QMessageBox, QProgressDialog,
                             QCheckBox, QDoubleSpinBox) # Importar QCheckBox y QDoubleSpinBox
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

        self.X_data = None # Esto mantendrá todos los datos FFT cargados antes de dividirlos
        self.y_labels = None # Esto mantendrá todas las etiquetas cargadas antes de dividirlas
        self.original_image_arrays_for_vis = None # Nuevo: Almacena matrices de imágenes en escala de grises originales para su visualización

        self.X_train = None # Nuevo: Funciones de entrenamiento (FFT)
        self.y_train = None # Nuevo: Etiquetas de entrenamiento
        self.X_test = None  # Nuevo: Funciones de prueba (FFT)
        self.y_test = None  # Nuevo: Etiquetas de prueba
        self.original_train_for_vis = None # Nuevo: Matrices de imágenes originales para la visualización del conjunto de entrenamiento
        self.original_test_for_vis = None # Nuevo: Matrices de imágenes originales para la visualización del conjunto de pruebas

        self.trained_weights = None # Para almacenar los pesos entrenados de la red neuronal

        # Mapeo de etiquetas numéricas a nombres de clase
        self.class_names = {
            1: "agua ",
            2: "vegetación ",
            3: "montañas ",
            4: "desiertos ",
            5: "ríos ",
            6: "ciudades "
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
        
        self.boton_marco = QPushButton("Marco 15x15")# Definido como self.boton_marco
        self.boton_marco.clicked.connect(self.al_marco15x15)
        layout_derecha.addWidget(self.boton_marco) # Corregido: se usó self.boton_marco
        
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

        self.boton_guardar_ml = QPushButton("Guardar Datos ML")     #supuestamente guardo los datos ML en 
        self.boton_guardar_ml.clicked.connect(self.guardar_datos_ml) # entrenamiento_X.txt y entranamiento_y.txt
        layout_derecha.addWidget(self.boton_guardar_ml) # pero cargar datos ml ya lo hace ( CR, no recuerdo )

        # Opciones de Regularización
        self.checkbox_regularizacion = QCheckBox("Habilitar Regularización")
        self.checkbox_regularizacion.setChecked(False) # Por defecto deshabilitada
        layout_derecha.addWidget(self.checkbox_regularizacion)

        lambda_layout = QHBoxLayout()
        lambda_layout.addWidget(QLabel("Valor de Lambda :"))
        self.spinbox_lambda = QDoubleSpinBox()
        self.spinbox_lambda.setRange(0.0, 10.0)
        self.spinbox_lambda.setSingleStep(0.01)
        self.spinbox_lambda.setValue(0.1) # Valor por defecto de lambda
        lambda_layout.addWidget(self.spinbox_lambda)
        layout_derecha.addLayout(lambda_layout)

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
        self.boton_visualizar_muestras.setVisible(False) # Ocultar inicialmente
        self.boton_matriz_confusion.setVisible(False) # Ocultar inicialmente

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

        # Convertir QImage a una matriz NumPy de forma robusta (fila por fila)
        try:
            image_array_list = []
            for row in range(qimage.height()):
                # Obtener un puntero al inicio de la línea de escaneo actual (fila)
                ptr = qimage.constScanLine(row)
                # Convertir la línea de escaneo en una matriz NumPy
                # El tamaño de la línea de escaneo es ancho * bytes por píxel. Para Grayscale8, bytes por píxel es 1.
                # Usar np.frombuffer con ptr.asstring() y el tamaño de fila esperado.
                row_data = np.frombuffer(ptr.asstring(qimage.width()), dtype=np.uint8)
                image_array_list.append(row_data)
            
            image_array = np.array(image_array_list)
            print(f"DEBUG: forma de la matriz de imágenes después de la conversión: {image_array.shape}")

        except Exception as e:
            QMessageBox.critical(self, "Error de Imagen", f"Error al convertir la imagen a array NumPy: {e}")
            print(f"ERROR: Fallo al convertir QImage a una matriz NumPy: {e}")
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
            print(f"DEBUG: Vector de entrada normalizado al rango [{np.min(input_vector):.4f}, {np.max(input_vector):.4f}] para la predicción.")
        else:
            print("ADVERTENCIA: Datos de entrenamiento no disponibles para una normalización consistente durante la predicción. Normalización según sus propios valores mínimos y máximos..")
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
        # No se guarda original_image_arrays_for_vis en el archivo, ya que es solo para visualización y se puede volver a generar.

        try:
            # Al guardar, guarde las etiquetas numéricas originales, no las codificadas one-hot, 
            # ya que loadtxt espera un solo valor por fila para y.
            # Convierta one-hot de nuevo a una etiqueta numérica única
            # Es asi o genera error de capacidad de vector no especificada correctamente
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
        Normaliza los datos X (magnitudes de FFT) en un rango [0, 1].
        Debe ejecutarse después de que los datos X se hayan cargado por completo.
        """
        if self.X_data is None or self.X_data.shape[0] == 0:
            print("ADVERTENCIA: No hay X_data para normalizar.")
            return

        # Calcular el mínimo y el máximo en todas las muestras y todas las características
        min_val = np.min(self.X_data)
        max_val = np.max(self.X_data)

        if (max_val - min_val) > 1e-8: # Evite la división por cero para características constantes
            self.X_data = (self.X_data - min_val) / (max_val - min_val)
            print(f"DEBUG: X_data normalizado al rango [{np.min(self.X_data):.4f}, {np.max(self.X_data):.4f}]")
        else:
            # Si todos los valores son iguales, configúrelo en 0
            self.X_data = np.zeros_like(self.X_data)
            print("WARNING: X_data tiene valores constantes, establecidos en cero después del intento de normalización.")


    def cargar_datos_ml(self):
        # Reinicializar datos al inicio para asegurar un estado limpio
        self.X_data = None
        self.y_labels = None
        self.original_image_arrays_for_vis = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        # Siempre procese desde Data_set para garantizar que original_image_arrays_for_vis esté completo
        QMessageBox.information(self, "Cargando Datos ML", 
                                "Procesando imágenes desde 'Data_set' para cargar datos de entrenamiento.")
        print("DEBUG: Ingresando cargar_datos_ml (procesando desde Data_set)...")

        image_size = 15
        vector_dim = image_size * image_size

        X_list = [] # Para magnitudes FFT
        y_list = [] # Para etiquetas numéricas
        original_image_arrays_list = [] # Para matrices de imágenes en escala de grises originales (para visualización)

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

                        img_grayscale = img.convert('L') # Convertir a escala de grises (luminancia)
                        img_resized = img_grayscale.resize((image_size, image_size))
                        
                        image_array = np.array(img_resized) # Esta es la matriz de escala de grises de 15x15
                        
                        if image_array.ndim != 2 or image_array.shape != (image_size, image_size):
                             print(f"Error: La imagen {image_path} no tiene el formato esperado (15x15 en escala de grises) después de la conversión. Dimensiones: {image_array.shape}")
                             continue

                        # Almacenar la matriz de escala de grises original para visualización
                        original_image_arrays_list.append(image_array)

                        # Calcular la magnitud de FFT para la entrada de red
                        fft_image = fft2(image_array)
                        fft_shifted = fftshift(fft_image)
                        fft_magnitude = np.abs(fft_shifted)

                        vectorized_image = fft_magnitude.flatten()
                        X_list.append(vectorized_image)

                        y_list.append(label) # Almacenar etiqueta numérica para conversión one-hot posterior
                        total_images_loaded += 1

                    except Exception as e:
                        print(f"Error al procesar la imagen {image_path}: {e}")
                        continue

        if X_list:
            self.X_data = np.asmatrix(X_list)
            self.original_image_arrays_for_vis = np.array(original_image_arrays_list) # Almacenar como matriz NumPy
            
            # Convertir y_list a codificación one-hot
            num_classes = len(class_labels)
            y_one_hot = np.zeros((len(y_list), num_classes))
            for idx, val in enumerate(y_list):
                y_one_hot[idx, val - 1] = 1 # Resta 1 porque los índices están basados en 0
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
        
        # Después de cargar/procesar, realice la normalización, la mezcla y la división
        if self.X_data is not None and self.y_labels is not None and self.X_data.shape[0] > 0: # Asegúrese de que haya datos antes de dividir
            self.normalize_features() # Normalización de llamadas aquí
            print(f"DEBUG: Estado antes de split_and_shuffle_data: X_data.shape={self.X_data.shape}, y_labels.shape={self.y_labels.shape}, original_image_arrays_for_vis is None: {self.original_image_arrays_for_vis is None}")
            self.split_and_shuffle_data()
            print(f"DEBUG: Estado después de split_and_shuffle_data: X_train.shape={self.X_train.shape if self.X_train is not None else 'None'}, y_train.shape={self.y_train.shape if self.y_train is not None else 'None'}")
            self.boton_guardar_ml.setVisible(True)
            self.boton_entrenar_ml.setVisible(True)
            self.boton_visualizar_muestras.setVisible(True) # Mostrar botón de visualización
            self.boton_matriz_confusion.setVisible(True) # Mostrar el botón de matriz de confusión
        else:
            print("DEBUG: No se pudo cargar o procesar datos ML válidos. Botones de ML ocultos.")
            self.boton_guardar_ml.setVisible(False)
            self.boton_entrenar_ml.setVisible(False)
            self.boton_visualizar_muestras.setVisible(False) # Ocultar el botón de visualización
            self.boton_matriz_confusion.setVisible(False) # Botón para ocultar la matriz de confusión

    # PARTE DE ENTRENAMIENTO -------------
    def split_and_shuffle_data(self):
        """
        Aleatoriza los datos X y las etiquetas Y, y luego los divide en un 80 % de conjuntos de entrenamiento y un 20 % de conjuntos de prueba.
        También baraja y divide los conjuntos de imágenes originales para Vis, si están disponibles.
        """
        print(f"DEPURACIÓN: Se está introduciendo split_and_shuffle_data. X_data es Ninguno: {self.X_data is None}, y_labels is None: {self.y_labels is None}, original_image_arrays_for_vis is None: {self.original_image_arrays_for_vis is None}")

        if self.X_data is None or self.y_labels is None: # Solo estos dos son estrictamente necesarios para el entrenamiento/prueba de ML
            QMessageBox.warning(self, "División de Datos", "No hay datos de características (X) o etiquetas (y) para aleatorizar y dividir. Carga los datos ML primero.")
            print("Error: Datos X o Y insuficientes para split_and_shuffle_data.")
            return

        m = self.X_data.shape[0] # Número de muestras
        if m == 0:
            QMessageBox.warning(self, "División de Datos", "No hay muestras de datos para aleatorizar y dividir. Asegúrate de que la carpeta 'Data_set' contenga imágenes válidas.")
            print("Error: 0 muestras en X_data. No se puede dividir.")
            return
        if m < 2: # Se necesitan al menos 2 muestras para una división significativa
            QMessageBox.warning(self, "División de Datos", "Se necesitan al menos 2 muestras para dividir los datos en conjuntos de entrenamiento y prueba.")
            print(f"Error: Solo {m} muestras. Se necesitan al menos 2.")
            return

        # Generar permutación aleatoria de índices
        permutation_indices = np.random.permutation(m)

        # Aplicar permutación a X_data y y_labels
        self.X_data = self.X_data[permutation_indices, :]
        self.y_labels = self.y_labels[permutation_indices, :]

        # Aplicar permutación a original_image_arrays_for_vis SÓLO si existe
        if self.original_image_arrays_for_vis is not None:
            self.original_image_arrays_for_vis = self.original_image_arrays_for_vis[permutation_indices, :, :] # Para matriz 3D
        else:
            print("DEBUG: original_image_arrays_for_vis es Ninguno. Se omite la reorganización de los datos de visualización.")


        # Determinar el punto de división (80% para el entrenamiento)
        split_idx = int(m * 0.8)

        # Dividir datos
        self.X_train = self.X_data[:split_idx, :]
        self.y_train = self.y_labels[:split_idx, :]
        if self.original_image_arrays_for_vis is not None:
            self.original_train_for_vis = self.original_image_arrays_for_vis[:split_idx, :, :]
        else:
            self.original_train_for_vis = None # Asegúrese de que sea Ninguno si los datos originales no estaban disponibles

        self.X_test = self.X_data[split_idx:, :]
        self.y_test = self.y_labels[split_idx:, :]
        if self.original_image_arrays_for_vis is not None:
            self.original_test_for_vis = self.original_image_arrays_for_vis[split_idx:, :, :]
        else:
            self.original_test_for_vis = None # Asegúrese de que sea Ninguno si los datos originales no estaban disponibles


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
        Visualiza un subconjunto de imágenes en escala de grises originales a partir de los datos de entrenamiento utilizando matplotlib.
        """
        if self.original_train_for_vis is None:
            QMessageBox.warning(self, "Visualizar Muestras", "No hay datos de entrenamiento cargados para visualizar. Carga los datos ML primero.")
            return

        num_images_to_show = min(9, self.original_train_for_vis.shape[0]) # Mostrar hasta 9 imágenes
        if num_images_to_show == 0:
            QMessageBox.information(self, "Visualizar Muestras", "No hay suficientes imágenes en el conjunto de entrenamiento para visualizar.")
            return

        fig, axes = plt.subplots(3, 3, figsize=(6, 6)) # Crea una cuadrícula de gráficos de 3x3
        axes = axes.flatten() # Aplanar la matriz 2D de ejes para facilitar la iteración

        image_size = 15 # Las imágenes son de 15x15 píxeles
        
        # Seleccione índices aleatorios del conjunto de entrenamiento para mostrar
        random_indices = np.random.choice(self.original_train_for_vis.shape[0], num_images_to_show, replace=False)

        for i, idx in enumerate(random_indices):
            image_2d = self.original_train_for_vis[idx, :, :] # Obtener la matriz de imágenes en escala de grises de 15x15

            # Obtener la etiqueta de clase original (antes de la codificación one-hot) para su visualización
            # Encontrar el índice del '1' en la etiqueta codificada one-hot
            original_label_idx = np.argmax(np.array(self.y_train[idx, :]).flatten())
            original_label_num = original_label_idx + 1 # Convertir de nuevo a número de clase indexado en 1
            class_name = self.class_names.get(original_label_num, "Desconocida")

            ax = axes[i]
            ax.imshow(image_2d, cmap='gray') # Mostrar como imagen en escala de grises
            ax.set_title(f"Clase: {class_name}")
            ax.axis('off') # Ocultar ejes

        # Ocultar cualquier subtrama no utilizada
        for i in range(num_images_to_show, 9):
            fig.delaxes(axes[i])

        plt.tight_layout() # Ajustar el diseño para evitar que los títulos se superpongan
        plt.show() # Mostrar la gráfica


    def train_ml_model(self):
        """
        Entrena el modelo de Machine Learning (Red Neuronal) con los datos cargados.
        """
        if self.X_train is None or self.y_train is None:    # Utilizar datos de entrenamiento
            QMessageBox.warning(self, "Entrenar Modelo ML", "Primero, carga y divide los datos de entrenamiento haciendo clic en 'Cargar Datos ML'.")
            print("Error: X_train o y_train es NULL o esta vacio. No se puede entrenar.")
            return
        
        QMessageBox.information(self, "Entrenar Modelo ML", "Iniciando el entrenamiento del modelo ML. Esto puede tomar un tiempo...")

#--RED NEURONAL ------------------------------------------------------------------
#---------------------------------------------------------------------------------

        # Configuración de la red
        input_layer_size = self.X_train.shape[1] # 225
        hidden_layer_size = 25 # Cambiado a 25 unidades según solicitud
        output_layer_size = self.y_train.shape[1] # 6 clases
        layers = [input_layer_size, hidden_layer_size, output_layer_size] #Una capa oculta según solicitud
        
        epochs = 5000 # número de iteraciones (puedes ajustar)
        alpha = 0.05 # tasa de aprendizaje (puedes ajustar)
        epsilon = 0.05 # Reducido para mitigar overflow en sigmoid

        # Obtener valores de regularización de la interfaz
        use_regularization = self.checkbox_regularizacion.isChecked()
        lambda_reg = self.spinbox_lambda.value() if use_regularization else 0.0

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

            # Pasar lambda_reg a la función de entrenamiento
            fit_result = fit_single_epoch(self.X_train, self.y_train, self.trained_weights, alpha, lambda_reg) # Utilizar datos de entrenamiento
            self.trained_weights = fit_result['weights']
            J = fit_result['J']
            
            progress_dialog.setValue(epoch)
            progress_dialog.setLabelText(f"Epoch {epoch}/{epochs}, Costo: {J:.6f}")
            QApplication.processEvents() # Permite que la GUI se actualice

        progress_dialog.close()
        QMessageBox.information(self, "Entrenamiento Completado", "El modelo ML ha sido entrenado exitosamente.")
        print("Entrenamiento completado. Pesos guardados en self.trained_weights.")
        
        # Evaluar en el conjunto de pruebas después del entrenamiento (opcional, pero es una buena práctica)
        if self.X_test is not None and self.y_test is not None:
            self.evaluate_model()

    def evaluate_model(self):
        """
        Evalúa el modelo entrenado en el conjunto de prueba.
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

        print(f"DEBUG: Forma de prueba X al inicio de mostrar_matriz_confusion: {self.X_test.shape}")
        print(f"DEBUG: forma de y_test al inicio de mostrar_matriz_confusion: {self.y_test.shape}")

        # 1. Obtener las predicciones del modelo para el conjunto de prueba
        y_pred_nums = []
        for i in range(self.X_test.shape[0]):
            input_vector = np.asmatrix(self.X_test[i, :])
            predicted_class_num = predict_single_image(input_vector, self.trained_weights, len(self.class_names))
            
            # Impresión de depuración para cada predicción
            print(f"DEBUG: Predicción {i}: {predicted_class_num}") 
            
            # Manejar casos donde predicted_class_num podría ser Ninguno o no válido
            if predicted_class_num is not None:
                y_pred_nums.append(predicted_class_num)
            else:
                # Si una predicción es NULL, indica un problema en el paso hacia adelante de la red neuronal.
                # Para la matriz de confusión, necesitamos una predicción válida. Asignar un marcador de posición o omitir.
                # Para depurar el problema de forma, debemos asegurarnos de que y_pred_nums tenga la misma longitud que y_true_nums.
                # Añadamos un marcador de posición por ahora para mantener la longitud constante.
                print(f"ADVERTENCIA: La clase prevista para la muestra {i} es NILL. Se añade el marcador de posición (1).")
                y_pred_nums.append(1) # Appending a default valid class (e.g., class 1) to maintain length

        y_pred_nums = np.array(y_pred_nums)

        # 2. Obtener las etiquetas verdaderas del conjunto de prueba (convertir de one-hot a numérico)
        # Convert self.y_test to a numpy array first to ensure argmax returns a 1D array
        y_true_nums = np.array(self.y_test).argmax(axis=1).flatten() + 1 # Convertir de 0-indexed a 1-indexed

        print(f"DEBUG: y_true_nums forma antes de confusion_matrix: {y_true_nums.shape}")
        print(f"DEBUG: y_pred_nums forma antes de confusion_matrix: {y_pred_nums.shape}")

        # 3. Calcular la matriz de confusión
        # Asegúrese de que las etiquetas sean números enteros para confusion_matrix
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
    # Recorte la entrada para evitar desbordamiento/desbordamiento insuficiente en exp para estabilidad numérica
    x_clipped = np.clip(x, -10, 10) # Rango de recorte más estrecho
    return 1/(1+np.exp(-x_clipped))

# Derivada de la sigmoidal
def s_prime(z):
    return np.multiply(z, 1.0-z)

# Función de activación Softmax para la capa de salida multiclase
def softmax(x):
    x_array = np.asarray(x) # Convertir a ndarray para usar np.max con keepdims
    e_x = np.exp(x_array - np.max(x_array, axis=0, keepdims=True))# Restar el máximo para la estabilidad numérica (ya implementado)
    sum_e_x = np.sum(e_x, axis=0, keepdims=True) # Evitar división por cero si la suma es 0 (aunque softmax debería evitarlo para valores reales)
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
def fit_single_epoch(X, Y, w, alpha, lambda_reg=0.0): # Añadir lambda_reg como parámetro
    m, n = X.shape
    num_output_classes = Y.shape[1]

    current_w_grad = [np.asmatrix(np.zeros(np.shape(w[i]))) for i in range(len(w))]
    h_total = np.zeros((m, num_output_classes)) 

    for i in range(m):
        a_s = [] # Almacena activaciones *después* de la función de activación (sin sesgo para la siguiente capa)
        a_biased_s = [] # Almacena activaciones *con* sesgo (entradas a matrices de ponderación)

        # Capa de entrada (Capa 0)
        a_prev_biased = np.asmatrix(np.append(1, X[i].T)).T 
        a_biased_s.append(a_prev_biased)

       # Propagación hacia adelante
        for j in range(len(w)):
            z = w[j] * a_prev_biased 

            if j == len(w) - 1:# Capa de salida
                a_curr = softmax(z)
            else: # Capas ocultas
                a_curr = sigmoid(z)

            a_s.append(a_curr)

            if j < len(w) - 1: # Si no es la última capa, agregue sesgo para la entrada de la siguiente capa
                a_prev_biased = np.asmatrix(np.append(1, a_curr.T)).T 
                a_biased_s.append(a_prev_biased)

        h_total[i, :] = a_s[-1].T # Activación final de la tienda para el cálculo de costos
        # Retropropagación
        delta_L = a_s[-1] - Y[i].T 

        #Degradado para los pesos de la última capa (capa de salida)
        current_w_grad[-1] += delta_L * a_biased_s[-1].T 

        # Retropropagación a través de capas ocultas
        for j in reversed(range(len(w) - 1)): 
            delta_next = w[j+1].T[1:] * delta_L
            delta_L = np.multiply(delta_next, s_prime(a_s[j])) 
            current_w_grad[j] += delta_L * a_biased_s[j].T 

    # Calcular el costo sin regularización
    h_total = np.clip(h_total, 1e-10, 1 - 1e-10) 
    J = -(1.0 / m) * np.sum(np.multiply(Y, np.log(h_total)) + np.multiply((1 - Y), np.log(1 - h_total)))

    # Añadir término de regularización al costo J
    if lambda_reg > 0:
        reg_term_cost = 0
        for l in range(len(w)): # Iterar a través de todas las matrices de pesos
            # Sumar los cuadrados de todos los pesos excluyendo la columna de bias (primera columna)
            reg_term_cost += np.sum(np.square(w[l][:, 1:]))
        J += (lambda_reg / (2 * m)) * reg_term_cost

    # Aplicar regularización a los gradientes
    if lambda_reg > 0:
        for j in range(len(w)):
            # Crear un término de gradiente de regularización
            reg_gradient_term = (lambda_reg / m) * w[j]
            reg_gradient_term[:, 0] = 0 # Establecer la regularización del bias a cero
            current_w_grad[j] += reg_gradient_term

    # Actualizar pesos
    for j in range(len(w)):
        w[j] -= alpha * (current_w_grad[j] / m)
    
    return {'weights': w, 'J': J}

# Función para realizar una predicción en una sola imagen
def predict_single_image(input_vector, trained_weights, num_output_classes):
    if trained_weights is None:
        return None

    a = np.asmatrix(np.append(1, input_vector.T)).T # Entrada a la primera capa con sesgo
    
    for j in range(len(trained_weights)):
        z = trained_weights[j] * a
        if j == len(trained_weights) - 1: # Capa de salida
            a = softmax(z)
        else: # Capas ocultas
            a = sigmoid(z)
        
        if j < len(trained_weights) - 1: # Si no es la última capa, agregue sesgo para la entrada de la siguiente capa
            a = np.asmatrix(np.append(1, a.T)).T

    predicted_class_index = np.argmax(a) + 1 
    return predicted_class_index


if __name__ == "__main__":
    app = QApplication([])
    ventana = MiVentana()
    ventana.show()
    app.exec_()
