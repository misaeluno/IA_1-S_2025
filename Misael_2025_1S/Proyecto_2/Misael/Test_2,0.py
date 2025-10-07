from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton,
                             QFileDialog, QLabel, QScrollArea, QWidget,
                             QVBoxLayout, QHBoxLayout, QMessageBox) # Importar QMessageBox
from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap, QImage

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
        # --- FIN NUEVOS BOTONES ---
        
        # Ocultar inicialmente los botones de marco, zoom, guardar y comprobación
        self.boton_marco.setVisible(False)
        self.zoom_label.setVisible(False)
        self.boton_guardar.setVisible(False)
        self.boton_comprobacion.setVisible(False)

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


if __name__ == "__main__":
    app = QApplication([])
    ventana = MiVentana()
    ventana.show()
    app.exec_()