from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton,
                             QFileDialog, QLabel, QScrollArea, QWidget,
                             QVBoxLayout, QHBoxLayout)
from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap, QImage

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
            
            # Ajustar x e y para que el marco no se salga de los límites del pixmap
            # Esto es más complejo si el pixmap es más pequeño que el label y está centrado.
            # Para simplificar, asumimos que estamos mapeando a coordenadas del pixmap.
            
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
        self.setWindowTitle("Hola, PyQt!")
        self.setGeometry(100, 100, 640, 480)

        # Variables para el marco y zoom
        self.marco_activo = False
        self.imagen_cargada = False
        self.zoom_label = QLabel()  # Label para mostrar el zoom

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
        boton_marco = QPushButton("marco")
        boton_marco.clicked.connect(self.al_marco15x15)
        layout_derecha.addWidget(boton_marco)

        # Label para el zoom (200x100)
        self.zoom_label.setFixedSize(200, 100)
        self.zoom_label.setStyleSheet("border: 1px solid black;")
        layout_derecha.addWidget(self.zoom_label)

        # Botón "Salir"
        boton_salir = QPushButton("salir")
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
            self.label_imagen.setPixmap(self.pixmap_original)
            self.imagen_cargada = True

    def al_marco15x15(self):
        self.marco_activo = not self.marco_activo
        self.label_imagen.set_show_frame(self.marco_activo) # Indicar al label si mostrar el marco
        print("Marco 15x15:", "Activado" if self.marco_activo else "Desactivado")

    def al_salir(self):
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

        # Calcular las coordenadas del centro del marco en la imagen *original*
        # Esto es crucial si la imagen está escalada dentro del QLabel
        
        # Obtener el pixmap actual del label para saber su tamaño real dibujado
        current_pixmap = self.label_imagen.pixmap()
        if current_pixmap.isNull():
            return
            
        # Calcular el factor de escala y el desplazamiento si la imagen se ajustó al label
        # Asumimos que la imagen se muestra centrada y proporcional si es más pequeña que el label
        # o que el label muestra una porción si es más grande (debido al QScrollArea)

        # Posición del ratón en las coordenadas del *pixmap original*
        # Esto requiere mapear la posición del ratón en el label_imagen al pixmap
        # asumiendo que el pixmap se muestra centrado o escalado.

        # Tamaño real del pixmap mostrado en el QLabel
        pixmap_w = current_pixmap.width()
        pixmap_h = current_pixmap.height()

        # Desplazamiento del pixmap dentro del QLabel si no llena el QLabel
        x_offset = (self.label_imagen.width() - pixmap_w) / 2 if self.label_imagen.width() > pixmap_w else 0
        y_offset = (self.label_imagen.height() - pixmap_h) / 2 if self.label_imagen.height() > pixmap_h else 0
        
        # Posición del ratón relativa al pixmap dibujado
        x_in_pixmap = mouse_pos_in_label.x() - x_offset
        y_in_pixmap = mouse_pos_in_label.y() - y_offset

        # Asegurarse de que las coordenadas estén dentro de los límites del pixmap original
        # y que la región 15x15 no se salga
        half_frame = self.label_imagen._frame_half_size
        
        # Coordenadas de inicio de la región 15x15 en el pixmap original
        # Asegurarse de que estas coordenadas estén dentro de los límites del pixmap
        src_x = max(0, int(x_in_pixmap - half_frame))
        src_y = max(0, int(y_in_pixmap - half_frame))

        # Ajustar el ancho y alto de la región si se acerca a los bordes
        src_w = self.label_imagen._frame_size
        src_h = self.label_imagen._frame_size

        # Asegurar que el final de la región no exceda el tamaño del pixmap
        if src_x + src_w > pixmap_w:
            src_w = pixmap_w - src_x
        if src_y + src_h > pixmap_h:
            src_h = pixmap_h - src_y
            
        # Si la región resultante es inválida (por ejemplo, 0 ancho/alto), no hacer nada
        if src_w <= 0 or src_h <= 0:
            self.zoom_label.clear() # Limpiar el zoom si no hay región válida
            return


        # region = self.label_imagen.pixmap().copy(QRect(src_x, src_y, src_w, src_h))
        # Para copiar del pixmap original y no del escalado en el label, es mejor
        # usar self.pixmap_original. Asumimos que self.pixmap_original es la fuente.
        # Necesitas el factor de escala para mapear de las coordenadas del label a las del pixmap original.

        if self.pixmap_original.isNull():
            return

        # Calcular el factor de escala real entre el pixmap original y el que se muestra en el label
        # (Esto es complejo porque QLabel escala la imagen para ajustarse, manteniendo proporciones)
        original_size = self.pixmap_original.size()
        label_size = self.label_imagen.size()
        
        # Calcular el factor de escala manteniendo el aspecto ratio
        scale_factor = min(label_size.width() / original_size.width(), label_size.height() / original_size.height())
        
        # Calcular el tamaño del pixmap dibujado en el QLabel
        drawn_pixmap_w = original_size.width() * scale_factor
        drawn_pixmap_h = original_size.height() * scale_factor
        
        # Calcular el desplazamiento del pixmap dentro del QLabel (centrado)
        drawn_offset_x = (label_size.width() - drawn_pixmap_w) / 2
        drawn_offset_y = (label_size.height() - drawn_pixmap_h) / 2

        # Convertir la posición del ratón (en coordenadas del QLabel) a coordenadas del pixmap original
        x_on_original = (mouse_pos_in_label.x() - drawn_offset_x) / scale_factor
        y_on_original = (mouse_pos_in_label.y() - drawn_offset_y) / scale_factor
        
        # Calcular la región a copiar del pixmap original
        x_region_orig = int(x_on_original - half_frame)
        y_region_orig = int(y_on_original - half_frame)

        # Asegurarse de que la región esté dentro de los límites del pixmap original
        x_region_orig = max(0, x_region_orig)
        y_region_orig = max(0, y_region_orig)
        
        # Asegurar que el ancho y alto de la región no excedan los límites del pixmap original
        width_region = self.label_imagen._frame_size
        height_region = self.label_imagen._frame_size
        
        if x_region_orig + width_region > original_size.width():
            width_region = original_size.width() - x_region_orig
        if y_region_orig + height_region > original_size.height():
            height_region = original_size.height() - y_region_orig
            
        if width_region <= 0 or height_region <= 0:
            self.zoom_label.clear()
            return
            
        # Copiar la región del pixmap original
        region = self.pixmap_original.copy(QRect(x_region_orig, y_region_orig, width_region, height_region))

        # Escalar a 200x100 (zoom de ~13x)
        # Considera si quieres Qt.IgnoreAspectRatio o Qt.KeepAspectRatio
        zoom_pixmap = region.scaled(200, 100, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.zoom_label.setPixmap(zoom_pixmap)


if __name__ == "__main__":
    app = QApplication([])
    ventana = MiVentana()
    ventana.show()
    app.exec_()