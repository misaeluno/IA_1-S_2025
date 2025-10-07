from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                             QFileDialog, QLabel, QScrollArea, QWidget, 
                             QVBoxLayout, QHBoxLayout)
from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap, QImage

class MiVentana(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hola, PyQt!")
        self.setGeometry(100, 100, 640, 480)
        
        # Variables para el marco y zoom
        self.marco_activo = False
        self.marco_pos = QPoint(0, 0)
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
        
        self.label_imagen = QLabel()
        self.label_imagen.setAlignment(Qt.AlignCenter)
        self.label_imagen.setMouseTracking(True)
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
            self.pixmap = QPixmap(ruta_imagen)
            self.label_imagen.setPixmap(self.pixmap)
            self.imagen_cargada = True

    def al_marco15x15(self):
        self.marco_activo = not self.marco_activo
        print("Marco 15x15:", "Activado" if self.marco_activo else "Desactivado")
        self.label_imagen.update()

    def al_salir(self):
        QApplication.quit()

    def mouseMoveEvent(self, event):
        if self.marco_activo and self.imagen_cargada:
            # Posición relativa al label_imagen
            pos_relativa = self.label_imagen.mapFromParent(event.pos())
            if 0 <= pos_relativa.x() <= self.label_imagen.width() and \
               0 <= pos_relativa.y() <= self.label_imagen.height():
                self.marco_pos = pos_relativa
                self.label_imagen.update()
                self.actualizar_zoom()

    def actualizar_zoom(self):
        """Extrae la región 15x15 bajo el marco y la amplía a 200x100."""
        if not self.imagen_cargada:
            return
        
        # Obtener la región 15x15 de la imagen original
        x, y = self.marco_pos.x() - 7, self.marco_pos.y() - 7
        region = self.label_imagen.pixmap().copy(QRect(x, y, 15, 15))
        
        # Escalar a 200x100 (zoom de ~13x)
        zoom_pixmap = region.scaled(200, 100, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.zoom_label.setPixmap(zoom_pixmap)

    def paintEvent(self, event):
        if self.marco_activo and self.imagen_cargada:
            painter = QPainter(self.label_imagen)
            pen = QPen(QColor(255, 0, 0), 2, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawRect(self.marco_pos.x() - 7, self.marco_pos.y() - 7, 15, 15)
            painter.end()

if __name__ == "__main__":
    app = QApplication([])
    ventana = MiVentana()
    ventana.show()
    app.exec_()