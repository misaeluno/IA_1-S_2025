import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt


class ImageWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Selector de Imágenes PNG")
        self.setGeometry(350, 50, 500, 500)

        # Widgets
        self.image_label = QLabel("Imagen no cargada", self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.select_button = QPushButton("Seleccionar Imagen PNG", self)
        self.select_button.clicked.connect(self.open_image_dialog)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def open_image_dialog(self):
        # Abre el diálogo para seleccionar archivo
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar Imagen",  # Título del diálogo
            "",  # Directorio inicial (vacío = directorio actual)
            "Imágenes PNG (*.png);;Todos los archivos (*)",  # Filtros
        )

        if file_path:  # Si se seleccionó un archivo
            self.load_image(file_path)

    def load_image(self, path):
        # Carga la imagen y la muestra en el QLabel
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            self.image_label.setPixmap(
                pixmap.scaled(
                    self.image_label.width(),
                    self.image_label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                )
            )
            self.image_label.setText("")  # Limpia el texto predeterminado
        else:
            self.image_label.setText("Error al cargar la imagen")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageWindow()
    window.show()
    sys.exit(app.exec())
