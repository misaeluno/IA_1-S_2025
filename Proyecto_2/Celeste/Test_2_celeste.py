import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class ImageWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Selector de Imágenes PNG")
        self.setGeometry(100, 100, 600, 400)

        # Widgets
        self.image = None

        self.image_label = QLabel("Imagen no cargada", self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.select_button = QPushButton("Seleccionar Imagen PNG", self)
        self.select_button.clicked.connect(self.open_image_dialog)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.select_button)
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
        self.image = cv2.imread(path)
        if not self.image is None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            height, width = gray_image.shape
            bytes_per_line = width
            q_img = QImage(gray_image.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
            self.image_label.setPixmap(QPixmap.fromImage(q_img))
            self.image_label.setText("")  # Limpia el texto predeterminado
        else:
            self.image_label.setText("Error al cargar la imagen")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageWindow()
    window.show()
    sys.exit(app.exec())
