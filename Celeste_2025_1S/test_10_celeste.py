import sys
import os
import cv2
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QComboBox
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QMouseEvent, QKeySequence, QShortcut
from PyQt6.QtCore import Qt, QPoint, QRect


class Бога(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IA: Segunda evaluación, Clasificador de Terrenos")
        self.setGeometry(100, 100, 1000, 600)

        self.categories = {
                    1: "agua",
                    2: "vegetacion",
                    3: "montañas",
                    4: "desiertos",
                    5: "rios",
                    6: "ciudades"
                }
        self.image = None
        self.grayImage = None
        self.currentPos = QPoint(0, 0)
        self.currentCat = 1
        self.currentTake = 1
        self.cropSize = 15
        self.zoomFactor = 10
        self.imageFilename = ""

        self.SetupUI()
        self.SetupShortcuts()
        self.CreateFolders()

    def SetupUI(self):
        # Widgets principales
        self.image_label = QLabel("Imagen no cargada", self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.image_label.setFixedSize(512, 512)
        self.image_label.setMouseTracking(True)
        self.image_label.mouseMoveEvent = self.MouseMoveEvent
        self.image_label.mousePressEvent = self.MouseClickEvent

        self.preview_label = QLabel("Vista previa (15x15)", self)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setFixedSize(150, 150)  # Tamaño fijo para la vista previa

        self.category_combo = QComboBox(self)
        for num, cat in self.categories.items():
            self.category_combo.addItem(f"{num}) {cat.capitalize()}", cat)
        self.category_combo.currentIndexChanged.connect(self.UpdateCategory)

        self.select_button = QPushButton("Seleccionar Imagen (Ctrl + O)", self)
        self.select_button.setFixedWidth(512)
        self.select_button.clicked.connect(self.OpenImageDialog)

        self.statusBar().showMessage("Seleccione una categoría y cargue una imagen")

        # Layouts
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        left_layout.addWidget(self.select_button)
        left_layout.addWidget(self.image_label)

        right_layout.addWidget(self.category_combo)
        right_layout.addSpacing(20)
        right_layout.addWidget(self.preview_label)

        main_layout.addLayout(left_layout, 70)
        main_layout.addLayout(right_layout, 30)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def SetupShortcuts(self):
         # Atajos para categorías (1-6)
         for num in self.categories.keys():
             shortcut = QShortcut(QKeySequence(str(num)), self)
             shortcut.activated.connect(lambda n=num: self.SetCategoryByNumber(n))

         # Atajo para guardar (Espacio)
         QShortcut(QKeySequence(Qt.Key.Key_Space), self).activated.connect(self.SaveCrop)

         # Atajo para abrir imagen (Ctrl+O)
         QShortcut(QKeySequence("Ctrl+O"), self).activated.connect(self.OpenImageDialog)

    def CreateFolders(self):
        self.output_dir = "dataset/samples"
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_fft_dir = "dataset/fourier"
        os.makedirs(self.output_fft_dir, exist_ok=True)

        folders = ["C1", "C2", "C3", "C4", "C5", "C6"]
        for category in folders:
            os.makedirs(os.path.join(self.output_dir, category), exist_ok=True)
            os.makedirs(os.path.join(self.output_fft_dir, category), exist_ok=True)

    def SetCategoryByNumber(self, number):
        if number in self.categories:
            self.category_combo.setCurrentIndex(number-1)
            self.statusBar().showMessage(f"Categoría seleccionada: {self.categories[number]}", 2000)

    def UpdateCategory(self, index):
        self.currentCat = self.category_combo.currentIndex() + 1
        self.statusBar().showMessage(f"Categoría cambiada a: {self.category_combo.currentText()}", 2000)

    def OpenImageDialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar Imagen",
            "",
            "Imágenes (*.png *.jpg *.jpeg);;Todos los archivos (*)",
        )
        if file_path:
            self.LoadImage(file_path)
            self.imageFilename = os.path.splitext(os.path.basename(file_path))[0]
            self.statusBar().showMessage(f"Imagen cargada: {self.imageFilename}", 3000)

    def LoadImage(self, path):
        self.image = cv2.imread(path)
        if self.image is not None:
            self.grayImage = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            height, width = self.grayImage.shape
            bytes_per_line = width
            q_img = QImage(self.grayImage.data, width, height, bytes_per_line,
                          QImage.Format.Format_Grayscale8, None, None)

            # Escalar la imagen para que quepa en la ventana manteniendo aspect ratio
            scaled_pixmap = QPixmap.fromImage(q_img).scaled(
                self.image_label.width(), self.image_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.setText("")

            # Guardar factores de escala para mapear coordenadas
            self.scale_x = width / scaled_pixmap.width()
            self.scale_y = height / scaled_pixmap.height()
        else: self.image_label.setText("Error al cargar la imagen")

    def MouseMoveEvent(self, ev : QMouseEvent | None):
        if ev is not None and self.grayImage is not None:
            x = int(ev.pos().x() * self.scale_x)
            y = int(ev.pos().y() * self.scale_y)

            height, width = self.grayImage.shape
            x = max(0, min(x, width - self.cropSize))
            y = max(0, min(y, height - self.cropSize))
            self.currentPos = QPoint(x, y)
            self.UpdatePreview()

    def MouseClickEvent(self, ev : QMouseEvent | None):
        if ev is not None and ev.button() == Qt.MouseButton.LeftButton and self.grayImage is not None:
            self.SaveCrop()

    def UpdatePreview(self):
        if self.grayImage is not None and self.currentPos is not None:
            x, y = self.currentPos.x(), self.currentPos.y()

            # Obtener el recorte de 15x15
            crop = self.grayImage[y:y+self.cropSize, x:x+self.cropSize]

            # Aplicar zoom a la vista previa
            zoomed_crop = cv2.resize(crop, None, fx=self.zoomFactor, fy=self.zoomFactor,
                                   interpolation=cv2.INTER_NEAREST)

            # Convertir a QImage y mostrar
            height, width = zoomed_crop.shape
            bytes_per_line = width
            q_img = QImage(zoomed_crop.data, width, height, bytes_per_line,
                          QImage.Format.Format_Grayscale8, None, None)

            self.preview_label.setPixmap(QPixmap.fromImage(q_img))

    def SaveCrop(self):
        if self.grayImage is not None and hasattr(self, 'imageFilename'):
            x, y = self.currentPos.x(), self.currentPos.y()
            crop = self.grayImage[y:y+self.cropSize, x:x+self.cropSize]

            base_name = f"I{self.currentCat}_{self.currentTake:03}.png"
            save_path = os.path.join(self.output_dir, 'C' + str(self.currentCat), base_name)
            print(f"Guardando recorte como {save_path}")
            cv2.imwrite(save_path, crop)

            if self.statusBar() is not None:
                self.statusBar().showMessage(f"Recorte guardado como {base_name}", 3000)
                self.currentTake += 1


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Бога()
    window.show()
    sys.exit(app.exec())
