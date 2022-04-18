import os
import sys

from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow

from garbage_classifier import Model
from gui.garbage_classifier_gui import Ui_MainWindow
from helpers import show_error_message, is_image, get_images_count


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.csv_path = './trained_model/12_classes_dict.csv'
        self.model_path = './trained_model/12_classes_EfficientNetB3.h5'

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.work_path = False
        self.images_count = 0
        self.model = False

        self.add_functions()
        self.ui.statusbar.showMessage('Загрузите модель')

    def add_functions(self):
        """ Назначение функций нажатию кнопок. """
        self.ui.select_path_btn.clicked.connect(self.get_path_dialog)
        self.ui.load_model_btn.clicked.connect(self.load_model)
        self.ui.predict_btn.clicked.connect(self.predict)
        self.ui.select_trained_model_action.triggered.connect(
            self.select_trained_model
        )
        self.ui.select_classes_csv_action.triggered.connect(
            self.select_csv_classes
        )

    def select_trained_model(self) -> None:
        """
        Выбор файла с обученной нейросетевой моделью в диалоговом окне.
        Архитектура нейросети - EfficientNetB3.
        """
        trained_model_file = QtWidgets.QFileDialog.getOpenFileName(
            caption='Выбрать обученную модель',
            filter='Trained model (*.h5)')[0]
        if trained_model_file:
            self.model_path = trained_model_file

    def select_csv_classes(self) -> None:
        """
        Выбор файла с классами, на которых обучена нейросеть, в диалоговом окне
        """
        csv_file = QtWidgets.QFileDialog.getOpenFileName(
            caption='Выбрать файл с классами',
            filter='Classes (*.csv)')[0]
        if csv_file:
            self.csv_path = csv_file

    def get_path_dialog(self) -> None:
        self.work_path = QtWidgets.QFileDialog.getExistingDirectory()
        if not self.work_path:
            return

        # Вывод адреса директории в строку
        self.ui.selected_path_text.setText(self.work_path)
        # Подсчет количества картинок в директории
        self.images_count = get_images_count(self.work_path)

        if self.images_count:
            # Отображения первого изображения, если оно есть
            self.show_example_image()
            self.ui.images_count_label.setText(f'Найдено изображений: '
                                               f'{self.images_count}')
        else:
            self.ui.example_image_label.clear()
            self.ui.images_count_label.setText('Изображений нет')

        # Активация кнопки "Распознать"
        self.ui.predict_btn.setEnabled(True)

    def show_example_image(self) -> None:
        """ Отображение первой картинки из выбранной директории. """
        example_image = False
        # Поиск первого изображения в директории
        for file in os.listdir(self.work_path):
            if is_image(file):
                example_image = self.work_path + '/' + file
                break
        if example_image:
            pixmap = QPixmap(example_image)
            # Масштабирование изображения под label
            scaled_pixmap = pixmap.scaled(self.ui.example_image_label.size(),
                                          Qt.KeepAspectRatioByExpanding)
            self.ui.example_image_label.setPixmap(scaled_pixmap)

    def load_model(self) -> None:
        """ Подготовка нейросетевой модели для классификации изображений. """
        try:
            self.model = Model(self.csv_path, self.model_path)
            self.ui.statusbar.showMessage('Модель загружена')
        except:
            show_error_message('Не удалось загрузить модель')

    def predict(self) -> None:
        """ Классификация изображений в выбранной директории. """
        if not self.model:
            show_error_message('Загрузите модель')
            return
        if not self.work_path or not self.images_count:
            show_error_message('Выберите директорию с изображениями')
            return

        klass, prob, _ = self.model.predict(self.work_path)
        # Отображение результатов
        message = (f'Image is of class {klass} with a probability of '
                   f'{prob * 100: 6.2f} %')
        self.ui.result_label.setText(message)


def application() -> None:
    """ Отрисовка окна. """
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()

    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    application()
