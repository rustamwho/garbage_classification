import os

from PyQt5.QtWidgets import QMessageBox


def show_error_message(text: str) -> None:
    """ Показ высплывающего окна с ошибкой.

    :param text: Текст ошибки
    """
    msg = QMessageBox()
    msg.setWindowTitle('Ошибка')
    msg.setText(text)
    msg.setIcon(QMessageBox.Warning)

    msg.exec_()


def is_image(file: str) -> bool:
    """ Возвращает True, если file - изображение. Если нет - False. """
    return any(file.endswith(ext) for ext in ('.jpeg', '.jpg', '.png'))


def get_images_count(path: str) -> int:
    """ Возвращает количество изображений в заданной директории. """
    images_count = 0
    for file in os.listdir(path):
        if is_image(file):
            images_count += 1
    return images_count


def get_images_in_path(path: str) -> list[str]:
    """ Возвращает список из названий изображений. """
    return [file for file in os.listdir(path) if is_image(file)]
