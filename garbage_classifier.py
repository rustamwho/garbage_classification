import os

from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import cv2 as cv2

from helpers import get_images_in_path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Уровень предупреждения


class Model:
    def __init__(self, csv_path: str, model_path: str):
        """ Подготовка нейросетевой модели для классификации.
        :param csv_path: адрес csv файла с классами
        :param model_path: адрес файла обученной модели (*.h5)
        """
        # Чтение классов, на которых обучена нейросеть, с csv файла
        self.class_df = pd.read_csv(csv_path)
        self.class_count = len(self.class_df['class'].unique())
        self.img_height = int(self.class_df['height'].iloc[0])
        self.img_width = int(self.class_df['width'].iloc[0])
        self.img_size = (self.img_width, self.img_height)
        scale = self.class_df['scale by'].iloc[0]
        # Вычисление значения для масштабирования пикселей изображения
        try:
            s = int(scale)
            self.s2 = 1
            self.s1 = 0
        except:
            split = scale.split('-')
            self.s1 = float(split[1])
            self.s2 = float(split[0].split('*')[1])

        # Загрузка модели из файла
        self.model = load_model(model_path)

    def _prepare_images(self, store_path: str):
        """ Преподготовка изображений для передачи в нейросеть. """
        images = get_images_in_path(store_path)
        image_list = []
        for image_file in images:
            try:
                img = cv2.imread(os.path.join(store_path, image_file))
                img = cv2.resize(img, self.img_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img * self.s2 - self.s1
                image_list.append(img)
            except:
                continue
        image_array = np.array(image_list)

        return images, image_array

    def predict(self, store_path, averaged=True):
        """ Классификация изображений. """
        images, image_array = self._prepare_images(store_path)

        predictions = self.model.predict(image_array)

        # Если надо брать среднее значение результатов
        if averaged:
            psum = [0] * self.class_count
            for prediction in predictions:
                for i in range(self.class_count):
                    # Сумма всех вероятностей
                    psum[i] = psum[i] + prediction[i]
            # Поиск индекса класса с максимальным значением суммы вероятностей
            index = np.argmax(psum)
            # Название класса по найденному индексу
            klass = self.class_df['class'].iloc[index]
            # Среднее значение вероятности по всем изображениям
            prob = psum[index] / len(images)
            return klass, prob, None
        # Индивидуальные результаты для каждого изображения
        else:
            pred_class = []
            prob_list = []
            for i, p in enumerate(predictions):
                # Индекс класса с максимальным значением вероятности
                index = np.argmax(p)
                # Название класса по индексу
                klass = self.class_df['class'].iloc[index]
                pred_class.append(klass)
                prob_list.append(p[index])
            # Составление DataFrame с результатами всех изображений
            Fseries = pd.Series(images, name='image file')
            Lseries = pd.Series(pred_class, name='species')
            Pseries = pd.Series(prob_list, name='probability')
            df = pd.concat([Fseries, Lseries, Pseries], axis=1)
            return None, None, df


def main():
    """ Ручной запуск нейросети. """
    store_path = './test_path/'
    csv_path = './trained_model/12_classes_dict.csv'
    model_path = './trained_model/12_classes_EfficientNetB3.h5'
    model = Model(csv_path, model_path)

    klass, prob, df = model.predict(store_path, averaged=True)
    msg = (
        f'image is of class {klass} with a probability of {prob * 100: 6.2f}%'
    )
    print(msg)


if __name__ == '__main__':
    main()
