# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'garbage_classifier_gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(637, 477)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.select_path_btn = QtWidgets.QPushButton(self.centralwidget)
        self.select_path_btn.setGeometry(QtCore.QRect(490, 60, 131, 32))
        self.select_path_btn.setObjectName("select_path_btn")
        self.selected_path_text = QtWidgets.QTextBrowser(self.centralwidget)
        self.selected_path_text.setGeometry(QtCore.QRect(10, 60, 471, 31))
        self.selected_path_text.setObjectName("selected_path_text")
        self.load_model_btn = QtWidgets.QPushButton(self.centralwidget)
        self.load_model_btn.setGeometry(QtCore.QRect(240, 10, 141, 32))
        self.load_model_btn.setObjectName("load_model_btn")
        self.predict_btn = QtWidgets.QPushButton(self.centralwidget)
        self.predict_btn.setEnabled(False)
        self.predict_btn.setGeometry(QtCore.QRect(260, 350, 113, 32))
        self.predict_btn.setObjectName("predict_btn")
        self.result_label = QtWidgets.QLabel(self.centralwidget)
        self.result_label.setEnabled(True)
        self.result_label.setGeometry(QtCore.QRect(140, 390, 361, 16))
        self.result_label.setText("")
        self.result_label.setObjectName("result_label")
        self.example_image_label = QtWidgets.QLabel(self.centralwidget)
        self.example_image_label.setGeometry(QtCore.QRect(180, 110, 271, 201))
        self.example_image_label.setText("")
        self.example_image_label.setObjectName("example_image_label")
        self.images_count_label = QtWidgets.QLabel(self.centralwidget)
        self.images_count_label.setGeometry(QtCore.QRect(240, 320, 161, 20))
        self.images_count_label.setText("")
        self.images_count_label.setObjectName("images_count_label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 637, 43))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.select_trained_model_action = QtWidgets.QAction(MainWindow)
        self.select_trained_model_action.setObjectName("select_trained_model_action")
        self.select_classes_csv_action = QtWidgets.QAction(MainWindow)
        self.select_classes_csv_action.setObjectName("select_classes_csv_action")
        self.menu.addAction(self.select_trained_model_action)
        self.menu.addAction(self.select_classes_csv_action)
        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Классификация мусора"))
        self.select_path_btn.setText(_translate("MainWindow", "Выбрать папку"))
        self.load_model_btn.setText(_translate("MainWindow", "Загрузить модель"))
        self.predict_btn.setText(_translate("MainWindow", "Распознать"))
        self.menu.setTitle(_translate("MainWindow", "Настройка"))
        self.select_trained_model_action.setText(_translate("MainWindow", "Выбрать обученную модель"))
        self.select_classes_csv_action.setText(_translate("MainWindow", "Выбрать csv классов"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())