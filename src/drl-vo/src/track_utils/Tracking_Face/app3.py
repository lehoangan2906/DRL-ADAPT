# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import numpy as np
import os
import sys
import cv2, imutils
from PyQt5.QtGui import QImage, QPixmap,QImage
from PyQt5.QtCore import QTimer,QThread, pyqtSignal
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import pandas as pd
import keyboard
import time
from time import sleep
import pickle

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1200, 1000)
        self.label_6 = QtWidgets.QLabel(Dialog)
        self.label_6.setGeometry(QtCore.QRect(810, 30, 300, 300))
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")
        self.formLayoutWidget = QtWidgets.QWidget(Dialog)
        self.formLayoutWidget.setGeometry(QtCore.QRect(770, 390, 361, 155))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout_2 = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout_2.setContentsMargins(0, 0, 0, 0)
        self.formLayout_2.setObjectName("formLayout_2")
        self.lineEdit_1 = QtWidgets.QLineEdit(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.lineEdit_1.setFont(font)
        self.lineEdit_1.setObjectName("lineEdit_1")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit_1)
        self.label_1 = QtWidgets.QLabel(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_1.setFont(font)
        self.label_1.setObjectName("label_1")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_1)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.lineEdit_2.setFont(font)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_2)
        self.label_2 = QtWidgets.QLabel(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.lineEdit_3.setFont(font)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.lineEdit_3)
        self.label_3 = QtWidgets.QLabel(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.lineEdit_4.setFont(font)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.lineEdit_4)
        self.label_4 = QtWidgets.QLabel(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setGeometry(QtCore.QRect(30, 70, 640, 480))
        self.label_5.setText("")
        self.label_5.setObjectName("label_5")
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(170, 640, 261, 51))
        font = QtGui.QFont()
        font.setFamily("SketchFlow Print")
        font.setPointSize(13)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setIconSize(QtCore.QSize(20, 16))
        self.pushButton.setCheckable(True)
        self.pushButton.setObjectName("pushButton")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(630, 630, 481, 191))
        font = QtGui.QFont()
        font.setPointSize(24)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        ##############################################
        self.timer = QTimer()
        self.pushButton.clicked.connect(self.show)
        # self.pushButton.clicked.connect(self.test)

        self.vid = cv2.VideoCapture(0)
        self.guide = cv2.VideoCapture('face.mp4')
        self.timer.timeout.connect(self.loadCamera)
        self.tmp = True
        self.timer.timeout.connect(self.loadGuide)
        # self.timer.timeout.connect(self.test)
        self.timer.start(30)
        self.thread = None
        self.processing = False
        self.frame_counter = 0
        ###########################################
        self.app = FaceAnalysis(providers=[('CUDAExecutionProvider'),'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.database = 'database.pkl'
        self.datas = []
    def loadCamera(self):

        ret, self.image = self.vid.read()
        self.test2 = self.image.copy()
        if ret:

            self.image = imutils.resize(self.image, height=480)
            self.image = imutils.resize(self.image, width=640)
            frame = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            self.label_5.setPixmap(QtGui.QPixmap.fromImage(self.image))
        if self.processing:
            self.process_frame(self.test2)
    def loadGuide(self):
        # while True:
        ret, self.image2 = self.guide.read()
        if ret:
            self.image2 = imutils.resize(self.image2, height=300)
            self.image2 = imutils.resize(self.image2, width=300)

            frame = cv2.cvtColor(self.image2, cv2.COLOR_BGR2RGB)
            self.image2 = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            self.label_6.setPixmap(QtGui.QPixmap.fromImage(self.image2))
        if not ret:
            self.guide.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def capture_frame(self):
        self.processing = True
        self.frame_counter = 0

    def process_frame(self, frame):


        # print(self.infomation_list)
        ###########################################
        face = self.app.get(frame)

        if len(face)!=0:
            box = list(map(int, face[0].bbox))
            print(box)
            crop = frame[box[1]:box[3],box[0]:box[2]]
            data = [self.infomation_list[0],self.infomation_list[1],self.infomation_list[2],self.infomation_list[3], face[0].embedding]
            self.datas.append(data)
            if not os.path.exists(os.path.join('F:\\PROJECT\\KHOA_LUAN_MTMC\\tmp',self.infomation_list[-1])):
                os.makedirs(os.path.join('F:\\PROJECT\\KHOA_LUAN_MTMC\\tmp',self.infomation_list[-1]))
            print(os.path.join(os.path.join('F:\\PROJECT\\KHOA_LUAN_MTMC\\tmp',self.infomation_list[-1]),self.infomation_list[1]+str(self.frame_counter)+'.jpg'))
            cv2.imwrite(os.path.join(os.path.join('F:\\PROJECT\\KHOA_LUAN_MTMC\\tmp',self.infomation_list[-1]),self.infomation_list[0]+str(self.frame_counter)+'.jpg'),crop)
        # print(frame)
        self.frame_counter += 1
        if self.frame_counter >= 30:
            self.processing = False
            df = pd.DataFrame(self.datas, columns=['Name','Date','Com','ID', 'embedding'])
            if not os.path.exists(self.database):
                df.to_pickle(self.database)
                self.show_info_messagebox3()
            else:
                old_df = pd.read_pickle(self.database)
                new_df = pd.concat([df, old_df], ignore_index=True)
                print(new_df)
                new_df.to_pickle(self.database)
                self.show_info_messagebox3()
    def show(self):
        Name = self.lineEdit_1.text()
        Date = self.lineEdit_2.text()
        Com = self.lineEdit_3.text()
        ID = self.lineEdit_4.text()
        self.infomation_list = [Name,Date,Com,ID]

        # self.timer.timeout.connect(self.test)
        if any(value == '' for value in self.infomation_list) and not all(value == '' for value in self.infomation_list):
            self.show_warning_messagebox1()
        elif all(value == '' for value in self.infomation_list):
            self.show_warning_messagebox2()
        else:
            _translate = QtCore.QCoreApplication.translate

            self.label.setText(_translate("Dialog", "Bắt đầu thu thập dữ liệu trong \n15s"))
            self.processing = True
            self.frame_counter = 0


    def show_warning_messagebox1(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)

        # setting message for Message Box
        msg.setText("Vui lòng nhập đầy đủ các thông tin")

        # setting Message box window title
        msg.setWindowTitle("Cảnh báo")

        # declaring buttons on Message Box
        msg.setStandardButtons(QMessageBox.Ok )

        # start the app
        retval = msg.exec_()
    def show_warning_messagebox2(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)

        # setting message for Message Box
        msg.setText("Vui lòng nhập thông tin trước khi lấy mẫu")

        # setting Message box window title
        msg.setWindowTitle("Cảnh báo")

        # declaring buttons on Message Box
        msg.setStandardButtons(QMessageBox.Ok )

        # start the app
        retval = msg.exec_()

    def show_info_messagebox3(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        # setting message for Message Box
        msg.setText("Đã hoàn thành lấy mẫu, vui lòng thoát ứng dụng ")

        # setting Message box window title
        msg.setWindowTitle("Hoàn thành")

        # declaring buttons on Message Box
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

        # start the app
        retval = msg.exec_()
    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Chương trình thu thập dữ liệu"))
        self.label_1.setText(_translate("Dialog", "Ngày tháng năm sinh"))
        self.label_2.setText(_translate("Dialog", "Bộ phận"))
        self.label_3.setText(_translate("Dialog", "ID"))
        self.label_4.setText(_translate("Dialog", "Họ và tên"))
        self.pushButton.setText(_translate("Dialog", "Bắt đầu thu thập dữ liệu"))
        self.label.setText(_translate("Dialog", "Bắt đầu thu thập dữ liệu, xin\n"
" hãy nhập thông tin cá nhân \n"
"phía bên  trên"))



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())