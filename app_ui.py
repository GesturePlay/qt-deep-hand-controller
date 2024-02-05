# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'app.ui'
##
## Created by: Qt User Interface Compiler version 6.6.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QLabel,
    QLineEdit, QMainWindow, QMenuBar, QPushButton,
    QSizePolicy, QStackedWidget, QStatusBar, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1024, 768)
        palette = QPalette()
        brush = QBrush(QColor(121, 131, 145, 255))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Window, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Window, brush)
        palette.setBrush(QPalette.Disabled, QPalette.Base, brush)
        palette.setBrush(QPalette.Disabled, QPalette.Window, brush)
        MainWindow.setPalette(palette)
        icon = QIcon()
        icon.addFile(u"Images/Logo/MainLogo.png", QSize(), QIcon.Normal, QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setIconSize(QSize(30, 30))
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.stackedWidget = QStackedWidget(self.centralwidget)
        self.stackedWidget.setObjectName(u"stackedWidget")
        self.stackedWidget.setGeometry(QRect(0, 0, 1021, 751))
        self.page_1 = QWidget()
        self.page_1.setObjectName(u"page_1")
        self.label_2 = QLabel(self.page_1)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(450, 0, 161, 61))
        palette1 = QPalette()
        brush1 = QBrush(QColor(255, 255, 255, 255))
        brush1.setStyle(Qt.SolidPattern)
        palette1.setBrush(QPalette.Active, QPalette.WindowText, brush1)
        palette1.setBrush(QPalette.Active, QPalette.Text, brush1)
        palette1.setBrush(QPalette.Active, QPalette.ButtonText, brush1)
        brush2 = QBrush(QColor(255, 255, 255, 128))
        brush2.setStyle(Qt.SolidPattern)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette1.setBrush(QPalette.Active, QPalette.PlaceholderText, brush2)
#endif
        palette1.setBrush(QPalette.Inactive, QPalette.WindowText, brush1)
        palette1.setBrush(QPalette.Inactive, QPalette.Text, brush1)
        palette1.setBrush(QPalette.Inactive, QPalette.ButtonText, brush1)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette1.setBrush(QPalette.Inactive, QPalette.PlaceholderText, brush2)
#endif
        palette1.setBrush(QPalette.Disabled, QPalette.WindowText, brush1)
        palette1.setBrush(QPalette.Disabled, QPalette.Text, brush1)
        palette1.setBrush(QPalette.Disabled, QPalette.ButtonText, brush1)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette1.setBrush(QPalette.Disabled, QPalette.PlaceholderText, brush2)
#endif
        self.label_2.setPalette(palette1)
        font = QFont()
        font.setFamilies([u"Eras Demi ITC"])
        font.setPointSize(26)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.label_2.setScaledContents(True)
        self.label_4 = QLabel(self.page_1)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(600, 500, 91, 41))
        palette2 = QPalette()
        palette2.setBrush(QPalette.Active, QPalette.WindowText, brush1)
        palette2.setBrush(QPalette.Active, QPalette.Text, brush1)
        palette2.setBrush(QPalette.Active, QPalette.ButtonText, brush1)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette2.setBrush(QPalette.Active, QPalette.PlaceholderText, brush2)
#endif
        palette2.setBrush(QPalette.Inactive, QPalette.WindowText, brush1)
        palette2.setBrush(QPalette.Inactive, QPalette.Text, brush1)
        palette2.setBrush(QPalette.Inactive, QPalette.ButtonText, brush1)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette2.setBrush(QPalette.Inactive, QPalette.PlaceholderText, brush2)
#endif
        palette2.setBrush(QPalette.Disabled, QPalette.WindowText, brush1)
        palette2.setBrush(QPalette.Disabled, QPalette.Text, brush1)
        palette2.setBrush(QPalette.Disabled, QPalette.ButtonText, brush1)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette2.setBrush(QPalette.Disabled, QPalette.PlaceholderText, brush2)
#endif
        self.label_4.setPalette(palette2)
        font1 = QFont()
        font1.setFamilies([u"Eras Demi ITC"])
        font1.setPointSize(20)
        self.label_4.setFont(font1)
        self.label_4.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.label_4.setScaledContents(True)
        self.line = QFrame(self.page_1)
        self.line.setObjectName(u"line")
        self.line.setGeometry(QRect(0, 360, 1024, 20))
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)
        self.AddProfileButton = QPushButton(self.page_1)
        self.AddProfileButton.setObjectName(u"AddProfileButton")
        self.AddProfileButton.setGeometry(QRect(590, 400, 100, 100))
        self.AddProfileButton.setCursor(QCursor(Qt.PointingHandCursor))
        self.AddProfileButton.setAutoFillBackground(False)
        self.AddProfileButton.setStyleSheet(u"QPushButton {\n"
"      background-color: rgba(255, 255, 255, 150);\n"
"      border-radius: 50%;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"      background-color: rgba(236, 236, 236, 30);\n"
"}")
        icon1 = QIcon()
        icon1.addFile(u"Images/1x/add.png", QSize(), QIcon.Normal, QIcon.Off)
        self.AddProfileButton.setIcon(icon1)
        self.AddProfileButton.setIconSize(QSize(50, 50))
        self.label_6 = QLabel(self.page_1)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(380, 510, 111, 41))
        palette3 = QPalette()
        palette3.setBrush(QPalette.Active, QPalette.WindowText, brush1)
        palette3.setBrush(QPalette.Active, QPalette.Text, brush1)
        palette3.setBrush(QPalette.Active, QPalette.ButtonText, brush1)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette3.setBrush(QPalette.Active, QPalette.PlaceholderText, brush2)
#endif
        palette3.setBrush(QPalette.Inactive, QPalette.WindowText, brush1)
        palette3.setBrush(QPalette.Inactive, QPalette.Text, brush1)
        palette3.setBrush(QPalette.Inactive, QPalette.ButtonText, brush1)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette3.setBrush(QPalette.Inactive, QPalette.PlaceholderText, brush2)
#endif
        palette3.setBrush(QPalette.Disabled, QPalette.WindowText, brush1)
        palette3.setBrush(QPalette.Disabled, QPalette.Text, brush1)
        palette3.setBrush(QPalette.Disabled, QPalette.ButtonText, brush1)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette3.setBrush(QPalette.Disabled, QPalette.PlaceholderText, brush2)
#endif
        self.label_6.setPalette(palette3)
        self.label_6.setFont(font1)
        self.label_6.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.label_6.setScaledContents(True)
        self.label_7 = QLabel(self.page_1)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setGeometry(QRect(80, 330, 311, 41))
        palette4 = QPalette()
        palette4.setBrush(QPalette.Active, QPalette.WindowText, brush1)
        palette4.setBrush(QPalette.Active, QPalette.Text, brush1)
        palette4.setBrush(QPalette.Active, QPalette.ButtonText, brush1)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette4.setBrush(QPalette.Active, QPalette.PlaceholderText, brush2)
#endif
        palette4.setBrush(QPalette.Inactive, QPalette.WindowText, brush1)
        palette4.setBrush(QPalette.Inactive, QPalette.Text, brush1)
        palette4.setBrush(QPalette.Inactive, QPalette.ButtonText, brush1)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette4.setBrush(QPalette.Inactive, QPalette.PlaceholderText, brush2)
#endif
        palette4.setBrush(QPalette.Disabled, QPalette.WindowText, brush1)
        palette4.setBrush(QPalette.Disabled, QPalette.Text, brush1)
        palette4.setBrush(QPalette.Disabled, QPalette.ButtonText, brush1)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette4.setBrush(QPalette.Disabled, QPalette.PlaceholderText, brush2)
#endif
        self.label_7.setPalette(palette4)
        font2 = QFont()
        font2.setFamilies([u"Eras Demi ITC"])
        font2.setPointSize(16)
        self.label_7.setFont(font2)
        self.label_7.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.label_7.setScaledContents(True)
        self.label_8 = QLabel(self.page_1)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setGeometry(QRect(320, 20, 401, 321))
        palette5 = QPalette()
        palette5.setBrush(QPalette.Active, QPalette.WindowText, brush1)
        palette5.setBrush(QPalette.Active, QPalette.Text, brush1)
        palette5.setBrush(QPalette.Active, QPalette.ButtonText, brush1)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette5.setBrush(QPalette.Active, QPalette.PlaceholderText, brush2)
#endif
        palette5.setBrush(QPalette.Inactive, QPalette.WindowText, brush1)
        palette5.setBrush(QPalette.Inactive, QPalette.Text, brush1)
        palette5.setBrush(QPalette.Inactive, QPalette.ButtonText, brush1)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette5.setBrush(QPalette.Inactive, QPalette.PlaceholderText, brush2)
#endif
        palette5.setBrush(QPalette.Disabled, QPalette.WindowText, brush1)
        palette5.setBrush(QPalette.Disabled, QPalette.Text, brush1)
        palette5.setBrush(QPalette.Disabled, QPalette.ButtonText, brush1)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette5.setBrush(QPalette.Disabled, QPalette.PlaceholderText, brush2)
#endif
        self.label_8.setPalette(palette5)
        font3 = QFont()
        font3.setFamilies([u"Eras Demi ITC"])
        font3.setPointSize(36)
        self.label_8.setFont(font3)
        self.label_8.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.label_8.setTextFormat(Qt.PlainText)
        self.label_8.setPixmap(QPixmap(u"Images/Logo/GesturePlayLogo_yello.png"))
        self.label_8.setScaledContents(True)
        self.label_3 = QLabel(self.page_1)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(510, 60, 41, 61))
        palette6 = QPalette()
        palette6.setBrush(QPalette.Active, QPalette.WindowText, brush1)
        palette6.setBrush(QPalette.Active, QPalette.Text, brush1)
        palette6.setBrush(QPalette.Active, QPalette.ButtonText, brush1)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette6.setBrush(QPalette.Active, QPalette.PlaceholderText, brush2)
#endif
        palette6.setBrush(QPalette.Inactive, QPalette.WindowText, brush1)
        palette6.setBrush(QPalette.Inactive, QPalette.Text, brush1)
        palette6.setBrush(QPalette.Inactive, QPalette.ButtonText, brush1)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette6.setBrush(QPalette.Inactive, QPalette.PlaceholderText, brush2)
#endif
        palette6.setBrush(QPalette.Disabled, QPalette.WindowText, brush1)
        palette6.setBrush(QPalette.Disabled, QPalette.Text, brush1)
        palette6.setBrush(QPalette.Disabled, QPalette.ButtonText, brush1)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette6.setBrush(QPalette.Disabled, QPalette.PlaceholderText, brush2)
#endif
        self.label_3.setPalette(palette6)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet(u"\n"
"color: rgb(255, 255, 255);")
        self.label_3.setScaledContents(True)
        self.label_5 = QLabel(self.page_1)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(600, 530, 91, 41))
        palette7 = QPalette()
        palette7.setBrush(QPalette.Active, QPalette.WindowText, brush1)
        palette7.setBrush(QPalette.Active, QPalette.Text, brush1)
        palette7.setBrush(QPalette.Active, QPalette.ButtonText, brush1)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette7.setBrush(QPalette.Active, QPalette.PlaceholderText, brush2)
#endif
        palette7.setBrush(QPalette.Inactive, QPalette.WindowText, brush1)
        palette7.setBrush(QPalette.Inactive, QPalette.Text, brush1)
        palette7.setBrush(QPalette.Inactive, QPalette.ButtonText, brush1)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette7.setBrush(QPalette.Inactive, QPalette.PlaceholderText, brush2)
#endif
        palette7.setBrush(QPalette.Disabled, QPalette.WindowText, brush1)
        palette7.setBrush(QPalette.Disabled, QPalette.Text, brush1)
        palette7.setBrush(QPalette.Disabled, QPalette.ButtonText, brush1)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette7.setBrush(QPalette.Disabled, QPalette.PlaceholderText, brush2)
#endif
        self.label_5.setPalette(palette7)
        self.label_5.setFont(font1)
        self.label_5.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.label_5.setScaledContents(True)
        self.Profile1 = QPushButton(self.page_1)
        self.Profile1.setObjectName(u"Profile1")
        self.Profile1.setGeometry(QRect(370, 390, 111, 121))
        self.Profile1.setCursor(QCursor(Qt.OpenHandCursor))
        self.Profile1.setStyleSheet(u"QPushButton{\n"
"      color: rgb(255, 255, 255);\n"
"      border:none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"      background-color: rgba(236, 236, 236, 30);\n"
"}")
        icon2 = QIcon()
        icon2.addFile(u"Images/1x/Profile1.png", QSize(), QIcon.Normal, QIcon.Off)
        self.Profile1.setIcon(icon2)
        self.Profile1.setIconSize(QSize(100, 100))
        self.label_9 = QLabel(self.page_1)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setGeometry(QRect(-2, -5, 1031, 731))
        self.label_9.setStyleSheet(u"background-color: rgb(64, 70, 77);")
        self.label_9.setAlignment(Qt.AlignBottom|Qt.AlignLeading|Qt.AlignLeft)
        self.stackedWidget.addWidget(self.page_1)
        self.label_9.raise_()
        self.label_2.raise_()
        self.label_4.raise_()
        self.line.raise_()
        self.AddProfileButton.raise_()
        self.label_6.raise_()
        self.label_7.raise_()
        self.label_8.raise_()
        self.label_3.raise_()
        self.label_5.raise_()
        self.Profile1.raise_()
        self.page_2 = QWidget()
        self.page_2.setObjectName(u"page_2")
        self.logoutBtn = QPushButton(self.page_2)
        self.logoutBtn.setObjectName(u"logoutBtn")
        self.logoutBtn.setGeometry(QRect(10, 660, 191, 51))
        font4 = QFont()
        font4.setFamilies([u"Eras Bold ITC"])
        font4.setPointSize(12)
        self.logoutBtn.setFont(font4)
        self.logoutBtn.setCursor(QCursor(Qt.PointingHandCursor))
        self.logoutBtn.setLayoutDirection(Qt.LeftToRight)
        self.logoutBtn.setStyleSheet(u"QPushButton{\n"
"      color: rgb(255, 255, 255);\n"
"      border:none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"      background-color: rgb(153, 178, 208);\n"
"}")
        self.logoutBtn.setIcon(icon2)
        self.logoutBtn.setIconSize(QSize(50, 55))
        self.cameraSettingsBtn = QPushButton(self.page_2)
        self.cameraSettingsBtn.setObjectName(u"cameraSettingsBtn")
        self.cameraSettingsBtn.setGeometry(QRect(0, 213, 231, 71))
        self.cameraSettingsBtn.setFont(font4)
        self.cameraSettingsBtn.setCursor(QCursor(Qt.PointingHandCursor))
        self.cameraSettingsBtn.setLayoutDirection(Qt.LeftToRight)
        self.cameraSettingsBtn.setStyleSheet(u"QPushButton{\n"
"      color: rgb(255, 255, 255);\n"
"      border:none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"      background-color: rgb(153, 178, 208);\n"
"}")
        icon3 = QIcon()
        icon3.addFile(u"Images/Icons/camera-setting-icon.png", QSize(), QIcon.Normal, QIcon.Off)
        self.cameraSettingsBtn.setIcon(icon3)
        self.cameraSettingsBtn.setIconSize(QSize(50, 55))
        self.controlsSettingsBtn = QPushButton(self.page_2)
        self.controlsSettingsBtn.setObjectName(u"controlsSettingsBtn")
        self.controlsSettingsBtn.setGeometry(QRect(0, 71, 231, 71))
        self.controlsSettingsBtn.setFont(font4)
        self.controlsSettingsBtn.setCursor(QCursor(Qt.PointingHandCursor))
        self.controlsSettingsBtn.setStyleSheet(u"QPushButton{\n"
"      color: rgb(255, 255, 255);\n"
"      border:none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"      background-color: rgb(153, 178, 208);\n"
"}")
        icon4 = QIcon()
        icon4.addFile(u"Images/Icons/controllerSettingsIcon.png", QSize(), QIcon.Normal, QIcon.Off)
        self.controlsSettingsBtn.setIcon(icon4)
        self.controlsSettingsBtn.setIconSize(QSize(50, 50))
        self.gameSelectionBtn = QPushButton(self.page_2)
        self.gameSelectionBtn.setObjectName(u"gameSelectionBtn")
        self.gameSelectionBtn.setGeometry(QRect(0, 0, 231, 71))
        self.gameSelectionBtn.setFont(font4)
        self.gameSelectionBtn.setCursor(QCursor(Qt.PointingHandCursor))
        self.gameSelectionBtn.setStyleSheet(u"background-color: rgb(99, 122, 155);\n"
"color: rgb(255, 255, 255);\n"
"border:none;")
        icon5 = QIcon()
        icon5.addFile(u"Images/Icons/ControllerIcon.png", QSize(), QIcon.Normal, QIcon.Off)
        self.gameSelectionBtn.setIcon(icon5)
        self.gameSelectionBtn.setIconSize(QSize(80, 80))
        self.profileSettingsBtn = QPushButton(self.page_2)
        self.profileSettingsBtn.setObjectName(u"profileSettingsBtn")
        self.profileSettingsBtn.setGeometry(QRect(0, 142, 231, 71))
        self.profileSettingsBtn.setFont(font4)
        self.profileSettingsBtn.setCursor(QCursor(Qt.PointingHandCursor))
        self.profileSettingsBtn.setLayoutDirection(Qt.LeftToRight)
        self.profileSettingsBtn.setStyleSheet(u"QPushButton{\n"
"      color: rgb(255, 255, 255);\n"
"      border:none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"      background-color: rgb(153, 178, 208);\n"
"}")
        icon6 = QIcon()
        icon6.addFile(u"Images/Icons/account-settings-11-128.png", QSize(), QIcon.Normal, QIcon.Off)
        self.profileSettingsBtn.setIcon(icon6)
        self.profileSettingsBtn.setIconSize(QSize(50, 55))
        self.label = QLabel(self.page_2)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(0, 0, 231, 731))
        self.label.setPixmap(QPixmap(u"Images/Backdrops/SidePanel.png"))
        self.label.setScaledContents(True)
        self.stackedWidget_2 = QStackedWidget(self.page_2)
        self.stackedWidget_2.setObjectName(u"stackedWidget_2")
        self.stackedWidget_2.setGeometry(QRect(229, -1, 791, 771))
        self.page_3 = QWidget()
        self.page_3.setObjectName(u"page_3")
        self.label_38 = QLabel(self.page_3)
        self.label_38.setObjectName(u"label_38")
        self.label_38.setGeometry(QRect(0, 0, 801, 111))
        self.label_38.setPixmap(QPixmap(u"Images/Backdrops/HeaderBackGround.png"))
        self.label_38.setScaledContents(True)
        self.label_39 = QLabel(self.page_3)
        self.label_39.setObjectName(u"label_39")
        self.label_39.setGeometry(QRect(20, 20, 251, 51))
        font5 = QFont()
        font5.setFamilies([u"Eras Bold ITC"])
        font5.setPointSize(22)
        self.label_39.setFont(font5)
        self.label_39.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.label_40 = QLabel(self.page_3)
        self.label_40.setObjectName(u"label_40")
        self.label_40.setGeometry(QRect(20, 50, 251, 51))
        font6 = QFont()
        font6.setFamilies([u"Eras Demi ITC"])
        font6.setPointSize(14)
        self.label_40.setFont(font6)
        self.label_40.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.label_41 = QLabel(self.page_3)
        self.label_41.setObjectName(u"label_41")
        self.label_41.setGeometry(QRect(60, 310, 101, 51))
        font7 = QFont()
        font7.setFamilies([u"Eras Demi ITC"])
        font7.setPointSize(11)
        self.label_41.setFont(font7)
        self.label_41.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.trackmaniaBtn = QPushButton(self.page_3)
        self.trackmaniaBtn.setObjectName(u"trackmaniaBtn")
        self.trackmaniaBtn.setGeometry(QRect(50, 200, 111, 121))
        self.trackmaniaBtn.setFont(font4)
        self.trackmaniaBtn.setCursor(QCursor(Qt.PointingHandCursor))
        self.trackmaniaBtn.setLayoutDirection(Qt.LeftToRight)
        self.trackmaniaBtn.setStyleSheet(u"QPushButton{\n"
"      background-color: none;\n"
"      border: none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"      \n"
"      background-color: rgba(255, 255, 255, 30);\n"
"}")
        icon7 = QIcon()
        icon7.addFile(u"Images/Icons/trachmaniaIcon.jpg", QSize(), QIcon.Normal, QIcon.Off)
        self.trackmaniaBtn.setIcon(icon7)
        self.trackmaniaBtn.setIconSize(QSize(150, 150))
        self.lineEdit_2 = QLineEdit(self.page_3)
        self.lineEdit_2.setObjectName(u"lineEdit_2")
        self.lineEdit_2.setGeometry(QRect(50, 140, 441, 31))
        font8 = QFont()
        font8.setFamilies([u"Eras Demi ITC"])
        font8.setPointSize(12)
        self.lineEdit_2.setFont(font8)
        self.lineEdit_2.setLayoutDirection(Qt.RightToLeft)
        self.lineEdit_2.setStyleSheet(u"background-color: rgb(121, 131, 145);\n"
"border-color: #ffffff;\n"
"color: rgb(255, 255, 255);")
        self.stackedWidget_2.addWidget(self.page_3)
        self.page_4 = QWidget()
        self.page_4.setObjectName(u"page_4")
        self.label_42 = QLabel(self.page_4)
        self.label_42.setObjectName(u"label_42")
        self.label_42.setGeometry(QRect(20, 30, 251, 51))
        self.label_42.setFont(font5)
        self.label_42.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.Profile1_3 = QPushButton(self.page_4)
        self.Profile1_3.setObjectName(u"Profile1_3")
        self.Profile1_3.setGeometry(QRect(50, 150, 151, 131))
        self.Profile1_3.setCursor(QCursor(Qt.ArrowCursor))
        self.Profile1_3.setStyleSheet(u"background-color: rgba(255, 255, 255, 0);")
        self.Profile1_3.setIcon(icon2)
        self.Profile1_3.setIconSize(QSize(100, 100))
        self.label_44 = QLabel(self.page_4)
        self.label_44.setObjectName(u"label_44")
        self.label_44.setGeometry(QRect(200, 220, 251, 51))
        font9 = QFont()
        font9.setFamilies([u"Eras Demi ITC"])
        font9.setPointSize(18)
        self.label_44.setFont(font9)
        self.label_44.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.removeProfileBtn_3 = QPushButton(self.page_4)
        self.removeProfileBtn_3.setObjectName(u"removeProfileBtn_3")
        self.removeProfileBtn_3.setGeometry(QRect(340, 280, 131, 50))
        font10 = QFont()
        font10.setFamilies([u"Eras Bold ITC"])
        self.removeProfileBtn_3.setFont(font10)
        self.removeProfileBtn_3.setCursor(QCursor(Qt.PointingHandCursor))
        self.removeProfileBtn_3.setStyleSheet(u"QPushButton {\n"
"    border-radius: 25px;\n"
"      \n"
"      color: rgb(255, 255, 255);\n"
"      background-color: rgb(212, 19, 19);\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"      background-color: rgb(255, 103, 65);\n"
"}")
        self.editProfileBtn_2 = QPushButton(self.page_4)
        self.editProfileBtn_2.setObjectName(u"editProfileBtn_2")
        self.editProfileBtn_2.setGeometry(QRect(290, 220, 21, 21))
        self.editProfileBtn_2.setCursor(QCursor(Qt.PointingHandCursor))
        self.editProfileBtn_2.setStyleSheet(u"QPushButton{\n"
"      background-color: rgba(255, 255, 255, 0);\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"      \n"
"      background-color: rgb(160, 173, 191);\n"
"}")
        icon8 = QIcon()
        icon8.addFile(u"Images/Icons/editicon.png", QSize(), QIcon.Normal, QIcon.Off)
        self.editProfileBtn_2.setIcon(icon8)
        self.editProfileBtn_2.setIconSize(QSize(25, 25))
        self.label_45 = QLabel(self.page_4)
        self.label_45.setObjectName(u"label_45")
        self.label_45.setGeometry(QRect(0, 0, 801, 111))
        self.label_45.setPixmap(QPixmap(u"Images/Backdrops/HeaderBackGround.png"))
        self.label_45.setScaledContents(True)
        self.stackedWidget_2.addWidget(self.page_4)
        self.Profile1_3.raise_()
        self.label_44.raise_()
        self.removeProfileBtn_3.raise_()
        self.editProfileBtn_2.raise_()
        self.label_45.raise_()
        self.label_42.raise_()
        self.page_5 = QWidget()
        self.page_5.setObjectName(u"page_5")
        self.removeProfileBtn_4 = QPushButton(self.page_5)
        self.removeProfileBtn_4.setObjectName(u"removeProfileBtn_4")
        self.removeProfileBtn_4.setGeometry(QRect(320, 580, 131, 50))
        self.removeProfileBtn_4.setFont(font10)
        self.removeProfileBtn_4.setCursor(QCursor(Qt.PointingHandCursor))
        self.removeProfileBtn_4.setStyleSheet(u"QPushButton {\n"
"    border-radius: 25px;\n"
"      \n"
"      color: rgb(255, 255, 255);\n"
"      \n"
"      background-color: rgb(122, 147, 183);\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"      \n"
"      background-color: rgb(145, 176, 218);\n"
"}")
        self.label_25 = QLabel(self.page_5)
        self.label_25.setObjectName(u"label_25")
        self.label_25.setGeometry(QRect(30, 130, 251, 51))
        font11 = QFont()
        font11.setFamilies([u"Eras Bold ITC"])
        font11.setPointSize(18)
        self.label_25.setFont(font11)
        self.label_25.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.label_43 = QLabel(self.page_5)
        self.label_43.setObjectName(u"label_43")
        self.label_43.setGeometry(QRect(20, 30, 251, 51))
        self.label_43.setFont(font5)
        self.label_43.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.CameraSelection = QComboBox(self.page_5)
        self.CameraSelection.setObjectName(u"CameraSelection")
        self.CameraSelection.setGeometry(QRect(30, 200, 431, 31))
        self.CameraSelection.setFont(font8)
        self.CameraSelection.setStyleSheet(u"background-color: rgb(144, 155, 172);\n"
"color: rgb(255, 255, 255);")
        self.label_46 = QLabel(self.page_5)
        self.label_46.setObjectName(u"label_46")
        self.label_46.setGeometry(QRect(30, 152, 251, 51))
        self.label_46.setFont(font2)
        self.label_46.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.label_47 = QLabel(self.page_5)
        self.label_47.setObjectName(u"label_47")
        self.label_47.setGeometry(QRect(0, 0, 801, 111))
        self.label_47.setPixmap(QPixmap(u"Images/Backdrops/HeaderBackGround.png"))
        self.label_47.setScaledContents(True)
        self.CameraPreview = QLabel(self.page_5)
        self.CameraPreview.setObjectName(u"CameraPreview")
        self.CameraPreview.setGeometry(QRect(160, 300, 241, 161))
        self.stackedWidget_2.addWidget(self.page_5)
        self.label_47.raise_()
        self.removeProfileBtn_4.raise_()
        self.label_25.raise_()
        self.label_43.raise_()
        self.CameraSelection.raise_()
        self.label_46.raise_()
        self.CameraPreview.raise_()
        self.page_6 = QWidget()
        self.page_6.setObjectName(u"page_6")
        self.label_32 = QLabel(self.page_6)
        self.label_32.setObjectName(u"label_32")
        self.label_32.setGeometry(QRect(0, 0, 801, 111))
        self.label_32.setPixmap(QPixmap(u"Images/Backdrops/HeaderBackGround.png"))
        self.label_32.setScaledContents(True)
        self.label_48 = QLabel(self.page_6)
        self.label_48.setObjectName(u"label_48")
        self.label_48.setGeometry(QRect(40, 118, 151, 28))
        self.label_48.setStyleSheet(u"background-color: rgb(255, 255, 255);")
        self.controllerBtn_4 = QPushButton(self.page_6)
        self.controllerBtn_4.setObjectName(u"controllerBtn_4")
        self.controllerBtn_4.setGeometry(QRect(115, 120, 75, 24))
        self.controllerBtn_4.setFont(font10)
        self.controllerBtn_4.setCursor(QCursor(Qt.PointingHandCursor))
        self.controllerBtn_4.setStyleSheet(u"color: rgb(96, 100, 106);\n"
"background-color: rgb(255, 255, 255);\n"
"border:none;")
        self.keyboardBtn_4 = QPushButton(self.page_6)
        self.keyboardBtn_4.setObjectName(u"keyboardBtn_4")
        self.keyboardBtn_4.setGeometry(QRect(41, 120, 75, 24))
        self.keyboardBtn_4.setFont(font10)
        self.keyboardBtn_4.setCursor(QCursor(Qt.PointingHandCursor))
        self.keyboardBtn_4.setStyleSheet(u"color: rgb(255, 255, 255);\n"
"background-color: rgb(122, 147, 183);\n"
"border:none;")
        self.label_50 = QLabel(self.page_6)
        self.label_50.setObjectName(u"label_50")
        self.label_50.setGeometry(QRect(20, 30, 251, 51))
        self.label_50.setFont(font5)
        self.label_50.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.Key_A = QLabel(self.page_6)
        self.Key_A.setObjectName(u"Key_A")
        self.Key_A.setGeometry(QRect(-40, 220, 300, 220))
        self.Key_A.setPixmap(QPixmap(u"Images/Icons/A.PNG"))
        self.Key_A.setScaledContents(True)
        self.Key_D = QLabel(self.page_6)
        self.Key_D.setObjectName(u"Key_D")
        self.Key_D.setGeometry(QRect(-40, 440, 300, 220))
        self.Key_D.setPixmap(QPixmap(u"Images/Icons/D.PNG"))
        self.Key_D.setScaledContents(True)
        self.Key_S = QLabel(self.page_6)
        self.Key_S.setObjectName(u"Key_S")
        self.Key_S.setGeometry(QRect(-40, 330, 300, 220))
        self.Key_S.setPixmap(QPixmap(u"Images/Icons/S.PNG"))
        self.Key_S.setScaledContents(True)
        self.Key_W = QLabel(self.page_6)
        self.Key_W.setObjectName(u"Key_W")
        self.Key_W.setGeometry(QRect(-40, 110, 300, 220))
        self.Key_W.setPixmap(QPixmap(u"Images/Icons/W.PNG"))
        self.Key_W.setScaledContents(True)
        self.imgLabel1 = QLabel(self.page_6)
        self.imgLabel1.setObjectName(u"imgLabel1")
        self.imgLabel1.setGeometry(QRect(130, 140, 200, 150))
        self.imgLabel1.setPixmap(QPixmap(u"Images/Icons/Click.PNG"))
        self.imgLabel1.setScaledContents(True)
        self.imgLabel2 = QLabel(self.page_6)
        self.imgLabel2.setObjectName(u"imgLabel2")
        self.imgLabel2.setGeometry(QRect(130, 250, 200, 150))
        self.imgLabel2.setPixmap(QPixmap(u"Images/Icons/Cursor.PNG"))
        self.imgLabel2.setScaledContents(True)
        self.imgLabel3 = QLabel(self.page_6)
        self.imgLabel3.setObjectName(u"imgLabel3")
        self.imgLabel3.setGeometry(QRect(140, 360, 200, 150))
        self.imgLabel3.setPixmap(QPixmap(u"Images/Icons/Fist.PNG"))
        self.imgLabel3.setScaledContents(True)
        self.imgLabel4 = QLabel(self.page_6)
        self.imgLabel4.setObjectName(u"imgLabel4")
        self.imgLabel4.setGeometry(QRect(130, 480, 200, 150))
        self.imgLabel4.setPixmap(QPixmap(u"Images/Icons/Flat.PNG"))
        self.imgLabel4.setScaledContents(True)
        self.Key_G = QLabel(self.page_6)
        self.Key_G.setObjectName(u"Key_G")
        self.Key_G.setGeometry(QRect(-40, 550, 300, 220))
        self.Key_G.setPixmap(QPixmap(u"Images/Icons/G.png"))
        self.Key_G.setScaledContents(True)
        self.imgLabel5 = QLabel(self.page_6)
        self.imgLabel5.setObjectName(u"imgLabel5")
        self.imgLabel5.setGeometry(QRect(130, 580, 200, 150))
        self.imgLabel5.setPixmap(QPixmap(u"Images/Icons/Gun.PNG"))
        self.imgLabel5.setScaledContents(True)
        self.Key_I = QLabel(self.page_6)
        self.Key_I.setObjectName(u"Key_I")
        self.Key_I.setGeometry(QRect(300, 110, 300, 220))
        self.Key_I.setPixmap(QPixmap(u"Images/Icons/I.png"))
        self.Key_I.setScaledContents(True)
        self.Key_O = QLabel(self.page_6)
        self.Key_O.setObjectName(u"Key_O")
        self.Key_O.setGeometry(QRect(300, 220, 300, 220))
        self.Key_O.setPixmap(QPixmap(u"Images/Icons/O.png"))
        self.Key_O.setScaledContents(True)
        self.Key_P = QLabel(self.page_6)
        self.Key_P.setObjectName(u"Key_P")
        self.Key_P.setGeometry(QRect(300, 330, 300, 220))
        self.Key_P.setPixmap(QPixmap(u"Images/Icons/P.png"))
        self.Key_P.setScaledContents(True)
        self.Key_U = QLabel(self.page_6)
        self.Key_U.setObjectName(u"Key_U")
        self.Key_U.setGeometry(QRect(300, 440, 300, 220))
        self.Key_U.setPixmap(QPixmap(u"Images/Icons/U.png"))
        self.Key_U.setScaledContents(True)
        self.imgLabel6 = QLabel(self.page_6)
        self.imgLabel6.setObjectName(u"imgLabel6")
        self.imgLabel6.setGeometry(QRect(490, 140, 200, 150))
        self.imgLabel6.setPixmap(QPixmap(u"Images/Icons/Inward.PNG"))
        self.imgLabel6.setScaledContents(True)
        self.imgLabel7 = QLabel(self.page_6)
        self.imgLabel7.setObjectName(u"imgLabel7")
        self.imgLabel7.setGeometry(QRect(490, 240, 200, 150))
        self.imgLabel7.setPixmap(QPixmap(u"Images/Icons/FacingAway.PNG"))
        self.imgLabel7.setScaledContents(True)
        self.imgLabel8 = QLabel(self.page_6)
        self.imgLabel8.setObjectName(u"imgLabel8")
        self.imgLabel8.setGeometry(QRect(490, 360, 200, 150))
        self.imgLabel8.setPixmap(QPixmap(u"Images/Icons/FacingTowards.PNG"))
        self.imgLabel8.setScaledContents(True)
        self.imgLabel9 = QLabel(self.page_6)
        self.imgLabel9.setObjectName(u"imgLabel9")
        self.imgLabel9.setGeometry(QRect(490, 470, 200, 150))
        self.imgLabel9.setPixmap(QPixmap(u"Images/Icons/Outwards.PNG"))
        self.imgLabel9.setScaledContents(True)
        self.saveChanges = QPushButton(self.page_6)
        self.saveChanges.setObjectName(u"saveChanges")
        self.saveChanges.setGeometry(QRect(410, 620, 231, 71))
        self.saveChanges.setFont(font4)
        self.saveChanges.setCursor(QCursor(Qt.PointingHandCursor))
        self.saveChanges.setLayoutDirection(Qt.LeftToRight)
        self.saveChanges.setStyleSheet(u"QPushButton{\n"
"      color: rgb(255, 255, 255);\n"
"      border: 1px solid white;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"      background-color: rgb(153, 178, 208);\n"
"}")
        self.saveChanges.setIconSize(QSize(60, 60))
        self.stackedWidget_2.addWidget(self.page_6)
        self.page_7 = QWidget()
        self.page_7.setObjectName(u"page_7")
        self.keyboardBtn_5 = QPushButton(self.page_7)
        self.keyboardBtn_5.setObjectName(u"keyboardBtn_5")
        self.keyboardBtn_5.setGeometry(QRect(41, 120, 75, 24))
        self.keyboardBtn_5.setFont(font10)
        self.keyboardBtn_5.setCursor(QCursor(Qt.PointingHandCursor))
        self.keyboardBtn_5.setStyleSheet(u"color: rgb(96, 100, 106);\n"
"background-color: rgb(255, 255, 255);\n"
"border:none;")
        self.label_35 = QLabel(self.page_7)
        self.label_35.setObjectName(u"label_35")
        self.label_35.setGeometry(QRect(40, 118, 151, 28))
        self.label_35.setStyleSheet(u"background-color: rgb(255, 255, 255);")
        self.controllerBtn_5 = QPushButton(self.page_7)
        self.controllerBtn_5.setObjectName(u"controllerBtn_5")
        self.controllerBtn_5.setGeometry(QRect(115, 120, 75, 24))
        self.controllerBtn_5.setFont(font10)
        self.controllerBtn_5.setCursor(QCursor(Qt.PointingHandCursor))
        self.controllerBtn_5.setStyleSheet(u"color: rgb(255, 255, 255);\n"
"background-color: rgb(122, 147, 183);\n"
"border:none;")
        self.label_51 = QLabel(self.page_7)
        self.label_51.setObjectName(u"label_51")
        self.label_51.setGeometry(QRect(0, 0, 801, 111))
        self.label_51.setPixmap(QPixmap(u"Images/Backdrops/HeaderBackGround.png"))
        self.label_51.setScaledContents(True)
        self.label_52 = QLabel(self.page_7)
        self.label_52.setObjectName(u"label_52")
        self.label_52.setGeometry(QRect(80, 191, 571, 491))
        self.label_52.setPixmap(QPixmap(u"Images/Icons/Controller Controls.png"))
        self.label_52.setScaledContents(True)
        self.label_53 = QLabel(self.page_7)
        self.label_53.setObjectName(u"label_53")
        self.label_53.setGeometry(QRect(20, 30, 251, 51))
        self.label_53.setFont(font5)
        self.label_53.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.stackedWidget_2.addWidget(self.page_7)
        self.label_35.raise_()
        self.controllerBtn_5.raise_()
        self.label_51.raise_()
        self.label_52.raise_()
        self.label_53.raise_()
        self.keyboardBtn_5.raise_()
        self.page_10 = QWidget()
        self.page_10.setObjectName(u"page_10")
        self.label_33 = QLabel(self.page_10)
        self.label_33.setObjectName(u"label_33")
        self.label_33.setGeometry(QRect(0, 0, 801, 111))
        self.label_33.setPixmap(QPixmap(u"Images/Backdrops/HeaderBackGround.png"))
        self.label_33.setScaledContents(True)
        self.label_59 = QLabel(self.page_10)
        self.label_59.setObjectName(u"label_59")
        self.label_59.setGeometry(QRect(60, 161, 361, 561))
        self.label_59.setPixmap(QPixmap(u"Images/Icons/Keyboard controls.png"))
        self.label_59.setScaledContents(True)
        self.label_60 = QLabel(self.page_10)
        self.label_60.setObjectName(u"label_60")
        self.label_60.setGeometry(QRect(20, 30, 251, 51))
        self.label_60.setFont(font5)
        self.label_60.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.label_11 = QLabel(self.page_10)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setGeometry(QRect(410, 150, 361, 531))
        self.label_11.setPixmap(QPixmap(u"Images/Icons/Component 10 \u2013 2.png"))
        self.label_11.setScaledContents(True)
        self.stackedWidget_2.addWidget(self.page_10)
        self.page_11 = QWidget()
        self.page_11.setObjectName(u"page_11")
        self.keyboardBtn_6 = QPushButton(self.page_11)
        self.keyboardBtn_6.setObjectName(u"keyboardBtn_6")
        self.keyboardBtn_6.setGeometry(QRect(41, 120, 75, 24))
        self.keyboardBtn_6.setFont(font10)
        self.keyboardBtn_6.setCursor(QCursor(Qt.PointingHandCursor))
        self.keyboardBtn_6.setStyleSheet(u"color: rgb(255, 255, 255);\n"
"background-color: rgb(122, 147, 183);\n"
"border:none;")
        self.controllerBtn_6 = QPushButton(self.page_11)
        self.controllerBtn_6.setObjectName(u"controllerBtn_6")
        self.controllerBtn_6.setGeometry(QRect(115, 120, 75, 24))
        self.controllerBtn_6.setFont(font10)
        self.controllerBtn_6.setCursor(QCursor(Qt.PointingHandCursor))
        self.controllerBtn_6.setStyleSheet(u"color: rgb(96, 100, 106);\n"
"background-color: rgb(255, 255, 255);\n"
"border:none;")
        self.label_61 = QLabel(self.page_11)
        self.label_61.setObjectName(u"label_61")
        self.label_61.setGeometry(QRect(40, 118, 151, 28))
        self.label_61.setStyleSheet(u"background-color: rgb(255, 255, 255);")
        self.label_12 = QLabel(self.page_11)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setGeometry(QRect(340, 140, 361, 531))
        self.label_12.setPixmap(QPixmap(u"Images/Icons/Component 10 \u2013 1.png"))
        self.label_12.setScaledContents(True)
        self.label_34 = QLabel(self.page_11)
        self.label_34.setObjectName(u"label_34")
        self.label_34.setGeometry(QRect(0, 0, 801, 111))
        self.label_34.setPixmap(QPixmap(u"Images/Backdrops/HeaderBackGround.png"))
        self.label_34.setScaledContents(True)
        self.label_62 = QLabel(self.page_11)
        self.label_62.setObjectName(u"label_62")
        self.label_62.setGeometry(QRect(60, 161, 361, 561))
        self.label_62.setPixmap(QPixmap(u"Images/Icons/Keyboard controls.png"))
        self.label_62.setScaledContents(True)
        self.label_63 = QLabel(self.page_11)
        self.label_63.setObjectName(u"label_63")
        self.label_63.setGeometry(QRect(20, 30, 251, 51))
        self.label_63.setFont(font5)
        self.label_63.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.pushButton_2 = QPushButton(self.page_11)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setGeometry(QRect(420, 610, 61, 31))
        self.pushButton_2.setStyleSheet(u"QPushButton {\n"
"background: rgba(200, 54, 54, 0);\n"
"      \n"
"\n"
"\n"
"}\n"
"\n"
"\n"
"\n"
"\n"
"")
        self.stackedWidget_2.addWidget(self.page_11)
        self.label_61.raise_()
        self.keyboardBtn_6.raise_()
        self.controllerBtn_6.raise_()
        self.label_34.raise_()
        self.label_62.raise_()
        self.label_63.raise_()
        self.label_12.raise_()
        self.pushButton_2.raise_()
        self.BtnTutorial = QPushButton(self.page_2)
        self.BtnTutorial.setObjectName(u"BtnTutorial")
        self.BtnTutorial.setGeometry(QRect(30, 590, 141, 50))
        font12 = QFont()
        font12.setFamilies([u"Eras Bold ITC"])
        font12.setPointSize(11)
        font12.setUnderline(False)
        self.BtnTutorial.setFont(font12)
        self.BtnTutorial.setCursor(QCursor(Qt.PointingHandCursor))
        self.BtnTutorial.setStyleSheet(u"QPushButton {\n"
"    border-radius: 25px;\n"
"      \n"
"      color: rgb(255, 255, 255);\n"
"      \n"
"      background-color: rgb(122, 147, 183);\n"
"\n"
"      border: 1px solid white;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"      \n"
"      background-color: rgb(145, 176, 218);\n"
"}")
        icon9 = QIcon()
        icon9.addFile(u"Images/Icons/gametutorial.png", QSize(), QIcon.Normal, QIcon.Off)
        self.BtnTutorial.setIcon(icon9)
        self.BtnTutorial.setIconSize(QSize(60, 60))
        self.stackedWidget.addWidget(self.page_2)
        self.label.raise_()
        self.logoutBtn.raise_()
        self.cameraSettingsBtn.raise_()
        self.controlsSettingsBtn.raise_()
        self.gameSelectionBtn.raise_()
        self.profileSettingsBtn.raise_()
        self.stackedWidget_2.raise_()
        self.BtnTutorial.raise_()
        self.page_8 = QWidget()
        self.page_8.setObjectName(u"page_8")
        self.controllerBtn = QPushButton(self.page_8)
        self.controllerBtn.setObjectName(u"controllerBtn")
        self.controllerBtn.setGeometry(QRect(345, 122, 75, 24))
        self.controllerBtn.setFont(font10)
        self.controllerBtn.setCursor(QCursor(Qt.PointingHandCursor))
        self.controllerBtn.setStyleSheet(u"color: rgb(96, 100, 106);\n"
"background-color: rgb(255, 255, 255);\n"
"border:none;")
        self.keyboardBtn = QPushButton(self.page_8)
        self.keyboardBtn.setObjectName(u"keyboardBtn")
        self.keyboardBtn.setGeometry(QRect(271, 122, 75, 24))
        self.keyboardBtn.setFont(font10)
        self.keyboardBtn.setCursor(QCursor(Qt.PointingHandCursor))
        self.keyboardBtn.setStyleSheet(u"color: rgb(255, 255, 255);\n"
"background-color: rgb(122, 147, 183);\n"
"border:none;")
        self.label_13 = QLabel(self.page_8)
        self.label_13.setObjectName(u"label_13")
        self.label_13.setGeometry(QRect(10, 20, 71, 91))
        self.label_13.setPixmap(QPixmap(u"Images/Icons/trachmaniaIcon.jpg"))
        self.label_13.setScaledContents(True)
        self.controlsSettingsBtn_2 = QPushButton(self.page_8)
        self.controlsSettingsBtn_2.setObjectName(u"controlsSettingsBtn_2")
        self.controlsSettingsBtn_2.setGeometry(QRect(0, 120, 231, 71))
        self.controlsSettingsBtn_2.setFont(font4)
        self.controlsSettingsBtn_2.setCursor(QCursor(Qt.PointingHandCursor))
        self.controlsSettingsBtn_2.setStyleSheet(u"background-color: rgb(99, 122, 155);\n"
"color: rgb(255, 255, 255);\n"
"border:none;")
        self.controlsSettingsBtn_2.setIcon(icon4)
        self.controlsSettingsBtn_2.setIconSize(QSize(50, 50))
        self.label_14 = QLabel(self.page_8)
        self.label_14.setObjectName(u"label_14")
        self.label_14.setGeometry(QRect(230, 0, 801, 111))
        self.label_14.setPixmap(QPixmap(u"Images/Backdrops/HeaderBackGround.png"))
        self.label_14.setScaledContents(True)
        self.label_15 = QLabel(self.page_8)
        self.label_15.setObjectName(u"label_15")
        self.label_15.setGeometry(QRect(0, 0, 231, 731))
        self.label_15.setPixmap(QPixmap(u"Images/Backdrops/SidePanel.png"))
        self.label_15.setScaledContents(True)
        self.logoutBtn_2 = QPushButton(self.page_8)
        self.logoutBtn_2.setObjectName(u"logoutBtn_2")
        self.logoutBtn_2.setGeometry(QRect(10, 660, 191, 51))
        self.logoutBtn_2.setFont(font4)
        self.logoutBtn_2.setCursor(QCursor(Qt.PointingHandCursor))
        self.logoutBtn_2.setLayoutDirection(Qt.LeftToRight)
        self.logoutBtn_2.setStyleSheet(u"QPushButton{\n"
"      color: rgb(255, 255, 255);\n"
"      border:none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"      background-color: rgb(153, 178, 208);\n"
"}")
        icon10 = QIcon()
        icon10.addFile(u"Images/Icons/Logouticon.png", QSize(), QIcon.Normal, QIcon.Off)
        self.logoutBtn_2.setIcon(icon10)
        self.logoutBtn_2.setIconSize(QSize(50, 55))
        self.label_16 = QLabel(self.page_8)
        self.label_16.setObjectName(u"label_16")
        self.label_16.setGeometry(QRect(270, 120, 151, 28))
        self.label_16.setStyleSheet(u"background-color: rgb(255, 255, 255);")
        self.launchGameBtn = QPushButton(self.page_8)
        self.launchGameBtn.setObjectName(u"launchGameBtn")
        self.launchGameBtn.setGeometry(QRect(830, 40, 171, 50))
        font13 = QFont()
        font13.setFamilies([u"Eras Bold ITC"])
        font13.setPointSize(11)
        self.launchGameBtn.setFont(font13)
        self.launchGameBtn.setCursor(QCursor(Qt.PointingHandCursor))
        self.launchGameBtn.setStyleSheet(u"QPushButton {\n"
"    border-radius: 25px;\n"
"      \n"
"      color: rgb(255, 255, 255);\n"
"      \n"
"      background-color: rgb(122, 147, 183);\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"      \n"
"      background-color: rgb(145, 176, 218);\n"
"}")
        icon11 = QIcon()
        icon11.addFile(u"Images/Icons/playButton.png", QSize(), QIcon.Normal, QIcon.Off)
        self.launchGameBtn.setIcon(icon11)
        self.launchGameBtn.setIconSize(QSize(50, 50))
        self.label_17 = QLabel(self.page_8)
        self.label_17.setObjectName(u"label_17")
        self.label_17.setGeometry(QRect(90, 60, 131, 21))
        font14 = QFont()
        font14.setFamilies([u"Eras Bold ITC"])
        font14.setPointSize(16)
        self.label_17.setFont(font14)
        self.label_17.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.label_18 = QLabel(self.page_8)
        self.label_18.setObjectName(u"label_18")
        self.label_18.setGeometry(QRect(320, 160, 361, 561))
        self.label_18.setPixmap(QPixmap(u"Images/Icons/Keyboard controls.png"))
        self.label_18.setScaledContents(True)
        self.controlsSettingsBtn_3 = QPushButton(self.page_8)
        self.controlsSettingsBtn_3.setObjectName(u"controlsSettingsBtn_3")
        self.controlsSettingsBtn_3.setGeometry(QRect(90, 80, 91, 31))
        font15 = QFont()
        font15.setFamilies([u"Eras Medium ITC"])
        font15.setPointSize(10)
        self.controlsSettingsBtn_3.setFont(font15)
        self.controlsSettingsBtn_3.setCursor(QCursor(Qt.PointingHandCursor))
        self.controlsSettingsBtn_3.setStyleSheet(u"QPushButton{\n"
"      color: rgb(255, 255, 255);\n"
"      border:none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"      background-color: rgb(153, 178, 208);\n"
"}")
        self.controlsSettingsBtn_3.setIconSize(QSize(50, 50))
        self.tutorialBtn = QPushButton(self.page_8)
        self.tutorialBtn.setObjectName(u"tutorialBtn")
        self.tutorialBtn.setGeometry(QRect(30, 590, 141, 50))
        self.tutorialBtn.setFont(font12)
        self.tutorialBtn.setCursor(QCursor(Qt.PointingHandCursor))
        self.tutorialBtn.setStyleSheet(u"QPushButton {\n"
"    border-radius: 25px;\n"
"      \n"
"      color: rgb(255, 255, 255);\n"
"      \n"
"      background-color: rgb(122, 147, 183);\n"
"\n"
"      border: 1px solid white;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"      \n"
"      background-color: rgb(145, 176, 218);\n"
"}")
        self.tutorialBtn.setIcon(icon9)
        self.tutorialBtn.setIconSize(QSize(60, 60))
        self.stackedWidget.addWidget(self.page_8)
        self.label_14.raise_()
        self.label_15.raise_()
        self.label_16.raise_()
        self.controllerBtn.raise_()
        self.keyboardBtn.raise_()
        self.label_13.raise_()
        self.controlsSettingsBtn_2.raise_()
        self.logoutBtn_2.raise_()
        self.launchGameBtn.raise_()
        self.label_17.raise_()
        self.label_18.raise_()
        self.controlsSettingsBtn_3.raise_()
        self.tutorialBtn.raise_()
        self.page_9 = QWidget()
        self.page_9.setObjectName(u"page_9")
        self.lineEdit_3 = QLineEdit(self.page_9)
        self.lineEdit_3.setObjectName(u"lineEdit_3")
        self.lineEdit_3.setGeometry(QRect(70, 480, 441, 31))
        self.lineEdit_3.setFont(font8)
        self.lineEdit_3.setLayoutDirection(Qt.RightToLeft)
        self.lineEdit_3.setStyleSheet(u"background-color: rgb(121, 131, 145);\n"
"border-color: #ffffff;\n"
"color: rgb(255, 255, 255);")
        self.label_55 = QLabel(self.page_9)
        self.label_55.setObjectName(u"label_55")
        self.label_55.setGeometry(QRect(20, 20, 251, 51))
        self.label_55.setFont(font5)
        self.label_55.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.label_56 = QLabel(self.page_9)
        self.label_56.setObjectName(u"label_56")
        self.label_56.setGeometry(QRect(0, 0, 1031, 111))
        self.label_56.setPixmap(QPixmap(u"Images/Backdrops/HeaderBackGround.png"))
        self.label_56.setScaledContents(True)
        self.label_57 = QLabel(self.page_9)
        self.label_57.setObjectName(u"label_57")
        self.label_57.setGeometry(QRect(70, 420, 281, 51))
        self.label_57.setFont(font6)
        self.label_57.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.label_54 = QLabel(self.page_9)
        self.label_54.setObjectName(u"label_54")
        self.label_54.setGeometry(QRect(370, 40, 251, 51))
        self.label_54.setFont(font5)
        self.label_54.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.label_58 = QLabel(self.page_9)
        self.label_58.setObjectName(u"label_58")
        self.label_58.setGeometry(QRect(70, 140, 251, 51))
        self.label_58.setFont(font6)
        self.label_58.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.Profile1_2 = QPushButton(self.page_9)
        self.Profile1_2.setObjectName(u"Profile1_2")
        self.Profile1_2.setGeometry(QRect(80, 200, 111, 121))
        self.Profile1_2.setCursor(QCursor(Qt.OpenHandCursor))
        self.Profile1_2.setStyleSheet(u"QPushButton{\n"
"      color: rgb(255, 255, 255);\n"
"      border:none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"      background-color: rgba(236, 236, 236, 30);\n"
"}")
        self.Profile1_2.setIcon(icon2)
        self.Profile1_2.setIconSize(QSize(100, 100))
        self.Profile1_4 = QPushButton(self.page_9)
        self.Profile1_4.setObjectName(u"Profile1_4")
        self.Profile1_4.setGeometry(QRect(230, 200, 111, 121))
        self.Profile1_4.setCursor(QCursor(Qt.OpenHandCursor))
        self.Profile1_4.setStyleSheet(u"QPushButton{\n"
"      color: rgb(255, 255, 255);\n"
"      border:none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"      background-color: rgba(236, 236, 236, 30);\n"
"}")
        self.Profile1_4.setIcon(icon2)
        self.Profile1_4.setIconSize(QSize(100, 100))
        self.Profile1_5 = QPushButton(self.page_9)
        self.Profile1_5.setObjectName(u"Profile1_5")
        self.Profile1_5.setGeometry(QRect(390, 200, 111, 121))
        self.Profile1_5.setCursor(QCursor(Qt.OpenHandCursor))
        self.Profile1_5.setStyleSheet(u"QPushButton{\n"
"      color: rgb(255, 255, 255);\n"
"      border:none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"      background-color: rgba(236, 236, 236, 30);\n"
"}")
        self.Profile1_5.setIcon(icon2)
        self.Profile1_5.setIconSize(QSize(100, 100))
        self.Profile1_6 = QPushButton(self.page_9)
        self.Profile1_6.setObjectName(u"Profile1_6")
        self.Profile1_6.setGeometry(QRect(550, 200, 111, 121))
        self.Profile1_6.setCursor(QCursor(Qt.OpenHandCursor))
        self.Profile1_6.setStyleSheet(u"QPushButton{\n"
"      color: rgb(255, 255, 255);\n"
"      border:none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"      background-color: rgba(236, 236, 236, 30);\n"
"}")
        self.Profile1_6.setIcon(icon2)
        self.Profile1_6.setIconSize(QSize(100, 100))
        self.Back = QPushButton(self.page_9)
        self.Back.setObjectName(u"Back")
        self.Back.setGeometry(QRect(510, 590, 141, 50))
        self.Back.setFont(font12)
        self.Back.setCursor(QCursor(Qt.PointingHandCursor))
        self.Back.setStyleSheet(u"QPushButton {\n"
"    border-radius: 25px;\n"
"      \n"
"      color: rgb(255, 255, 255);\n"
"      \n"
"      background-color: rgb(122, 147, 183);\n"
"\n"
"      border: 1px solid white;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"      \n"
"      background-color: rgb(145, 176, 218);\n"
"}")
        self.Back.setIconSize(QSize(60, 60))
        self.Ok = QPushButton(self.page_9)
        self.Ok.setObjectName(u"Ok")
        self.Ok.setGeometry(QRect(680, 590, 141, 50))
        self.Ok.setFont(font12)
        self.Ok.setCursor(QCursor(Qt.PointingHandCursor))
        self.Ok.setStyleSheet(u"QPushButton {\n"
"    border-radius: 25px;\n"
"      \n"
"      color: rgb(255, 255, 255);\n"
"      \n"
"      background-color: rgb(122, 147, 183);\n"
"\n"
"      border: 1px solid white;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"      \n"
"      background-color: rgb(145, 176, 218);\n"
"}")
        self.Ok.setIconSize(QSize(60, 60))
        self.Profile1_7 = QPushButton(self.page_9)
        self.Profile1_7.setObjectName(u"Profile1_7")
        self.Profile1_7.setGeometry(QRect(710, 200, 111, 121))
        self.Profile1_7.setCursor(QCursor(Qt.OpenHandCursor))
        self.Profile1_7.setStyleSheet(u"QPushButton{\n"
"      color: rgb(255, 255, 255);\n"
"      border:none;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"      background-color: rgba(236, 236, 236, 30);\n"
"}")
        self.Profile1_7.setIcon(icon2)
        self.Profile1_7.setIconSize(QSize(100, 100))
        self.stackedWidget.addWidget(self.page_9)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1024, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.stackedWidget.setCurrentIndex(1)
        self.stackedWidget_2.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"GesturePlay", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Welcome", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Add a", None))
        self.AddProfileButton.setText("")
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Profile1", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Who's playing today?", None))
        self.label_8.setText("")
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"to", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Profile", None))
        self.Profile1.setText("")
        self.label_9.setText("")
        self.logoutBtn.setText(QCoreApplication.translate("MainWindow", u"Change Profile", None))
        self.cameraSettingsBtn.setText(QCoreApplication.translate("MainWindow", u" Camera Settings", None))
        self.controlsSettingsBtn.setText(QCoreApplication.translate("MainWindow", u"   Controls Settings", None))
        self.gameSelectionBtn.setText(QCoreApplication.translate("MainWindow", u"Game Selection", None))
        self.profileSettingsBtn.setText(QCoreApplication.translate("MainWindow", u" Profile Settings", None))
        self.label.setText("")
        self.label_38.setText("")
        self.label_39.setText(QCoreApplication.translate("MainWindow", u"Game Selection", None))
        self.label_40.setText(QCoreApplication.translate("MainWindow", u"Please Select a game", None))
        self.label_41.setText(QCoreApplication.translate("MainWindow", u"Trackmania", None))
        self.trackmaniaBtn.setText("")
        self.lineEdit_2.setText("")
        self.lineEdit_2.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Search for a game", None))
        self.label_42.setText(QCoreApplication.translate("MainWindow", u"Profile Settings", None))
        self.Profile1_3.setText("")
        self.label_44.setText(QCoreApplication.translate("MainWindow", u"Profile1", None))
        self.removeProfileBtn_3.setText(QCoreApplication.translate("MainWindow", u"Remove Profile", None))
        self.editProfileBtn_2.setText("")
        self.label_45.setText("")
        self.removeProfileBtn_4.setText(QCoreApplication.translate("MainWindow", u"Save Changes", None))
        self.label_25.setText(QCoreApplication.translate("MainWindow", u"Webcam", None))
        self.label_43.setText(QCoreApplication.translate("MainWindow", u"Camera Settings", None))
        self.CameraSelection.setPlaceholderText(QCoreApplication.translate("MainWindow", u"System Default", None))
        self.label_46.setText(QCoreApplication.translate("MainWindow", u"Capture Webcam", None))
        self.label_47.setText("")
        self.CameraPreview.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.label_32.setText("")
        self.label_48.setText("")
        self.controllerBtn_4.setText(QCoreApplication.translate("MainWindow", u"Controller", None))
        self.keyboardBtn_4.setText(QCoreApplication.translate("MainWindow", u"Keyboard", None))
        self.label_50.setText(QCoreApplication.translate("MainWindow", u"Controls Settings", None))
        self.Key_A.setText("")
        self.Key_D.setText("")
        self.Key_S.setText("")
        self.Key_W.setText("")
        self.imgLabel1.setText("")
        self.imgLabel2.setText("")
        self.imgLabel3.setText("")
        self.imgLabel4.setText("")
        self.Key_G.setText("")
        self.imgLabel5.setText("")
        self.Key_I.setText("")
        self.Key_O.setText("")
        self.Key_P.setText("")
        self.Key_U.setText("")
        self.imgLabel6.setText("")
        self.imgLabel7.setText("")
        self.imgLabel8.setText("")
        self.imgLabel9.setText("")
        self.saveChanges.setText(QCoreApplication.translate("MainWindow", u"Save Changes", None))
        self.keyboardBtn_5.setText(QCoreApplication.translate("MainWindow", u"Keyboard", None))
        self.label_35.setText("")
        self.controllerBtn_5.setText(QCoreApplication.translate("MainWindow", u"Controller", None))
        self.label_51.setText("")
        self.label_52.setText("")
        self.label_53.setText(QCoreApplication.translate("MainWindow", u"Controls Settings", None))
        self.label_33.setText("")
#if QT_CONFIG(whatsthis)
        self.label_59.setWhatsThis("")
#endif // QT_CONFIG(whatsthis)
        self.label_59.setText("")
        self.label_60.setText(QCoreApplication.translate("MainWindow", u"Tutorial", None))
        self.label_11.setText("")
        self.keyboardBtn_6.setText(QCoreApplication.translate("MainWindow", u"Keyboard", None))
        self.controllerBtn_6.setText(QCoreApplication.translate("MainWindow", u"Controller", None))
        self.label_61.setText("")
        self.label_12.setText("")
        self.label_34.setText("")
#if QT_CONFIG(whatsthis)
        self.label_62.setWhatsThis("")
#endif // QT_CONFIG(whatsthis)
        self.label_62.setText("")
        self.label_63.setText(QCoreApplication.translate("MainWindow", u"Controls Settings", None))
        self.pushButton_2.setText("")
        self.BtnTutorial.setText(QCoreApplication.translate("MainWindow", u"Tutorial", None))
        self.controllerBtn.setText(QCoreApplication.translate("MainWindow", u"Controller", None))
        self.keyboardBtn.setText(QCoreApplication.translate("MainWindow", u"Keyboard", None))
        self.label_13.setText("")
        self.controlsSettingsBtn_2.setText(QCoreApplication.translate("MainWindow", u"   Controls Settings", None))
        self.label_14.setText("")
        self.label_15.setText("")
        self.logoutBtn_2.setText(QCoreApplication.translate("MainWindow", u"Change Profile", None))
        self.label_16.setText("")
        self.launchGameBtn.setText(QCoreApplication.translate("MainWindow", u"Launch Game", None))
        self.label_17.setText(QCoreApplication.translate("MainWindow", u"Trackmania", None))
        self.label_18.setText("")
        self.controlsSettingsBtn_3.setText(QCoreApplication.translate("MainWindow", u"Change Game", None))
        self.tutorialBtn.setText(QCoreApplication.translate("MainWindow", u"Tutorial", None))
        self.lineEdit_3.setText(QCoreApplication.translate("MainWindow", u" Nickname of profile", None))
        self.lineEdit_3.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Search for a game", None))
        self.label_55.setText(QCoreApplication.translate("MainWindow", u"Game Selection", None))
        self.label_56.setText("")
        self.label_57.setText(QCoreApplication.translate("MainWindow", u"Enter a nickname for this user", None))
        self.label_54.setText(QCoreApplication.translate("MainWindow", u"Create A Profile ", None))
        self.label_58.setText(QCoreApplication.translate("MainWindow", u"Select an Icon", None))
        self.Profile1_2.setText("")
        self.Profile1_4.setText("")
        self.Profile1_5.setText("")
        self.Profile1_6.setText("")
        self.Back.setText(QCoreApplication.translate("MainWindow", u"Back", None))
        self.Ok.setText(QCoreApplication.translate("MainWindow", u"Ok", None))
        self.Profile1_7.setText("")
    # retranslateUi

