# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '\\AKC0SS-N086\RACE_Users\rick.towler\My Documents\AFSCGit\CamTrawl\CamtrawlBrowser\QImageViewer\ui\imageAdjustmentsDlg.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_imageAdjustmentsDlg(object):
    def setupUi(self, imageAdjustmentsDlg):
        imageAdjustmentsDlg.setObjectName(_fromUtf8("imageAdjustmentsDlg"))
        imageAdjustmentsDlg.resize(414, 660)
        imageAdjustmentsDlg.setMinimumSize(QtCore.QSize(316, 660))
        self.verticalLayout_2 = QtGui.QVBoxLayout(imageAdjustmentsDlg)
        self.verticalLayout_2.setMargin(5)
        self.verticalLayout_2.setSpacing(4)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.gbBrightnessContrast = QtGui.QGroupBox(imageAdjustmentsDlg)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.gbBrightnessContrast.setFont(font)
        self.gbBrightnessContrast.setCheckable(True)
        self.gbBrightnessContrast.setChecked(False)
        self.gbBrightnessContrast.setObjectName(_fromUtf8("gbBrightnessContrast"))
        self.verticalLayout_4 = QtGui.QVBoxLayout(self.gbBrightnessContrast)
        self.verticalLayout_4.setMargin(5)
        self.verticalLayout_4.setSpacing(3)
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.label_8 = QtGui.QLabel(self.gbBrightnessContrast)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label_8.setFont(font)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.verticalLayout_4.addWidget(self.label_8)
        self.brightnessSlider = QtGui.QSlider(self.gbBrightnessContrast)
        self.brightnessSlider.setMinimum(-100)
        self.brightnessSlider.setMaximum(100)
        self.brightnessSlider.setOrientation(QtCore.Qt.Horizontal)
        self.brightnessSlider.setTickPosition(QtGui.QSlider.TicksAbove)
        self.brightnessSlider.setTickInterval(50)
        self.brightnessSlider.setObjectName(_fromUtf8("brightnessSlider"))
        self.verticalLayout_4.addWidget(self.brightnessSlider)
        self.bcAutomatic = QtGui.QRadioButton(self.gbBrightnessContrast)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.bcAutomatic.setFont(font)
        self.bcAutomatic.setChecked(False)
        self.bcAutomatic.setObjectName(_fromUtf8("bcAutomatic"))
        self.verticalLayout_4.addWidget(self.bcAutomatic)
        self.gbAutoBC = QtGui.QGroupBox(self.gbBrightnessContrast)
        self.gbAutoBC.setTitle(_fromUtf8(""))
        self.gbAutoBC.setObjectName(_fromUtf8("gbAutoBC"))
        self.formLayout_4 = QtGui.QFormLayout(self.gbAutoBC)
        self.formLayout_4.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout_4.setObjectName(_fromUtf8("formLayout_4"))
        self.label_10 = QtGui.QLabel(self.gbAutoBC)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.label_10.setFont(font)
        self.label_10.setObjectName(_fromUtf8("label_10"))
        self.formLayout_4.setWidget(0, QtGui.QFormLayout.LabelRole, self.label_10)
        self.bcClipLimit = QtGui.QSlider(self.gbAutoBC)
        self.bcClipLimit.setMinimum(5)
        self.bcClipLimit.setMaximum(50)
        self.bcClipLimit.setProperty("value", 30)
        self.bcClipLimit.setOrientation(QtCore.Qt.Horizontal)
        self.bcClipLimit.setTickPosition(QtGui.QSlider.TicksAbove)
        self.bcClipLimit.setTickInterval(5)
        self.bcClipLimit.setObjectName(_fromUtf8("bcClipLimit"))
        self.formLayout_4.setWidget(0, QtGui.QFormLayout.FieldRole, self.bcClipLimit)
        self.verticalLayout_4.addWidget(self.gbAutoBC)
        self.bcManual = QtGui.QRadioButton(self.gbBrightnessContrast)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.bcManual.setFont(font)
        self.bcManual.setChecked(True)
        self.bcManual.setObjectName(_fromUtf8("bcManual"))
        self.verticalLayout_4.addWidget(self.bcManual)
        self.gbManualBC = QtGui.QGroupBox(self.gbBrightnessContrast)
        self.gbManualBC.setTitle(_fromUtf8(""))
        self.gbManualBC.setObjectName(_fromUtf8("gbManualBC"))
        self.verticalLayout_5 = QtGui.QVBoxLayout(self.gbManualBC)
        self.verticalLayout_5.setObjectName(_fromUtf8("verticalLayout_5"))
        self.verticalLayout_6 = QtGui.QVBoxLayout()
        self.verticalLayout_6.setObjectName(_fromUtf8("verticalLayout_6"))
        self.label_7 = QtGui.QLabel(self.gbManualBC)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label_7.setFont(font)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.verticalLayout_6.addWidget(self.label_7)
        self.contrastSlider = QtGui.QSlider(self.gbManualBC)
        self.contrastSlider.setMinimum(-100)
        self.contrastSlider.setMaximum(100)
        self.contrastSlider.setProperty("value", 0)
        self.contrastSlider.setSliderPosition(0)
        self.contrastSlider.setOrientation(QtCore.Qt.Horizontal)
        self.contrastSlider.setTickPosition(QtGui.QSlider.TicksAbove)
        self.contrastSlider.setTickInterval(50)
        self.contrastSlider.setObjectName(_fromUtf8("contrastSlider"))
        self.verticalLayout_6.addWidget(self.contrastSlider)
        self.verticalLayout_5.addLayout(self.verticalLayout_6)
        self.verticalLayout_4.addWidget(self.gbManualBC)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.pbBCReset = QtGui.QPushButton(self.gbBrightnessContrast)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.pbBCReset.setFont(font)
        self.pbBCReset.setObjectName(_fromUtf8("pbBCReset"))
        self.horizontalLayout.addWidget(self.pbBCReset)
        self.verticalLayout_4.addLayout(self.horizontalLayout)
        self.verticalLayout_2.addWidget(self.gbBrightnessContrast)
        self.gbColorCorrection = QtGui.QGroupBox(imageAdjustmentsDlg)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.gbColorCorrection.setFont(font)
        self.gbColorCorrection.setCheckable(True)
        self.gbColorCorrection.setChecked(False)
        self.gbColorCorrection.setObjectName(_fromUtf8("gbColorCorrection"))
        self.verticalLayout_3 = QtGui.QVBoxLayout(self.gbColorCorrection)
        self.verticalLayout_3.setMargin(5)
        self.verticalLayout_3.setSpacing(3)
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.ccSimpleBalance = QtGui.QRadioButton(self.gbColorCorrection)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.ccSimpleBalance.setFont(font)
        self.ccSimpleBalance.setObjectName(_fromUtf8("ccSimpleBalance"))
        self.verticalLayout_3.addWidget(self.ccSimpleBalance)
        self.gbAutoCC = QtGui.QGroupBox(self.gbColorCorrection)
        self.gbAutoCC.setTitle(_fromUtf8(""))
        self.gbAutoCC.setObjectName(_fromUtf8("gbAutoCC"))
        self.formLayout_3 = QtGui.QFormLayout(self.gbAutoCC)
        self.formLayout_3.setObjectName(_fromUtf8("formLayout_3"))
        self.label_6 = QtGui.QLabel(self.gbAutoCC)
        self.label_6.setMinimumSize(QtCore.QSize(125, 0))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.label_6.setFont(font)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.formLayout_3.setWidget(0, QtGui.QFormLayout.LabelRole, self.label_6)
        self.ccSatLevel = QtGui.QSlider(self.gbAutoCC)
        self.ccSatLevel.setMinimum(1)
        self.ccSatLevel.setMaximum(16)
        self.ccSatLevel.setPageStep(2)
        self.ccSatLevel.setProperty("value", 8)
        self.ccSatLevel.setOrientation(QtCore.Qt.Horizontal)
        self.ccSatLevel.setTickPosition(QtGui.QSlider.TicksAbove)
        self.ccSatLevel.setTickInterval(2)
        self.ccSatLevel.setObjectName(_fromUtf8("ccSatLevel"))
        self.formLayout_3.setWidget(0, QtGui.QFormLayout.FieldRole, self.ccSatLevel)
        self.verticalLayout_3.addWidget(self.gbAutoCC)
        self.ccAdaptive = QtGui.QRadioButton(self.gbColorCorrection)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.ccAdaptive.setFont(font)
        self.ccAdaptive.setChecked(False)
        self.ccAdaptive.setObjectName(_fromUtf8("ccAdaptive"))
        self.verticalLayout_3.addWidget(self.ccAdaptive)
        self.gbAdaptiveCC = QtGui.QGroupBox(self.gbColorCorrection)
        self.gbAdaptiveCC.setTitle(_fromUtf8(""))
        self.gbAdaptiveCC.setObjectName(_fromUtf8("gbAdaptiveCC"))
        self.formLayout_2 = QtGui.QFormLayout(self.gbAdaptiveCC)
        self.formLayout_2.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout_2.setObjectName(_fromUtf8("formLayout_2"))
        self.label_5 = QtGui.QLabel(self.gbAdaptiveCC)
        self.label_5.setMinimumSize(QtCore.QSize(80, 0))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.label_5.setFont(font)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.formLayout_2.setWidget(0, QtGui.QFormLayout.LabelRole, self.label_5)
        self.ccClipLimit = QtGui.QSlider(self.gbAdaptiveCC)
        self.ccClipLimit.setMinimum(5)
        self.ccClipLimit.setMaximum(50)
        self.ccClipLimit.setProperty("value", 30)
        self.ccClipLimit.setOrientation(QtCore.Qt.Horizontal)
        self.ccClipLimit.setTickPosition(QtGui.QSlider.TicksAbove)
        self.ccClipLimit.setTickInterval(5)
        self.ccClipLimit.setObjectName(_fromUtf8("ccClipLimit"))
        self.formLayout_2.setWidget(0, QtGui.QFormLayout.FieldRole, self.ccClipLimit)
        self.verticalLayout_3.addWidget(self.gbAdaptiveCC)
        self.ccManual = QtGui.QRadioButton(self.gbColorCorrection)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.ccManual.setFont(font)
        self.ccManual.setChecked(True)
        self.ccManual.setAutoExclusive(True)
        self.ccManual.setObjectName(_fromUtf8("ccManual"))
        self.verticalLayout_3.addWidget(self.ccManual)
        self.gbManualCC = QtGui.QGroupBox(self.gbColorCorrection)
        self.gbManualCC.setTitle(_fromUtf8(""))
        self.gbManualCC.setObjectName(_fromUtf8("gbManualCC"))
        self.verticalLayout_7 = QtGui.QVBoxLayout(self.gbManualCC)
        self.verticalLayout_7.setObjectName(_fromUtf8("verticalLayout_7"))
        self.formLayout = QtGui.QFormLayout()
        self.formLayout.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout.setObjectName(_fromUtf8("formLayout"))
        self.label = QtGui.QLabel(self.gbManualCC)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setObjectName(_fromUtf8("label"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.label)
        self.redSlider = QtGui.QSlider(self.gbManualCC)
        self.redSlider.setMinimum(-50)
        self.redSlider.setMaximum(50)
        self.redSlider.setOrientation(QtCore.Qt.Horizontal)
        self.redSlider.setTickPosition(QtGui.QSlider.TicksAbove)
        self.redSlider.setTickInterval(10)
        self.redSlider.setObjectName(_fromUtf8("redSlider"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.FieldRole, self.redSlider)
        self.label_2 = QtGui.QLabel(self.gbManualCC)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label_2.setFont(font)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.LabelRole, self.label_2)
        self.greenSlider = QtGui.QSlider(self.gbManualCC)
        self.greenSlider.setMinimum(-50)
        self.greenSlider.setMaximum(50)
        self.greenSlider.setOrientation(QtCore.Qt.Horizontal)
        self.greenSlider.setTickPosition(QtGui.QSlider.TicksAbove)
        self.greenSlider.setTickInterval(10)
        self.greenSlider.setObjectName(_fromUtf8("greenSlider"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.FieldRole, self.greenSlider)
        self.label_3 = QtGui.QLabel(self.gbManualCC)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label_3.setFont(font)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.formLayout.setWidget(2, QtGui.QFormLayout.LabelRole, self.label_3)
        self.blueSlider = QtGui.QSlider(self.gbManualCC)
        self.blueSlider.setMinimum(-50)
        self.blueSlider.setMaximum(50)
        self.blueSlider.setOrientation(QtCore.Qt.Horizontal)
        self.blueSlider.setTickPosition(QtGui.QSlider.TicksAbove)
        self.blueSlider.setTickInterval(10)
        self.blueSlider.setObjectName(_fromUtf8("blueSlider"))
        self.formLayout.setWidget(2, QtGui.QFormLayout.FieldRole, self.blueSlider)
        self.verticalLayout_7.addLayout(self.formLayout)
        self.redSlider.raise_()
        self.verticalLayout_3.addWidget(self.gbManualCC)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.pbColorReset = QtGui.QPushButton(self.gbColorCorrection)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.pbColorReset.setFont(font)
        self.pbColorReset.setObjectName(_fromUtf8("pbColorReset"))
        self.horizontalLayout_3.addWidget(self.pbColorReset)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.verticalLayout_2.addWidget(self.gbColorCorrection)
        self.cbDenoise = QtGui.QCheckBox(imageAdjustmentsDlg)
        self.cbDenoise.setEnabled(True)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.cbDenoise.setFont(font)
        self.cbDenoise.setObjectName(_fromUtf8("cbDenoise"))
        self.verticalLayout_2.addWidget(self.cbDenoise)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        spacerItem2 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem2)
        self.pbApply = QtGui.QPushButton(imageAdjustmentsDlg)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pbApply.setFont(font)
        self.pbApply.setObjectName(_fromUtf8("pbApply"))
        self.horizontalLayout_2.addWidget(self.pbApply)
        self.pbCancel = QtGui.QPushButton(imageAdjustmentsDlg)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pbCancel.setFont(font)
        self.pbCancel.setObjectName(_fromUtf8("pbCancel"))
        self.horizontalLayout_2.addWidget(self.pbCancel)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)

        self.retranslateUi(imageAdjustmentsDlg)
        QtCore.QMetaObject.connectSlotsByName(imageAdjustmentsDlg)

    def retranslateUi(self, imageAdjustmentsDlg):
        imageAdjustmentsDlg.setWindowTitle(_translate("imageAdjustmentsDlg", "Image Adjustments:", None))
        self.gbBrightnessContrast.setTitle(_translate("imageAdjustmentsDlg", "Brightness and Contrast", None))
        self.label_8.setText(_translate("imageAdjustmentsDlg", "Brightness", None))
        self.bcAutomatic.setText(_translate("imageAdjustmentsDlg", "Automatic Contrast (CLAHE)", None))
        self.label_10.setText(_translate("imageAdjustmentsDlg", "Clip Limit", None))
        self.bcManual.setText(_translate("imageAdjustmentsDlg", "Manual Contrast", None))
        self.label_7.setText(_translate("imageAdjustmentsDlg", "Contrast", None))
        self.pbBCReset.setText(_translate("imageAdjustmentsDlg", "Reset", None))
        self.gbColorCorrection.setTitle(_translate("imageAdjustmentsDlg", "Color Correction", None))
        self.ccSimpleBalance.setText(_translate("imageAdjustmentsDlg", "Automatic - Auto Levels", None))
        self.label_6.setText(_translate("imageAdjustmentsDlg", "Saturation Level", None))
        self.ccAdaptive.setText(_translate("imageAdjustmentsDlg", "Automatic - Adaptive Equalization", None))
        self.label_5.setText(_translate("imageAdjustmentsDlg", "Clip Limit", None))
        self.ccManual.setText(_translate("imageAdjustmentsDlg", "Manual", None))
        self.label.setText(_translate("imageAdjustmentsDlg", "R", None))
        self.label_2.setText(_translate("imageAdjustmentsDlg", "G", None))
        self.label_3.setText(_translate("imageAdjustmentsDlg", "B", None))
        self.pbColorReset.setText(_translate("imageAdjustmentsDlg", "Reset", None))
        self.cbDenoise.setToolTip(_translate("imageAdjustmentsDlg", "<html><head/><body><p>and da funk</p></body></html>", None))
        self.cbDenoise.setText(_translate("imageAdjustmentsDlg", "Denoise (SLOW!)", None))
        self.pbApply.setText(_translate("imageAdjustmentsDlg", "Apply", None))
        self.pbCancel.setText(_translate("imageAdjustmentsDlg", "Cancel", None))

