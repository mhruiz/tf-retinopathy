#!/usr/bin/env python


#############################################################################
##
## Copyright (C) 2013 Riverbank Computing Limited.
## Copyright (C) 2010 Nokia Corporation and/or its subsidiary(-ies).
## All rights reserved.
##
## This file is part of the examples of PyQt.
##
## $QT_BEGIN_LICENSE:BSD$
## You may use this file under the terms of the BSD license as follows:
##
## "Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are
## met:
##   * Redistributions of source code must retain the above copyright
##     notice, this list of conditions and the following disclaimer.
##   * Redistributions in binary form must reproduce the above copyright
##     notice, this list of conditions and the following disclaimer in
##     the documentation and/or other materials provided with the
##     distribution.
##   * Neither the name of Nokia Corporation and its Subsidiary(-ies) nor
##     the names of its contributors may be used to endorse or promote
##     products derived from this software without specific prior written
##     permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
## "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
## LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
## A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
## OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
## SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
## LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
## DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
## THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
## (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
## $QT_END_LICENSE$
##
#############################################################################

from PyQt5 import QtCore
from PyQt5.QtCore import QDir, Qt
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap, QKeySequence
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QLabel,
        QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy, QShortcut)
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter

from pathlib import Path
import application
import os

APPLICATION_TITLE = 'Herramienta diagnóstico para RD'

class ImageViewer(QMainWindow):
    def __init__(self):
        super(ImageViewer, self).__init__()

        self.printer = QPrinter()
        self.scaleFactor = 0.0

        self.imageLabel = QLabel()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)

        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.setCentralWidget(self.scrollArea)

        self.createActions()
        self.createMenus()

        self.setWindowTitle(APPLICATION_TITLE)
        self.resize(750, 580)

        self.model = None

        self.image_file_formats = ['png', 'jpg', 'jpeg', 'bmp', 'tif']
        self.directory = ''
        self.images_in_dir = []
        self.current_image_index = 0
        self.current_filename = ''

        self.key_pressed = {x: False for x in [Qt.Key.Key_Control]}


    def previous_image(self):

        if self.images_in_dir:
            self.current_image_index = (self.current_image_index - 1 + len(self.images_in_dir)) % len(self.images_in_dir) 

            file_name = self.directory + '/' + self.images_in_dir[self.current_image_index]

            self.load_image(file_name)

    def next_image(self):

        if self.images_in_dir:
            self.current_image_index = (self.current_image_index + 1) % len(self.images_in_dir) 

            file_name = self.directory + '/' + self.images_in_dir[self.current_image_index]

            self.load_image(file_name)


    def load_image(self, fileName):
        qimage = QImage(fileName)
        if qimage.isNull():
            QMessageBox.information(self, APPLICATION_TITLE,
                    "No se pudo cargar %s." % fileName)
            return
        
        # Load and crop retina image
        image = application.prepare_image(fileName)

        # Convert to QImage object to show the same image that the CNN will receive
        # height, width, channel = image.shape
        # bytesPerLine = 3 * width
        # qimage = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)

        self.imageLabel.setPixmap(QPixmap.fromImage(qimage))
        self.scaleFactor = 1.0

        self.printAct.setEnabled(True)
        self.fitToWindowAct.setEnabled(True)
        self.updateActions()

        if not self.fitToWindowAct.isChecked():
            self.imageLabel.adjustSize()
        
        self.current_filename = fileName.split('/')[-1]

        window_title = APPLICATION_TITLE + ' - ' + self.current_filename

        self.setWindowTitle(window_title + ' - Cargando...')

        if self.model is None:
            self.model = application.load_model(application.model_config, application.model_weights)

        # image = application.prepare_image(fileName)

        if image is not None:
            prediction, output = application.predict_image(image, self.model)
            self.setWindowTitle(window_title + ' - Diagnóstico: ' + output)
        else:
            self.setWindowTitle(window_title + ' - No se ha podido detectar el disco óptico')
    

    def open(self):
        fileName, _ = QFileDialog.getOpenFileName(self, 
                                                  'Seleccionar imagen', 
                                                  QDir.currentPath(), 
                                                  ('Imágenes (' + ' '.join(['*.' + x for x in self.image_file_formats]) + ');; Todos los ficheros (*)'))
        if fileName:
            self.directory = str(Path(fileName).parent.absolute()).replace('\\','/')

            is_image = lambda x: x.split('.')[-1] in self.image_file_formats

            self.images_in_dir = list(filter(is_image, os.listdir(self.directory)))

            self.current_image_index = self.images_in_dir.index(fileName.split('/')[-1])

            self.load_image(fileName)

    def print_(self):
        dialog = QPrintDialog(self.printer, self)
        if dialog.exec_():
            painter = QPainter(self.printer)
            rect = painter.viewport()
            size = self.imageLabel.pixmap().size()
            size.scale(rect.size(), Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.imageLabel.pixmap().rect())
            painter.drawPixmap(0, 0, self.imageLabel.pixmap())

    def zoomIn(self):
        self.scaleImage(1.25)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.imageLabel.adjustSize()
        self.scaleFactor = 1.0

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()

        self.updateActions()

    def about(self):
        QMessageBox.about(self, "Acerca de " + APPLICATION_TITLE,

                '<p>La <b>' + APPLICATION_TITLE + '</b> ha sido desarrollada tomando como base una implementación de un '
                'visualizador de imágenes, <a href="https://github.com/baoboa/pyqt5/blob/master/examples/widgets/imageviewer.py">Image Viewer</a>, realizado en Python, haciendo uso del módulo <b>PyQt5</b>. '
                'Sobre ésta, se ha introducido un modelo de red neuronal convolucional entrenado '
                'para la detección de Retinopatía Diabética en imágenes de fondo de ojo.</p>'

                '<p>El modelo ha sido desarrollado y entrenado empleando Tensorflow en su versión 2.4, '
                'y la aplicación ha sido exportada mediante PyInstaller.</p>'

                "<p>The <b>Image Viewer</b> example shows how to combine "
                "QLabel and QScrollArea to display an image. QLabel is "
                "typically used for displaying text, but it can also display "
                "an image. QScrollArea provides a scrolling view around "
                "another widget. If the child widget exceeds the size of the "
                "frame, QScrollArea automatically provides scroll bars.</p>"

                "<p>The example demonstrates how QLabel's ability to scale "
                "its contents (QLabel.scaledContents), and QScrollArea's "
                "ability to automatically resize its contents "
                "(QScrollArea.widgetResizable), can be used to implement "
                "zooming and scaling features.</p>"

                "<p>In addition the example shows how to use QPainter to "
                "print an image.</p>")

    def createActions(self):
        self.openAct = QAction("&Abrir imagen...", self, shortcut="Ctrl+O",
                triggered=self.open)

        self.printAct = QAction("&Imprimir...", self, shortcut="Ctrl+P",
                enabled=False, triggered=self.print_)

        self.exitAct = QAction("Salir", self, shortcut="Ctrl+Q",
                triggered=self.close)

        self.zoomInAct = QAction("Ampliar (25%)", self, shortcut="Ctrl++",
                enabled=False, triggered=self.zoomIn)

        self.zoomOutAct = QAction("Alejar (25%)", self, shortcut="Ctrl+-",
                enabled=False, triggered=self.zoomOut)

        self.normalSizeAct = QAction("Tamaño original", self, shortcut="Ctrl+S",
                enabled=False, triggered=self.normalSize)

        self.fitToWindowAct = QAction("Ajustar a la ventana", self, enabled=False,
                checkable=True, shortcut="Ctrl+F", triggered=self.fitToWindow)

        self.aboutAct = QAction(f"&Acerca de {APPLICATION_TITLE}", self, triggered=self.about)

        self.aboutQtAct = QAction("Acerca de &Qt", self,
                triggered=QApplication.instance().aboutQt)

        #############
        self.shortcut_previous = QShortcut(QKeySequence("Left"), self)
        self.shortcut_previous.activated.connect(self.previous_image)

        self.shortcut_next = QShortcut(QKeySequence("Right"), self)
        self.shortcut_next.activated.connect(self.next_image)



    def createMenus(self):
        self.fileMenu = QMenu("Archivo", self)
        self.fileMenu.addAction(self.openAct)
        # self.fileMenu.addAction(self.printAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.viewMenu = QMenu("Ver", self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)

        self.helpMenu = QMenu("Ayuda", self)
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.aboutQtAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.helpMenu)

    def updateActions(self):
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.imageLabel.resize(self.scaleFactor * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

        self.zoomInAct.setEnabled(self.scaleFactor < 3.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.333)

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                                + ((factor - 1) * scrollBar.pageStep()/2)))



    # def keyPressEvent(self, event: QtGui.QKeyEvent):
    #     # print(event.key() == QtCore.Qt.Key_P)
    #     if not event.isAutoRepeat():
    #         print(event.text())
    #         print(event.key() == Qt.Key.Key_Control)
    #     return super().keyPressEvent(event)
        

    # def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
    #     if not event.isAutoRepeat():
    #         print('Soltada', event.text())
    #     return super().keyReleaseEvent(event)


    # def wheelEvent(self, event):
    #     print(event.angleDelta())


if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    imageViewer = ImageViewer()
    imageViewer.show()
    sys.exit(app.exec_())