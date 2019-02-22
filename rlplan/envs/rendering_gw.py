import sys
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow
from PyQt5.QtGui import QPainter, QColor, QBrush

TILE_SIZE = 100


class Renderer:
    def __init__(self, gridworld):
        self.gw = gridworld
        self.width = self.gw.ncols*TILE_SIZE
        self.height = self.gw.nrows*TILE_SIZE
        self.qt_app = QApplication(sys.argv)
        self.win = Window()
        self.widget = GridWorldWidget(self.gw)
        self.win.setCentralWidget(self.widget)
        self.win.resize(self.width, self.height)

    def run(self):
        if not self.win.finished:
            self.win.show()
            self.qt_app.exec_()


class Window(QMainWindow):

    def __init__(self):
        super().__init__()
        self.finished = False

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Return:
            self.close()
        elif e.key() == Qt.Key_Escape:
            self.close()
            self.finished = True


class GridWorldWidget(QWidget):

    def __init__(self, gw):
        super().__init__()
        self.gw = gw
        self.color1 = QColor(0, 0, 0)
        self.color2 = QColor(0, 0, 0)
        self.wall_color = QColor(120, 120, 120)

    def paintEvent(self, event):
        painter = QPainter(self)         # recupere le QPainter du widget
        painter.setPen(QColor(255, 255, 255))

        # draw grid
        aux = 1
        for rr in range(self.gw.nrows):
            for cc in range(self.gw.ncols):
                x0 = cc * TILE_SIZE
                y0 = rr * TILE_SIZE
                if aux == 1:
                    color = self.color1
                else:
                    color = self.color2
                if (rr, cc) in self.gw.walls:
                    color = self.wall_color
                if (rr, cc) in self.gw.reward_at:
                    reward = self.gw.reward_at[(rr,cc)]
                    if reward >= 0:
                        color = QColor(0, 200, 0)
                    else:
                        color = QColor(200, 0, 0)

                aux = -aux
                painter.fillRect(x0, y0, TILE_SIZE, TILE_SIZE, QBrush(color))
                painter.drawRect(x0, y0, TILE_SIZE, TILE_SIZE)

        # draw current state
        row, col = self.gw.index2coord[self.gw.state]
        x = col
        y = row
        x = (TILE_SIZE*x) + TILE_SIZE//2
        y = (TILE_SIZE * y) + TILE_SIZE // 2
        center = QPoint(x, y)
        painter.setPen(QColor(0, 0, 200))
        painter.setBrush(QColor(0, 0, 200))
        painter.drawEllipse(center, TILE_SIZE//4, TILE_SIZE//4)
        return

