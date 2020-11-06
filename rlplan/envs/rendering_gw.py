import sys
from PyQt5.QtCore import Qt, QPoint, QTimer
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow
from PyQt5.QtGui import QPainter, QColor, QBrush

TILE_SIZE = 50
RUNNING = False
STOP = False
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)


class Renderer:
    """
    :param mode: 'manual' to quit window with ENTER
                 'callback'to call callback_func every `delay` milliseconds
    """
    def __init__(self, render_info, mode='manual', callback_func=None, delay=500):
        self.render_info = render_info
        self.width = self.render_info['ncols']*TILE_SIZE
        self.height = self.render_info['nrows']*TILE_SIZE
        self.mode = mode
        self.callback_func = callback_func
        self.delay = delay

        self.win = Window()
        self.widget = GridWorldWidget(self.render_info)
        self.win.setCentralWidget(self.widget)
        self.win.resize(self.width, self.height)
        self.timer = QTimer()

    def reset(self):
        global STOP
        STOP = False

    def run(self, render_info=None):
        """
        :param render_info:
        :return:
        """
        global RUNNING, STOP

        # update render_info if necessary
        if render_info is not None:
            self.render_info = render_info
            self.widget.render_info = render_info

        # run 'callback' mode
        if self.mode == 'callback':
            if not STOP:
                self.timer.singleShot(self.delay, self.callback_func)
                self.widget.repaint()
                if not RUNNING:
                    self.win.show()
                    self.win.setFocus()
                    RUNNING = True
                    app.exec_()
            if STOP:
                self.timer.stop()
                return 1
            return 0

        # run 'manual' mode
        elif self.mode == 'manual':
            if not RUNNING:
                self.win.show()
                self.win.setFocus()
                RUNNING = True
                app.exec_()
            return 0


class Window(QMainWindow):

    def __init__(self):
        super().__init__()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape or e.key() == Qt.Key_Return:
            self.finish()

    def closeEvent(self, event):
        self.finish()

    def finish(self):
        global RUNNING, STOP
        self.close()
        app_instance = QApplication.instance()
        if app_instance is not None:
            app_instance.quit()
        RUNNING = False
        STOP = True


class GridWorldWidget(QWidget):

    def __init__(self, render_info):
        super().__init__()
        self.render_info = render_info
        self.color1 = QColor(0, 0, 0)
        self.color2 = QColor(0, 0, 0)
        self.wall_color = QColor(120, 120, 120)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QColor(120, 120, 120))

        # draw grid
        aux = 1
        for rr in range(self.render_info['nrows']):
            for cc in range(self.render_info['ncols']):
                x0 = cc * TILE_SIZE
                y0 = rr * TILE_SIZE
                if aux == 1:
                    color = self.color1
                else:
                    color = self.color2
                if (rr, cc) in self.render_info['walls']:
                    color = self.wall_color
                if (rr, cc) in self.render_info['reward_at']:
                    reward = self.render_info['reward_at'][(rr, cc)]
                    if reward >= 0:
                        color = QColor(0, reward*200, 0)
                    else:
                        color = QColor(200, 0, 0)

                aux = -aux
                painter.fillRect(x0, y0, TILE_SIZE, TILE_SIZE, QBrush(color))
                painter.drawRect(x0, y0, TILE_SIZE, TILE_SIZE)

        # draw current state
        row, col = self.render_info['current_state']
        x = col
        y = row
        x = (TILE_SIZE*x) + TILE_SIZE//2
        y = (TILE_SIZE * y) + TILE_SIZE // 2
        center = QPoint(x, y)
        painter.setPen(QColor(0, 0, 200))
        painter.setBrush(QColor(0, 0, 200))
        painter.drawEllipse(center, TILE_SIZE//4, TILE_SIZE//4)
        return

