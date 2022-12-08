import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import socket
import threading
import struct
import os
import time


send_pred_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Internet  # UDP
OUT_UDP_PORT = 7777

HOST = "0.0.0.0"
PORT = 8001
CHUNK = 1000      # buffer size for socket
buffer_size = 500
phone_keys = ['unix_timestamp', 'acc_x', 'acc_y', 'acc_z', 'quart_x', 'quart_y', 'quart_z', 'quart_w', 'grav_x', 'grav_y', 'grav_z', 'roll', 'pitch', 'yaw'] 
classes = ['Left front pocket', 'Right front pocket', 'Back left pocket', 'Back right pocket', 'Tote bag']
class_names = ['front_left_pocket', 'front_right_pocket', 'back_left_pocket', 'back_right_pocket']
current_class = 0

class_buffer = np.zeros((buffer_size, 1)) # init with zero
time_buffer = np.zeros((buffer_size, 1)) # init with zero
raw_acc_buffer = np.zeros((buffer_size, 3)) # init with zero accel
raw_grav_buffer = np.zeros((buffer_size, 3)) # init with zero accel
raw_ori_buffer = np.zeros((buffer_size, 3)) # init with zero accel
raw_quat_buffer = np.array([[0, 0, 0, 1]] * buffer_size) # init with identity rotations

save_data = []

window_open = True
data_len = 1000

base_colors = [(255, 255, 0, 200), (255, 146, 0, 200), (255, 185, 224, 200), (248, 225, 64, 200)]


##########################################
####     Misc Function Definitions    ####
##########################################

class KeyPressWindow(pg.GraphicsLayoutWidget):
    sigKeyPress = QtCore.pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def keyPressEvent(self, ev):
        self.scene().keyPressEvent(ev)
        self.sigKeyPress.emit(ev)

def keyPressed(evt):
    global current_class
    if(evt.key() == 16777234 and not record_state): # left arrow event
        current_class -= 1
        if(current_class < 0):
            current_class = len(classes)-1
        func_txt.setText(classes[current_class])
    if(evt.key() == 16777236 and not record_state): # right arrow event
        current_class += 1
        if(current_class >= len(classes)):
            current_class = 0
        func_txt.setText(classes[current_class])
    if(evt.key() == 32): # space bar event
        funcButtonClicked()

record_state = False
def funcButtonClicked():
    global record_state

    if(not record_state):
        ib_func.setStyleSheet("background-image: url(stop.png);" + style_base)
        func_txt.setColor((255, 0, 0))
        record_state = True
        print("Recording")
    else:
        ib_func.setStyleSheet("background-image: url(record.png);" + style_base)
        func_txt.setColor((255, 255, 255))
        record_state = False
        print("Stopped")
        save_data_file()


#########################
####    Qt Section   ####
#########################

app = pg.mkQApp("Plotting Example")
win = KeyPressWindow(show=False, title="Basic plotting examples")
win.sigKeyPress.connect(keyPressed)
win.resize(1000,600)
win.setWindowTitle('pyqtgraph example: Plotting')

wd = pg.GraphicsWindow(title="Pocket Detect Server")
layoutgb = QtGui.QGridLayout()
layoutgb.setRowStretch(0, 5)
layoutgb.setRowStretch(1, 20)

layoutgb.setColumnStretch(0, 20)
layoutgb.setColumnStretch(1, 20)
wd.setLayout(layoutgb)

style_base = "background-color: rgb(0, 0, 0);background-position: center center;background-repeat: no-repeat;border: 4px solid white "
ib_func = QtGui.QPushButton()
ib_func.setStyleSheet("background-image: url(record.png);" + style_base)
layoutgb.addWidget(ib_func, 0, 0)
ib_func.clicked.connect(funcButtonClicked)
ib_func.sizeHint = lambda: pg.QtCore.QSize(50, 50)

func_txt_w = pg.GraphicsLayoutWidget()
vb = func_txt_w.addViewBox()
func_txt = pg.TextItem("Left pocket", anchor=(0.5,0.5))
func_txt.setColor((255, 255, 255))
func_txt.setFont(QtGui.QFont("Bahnschrift SemiBold", 20, QtGui.QFont.Bold))
func_txt.setPos(0.5, 0.5)
vb.addItem(func_txt, ignoreBounds=True)
layoutgb.addWidget(func_txt_w, 0, 1)


a1 = win.addPlot(title="x")
a1.setLabel('left', "accel")
a2 = win.addPlot(title="y")
a3 = win.addPlot(title="z")
a1.enableAutoRange('xy', False)
a2.enableAutoRange('xy', False)
a3.enableAutoRange('xy', False)
a1.setXRange(0, buffer_size, padding=0)
a2.setXRange(0, buffer_size, padding=0)
a3.setXRange(0, buffer_size, padding=0)
a1.setYRange(-3, 3, padding=0.1)
a2.setYRange(-3, 3, padding=0.1)
a3.setYRange(-3, 3, padding=0.1)
acc_curve1 = a1.plot(raw_acc_buffer[:, 0], pen=(0, 255, 0))
acc_curve2 = a2.plot(raw_acc_buffer[:, 1], pen=(0, 255, 0))
acc_curve3 = a3.plot(raw_acc_buffer[:, 2], pen=(0, 255, 0))

win.nextRow()

g1 = win.addPlot(title="x")
g1.setLabel('left', "gravity")
g2 = win.addPlot(title="y")
g3 = win.addPlot(title="z")
g1.enableAutoRange('xy', False)
g2.enableAutoRange('xy', False)
g3.enableAutoRange('xy', False)
g1.setXRange(0, buffer_size, padding=0)
g2.setXRange(0, buffer_size, padding=0)
g3.setXRange(0, buffer_size, padding=0)
g1.setYRange(-1, 1, padding=0.1)
g2.setYRange(-1, 1, padding=0.1)
g3.setYRange(-1, 1, padding=0.1)
grav_curve1 = g1.plot(raw_grav_buffer[:, 0], pen=(255, 0, 255))
grav_curve2 = g2.plot(raw_grav_buffer[:, 1], pen=(255, 0, 255))
grav_curve3 = g3.plot(raw_grav_buffer[:, 2], pen=(255, 0, 255))

win.nextRow()

o1 = win.addPlot(title="roll")
o1.setLabel('left', "euler")
o2 = win.addPlot(title="pitch")
o3 = win.addPlot(title="yaw")
o1.enableAutoRange('xy', False)
o2.enableAutoRange('xy', False)
o3.enableAutoRange('xy', False)
o1.setXRange(0, buffer_size, padding=0)
o2.setXRange(0, buffer_size, padding=0)
o3.setXRange(0, buffer_size, padding=0)
o1.setYRange(-150, 150, padding=0.1)
o2.setYRange(-150, 150, padding=0.1)
o3.setYRange(-150, 150, padding=0.1)
ori_curve1 = o1.plot(raw_ori_buffer[:, 0], pen=(255, 255, 0))
ori_curve2 = o2.plot(raw_ori_buffer[:, 1], pen=(255, 255, 0))
ori_curve3 = o3.plot(raw_ori_buffer[:, 2], pen=(255, 255, 0))

layoutgb.addWidget(win, 1, 0, 1, 2)


#############################
####    Main Functions   ####
#############################

send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Internet  # UDP
OUT_UDP_IP = "127.0.0.1"

        
def save_data_file():
    global save_data

    outfile = "data/pocket_raw_data_" + class_names[current_class] + '_' + time.strftime("%Y%m%d_%H%M%S") 

    if(len(save_data) == 0):
        print('Saved nothing')
        return

    # Save the collected data
    sd = np.vstack(save_data)

    print("Collected data samples: ", sd.shape, "Saved in: ", outfile)
    np.save(outfile, sd)
    np.savetxt(outfile + ".csv", sd, delimiter=",")

    save_data = []


def process_data(message):
    global time_buffer, raw_acc_buffer, raw_grav_buffer, raw_quat_buffer, raw_ori_buffer

    """Receive data from socket.
    """
    message = message.strip()
    if not message:
        return
    message = message.decode('utf-8')
    if message == 'stop':
        return
    if ':' not in message:
        print(message)
        return

    try:
        device_id, raw_data_str = message.split(";")
        device_type, data_str = raw_data_str.split(':')
    except Exception as e:
        print(e, message)
        return

    data = []
    for d in data_str.strip().split(' '):
        try:
            data.append(float(d))
        except Exception as e:
            print(e)
            continue
    
    if len(data) != len(phone_keys):
        print("something is missing...skipping packet")
        return

    device_name = "phone"

    #print(device_type, data)

    # update the buffers
    curr_class = np.array(current_class).reshape(1, 1)
    curr_time = np.array(data[0]).reshape(1, 1)
    curr_acc = np.array(data[1:4]).reshape(1, 3)
    curr_quat = np.array(data[4:8]).reshape(1, 4)
    curr_grav = np.array(data[8:11]).reshape(1, 3)
    curr_ori = np.array(data[11:14]).reshape(1, 3) * (180 / np.pi)

    time_buffer = np.concatenate([time_buffer[1:], curr_time])
    raw_acc_buffer = np.concatenate([raw_acc_buffer[1:], curr_acc])
    raw_grav_buffer = np.concatenate([raw_grav_buffer[1:], curr_grav])
    raw_ori_buffer = np.concatenate([raw_ori_buffer[1:], curr_ori])
    raw_quat_buffer = np.concatenate([raw_quat_buffer[1:], curr_quat])

    z = np.hstack([curr_class, curr_time, curr_acc, curr_grav, curr_ori, curr_quat])
    save_data.append(z)

    return device_name


def get_data():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:  # UDP
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))

        while True:
            try:
                data, addr = s.recvfrom(CHUNK)
                device_id = process_data(data)
            except KeyboardInterrupt:
                print('===== close socket =====')
                os._exit(0)
            except Exception as e:
                print(e)
                pass


def update():
    # Update all curve plots
    acc_curve1.setData(raw_acc_buffer[:, 0])
    acc_curve2.setData(raw_acc_buffer[:, 1])
    acc_curve3.setData(raw_acc_buffer[:, 2])
    grav_curve1.setData(raw_grav_buffer[:, 0])
    grav_curve2.setData(raw_grav_buffer[:, 1])
    grav_curve3.setData(raw_grav_buffer[:, 2])
    ori_curve1.setData(raw_ori_buffer[:, 0])
    ori_curve2.setData(raw_ori_buffer[:, 1])
    ori_curve3.setData(raw_ori_buffer[:, 2])

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start()


if __name__ == '__main__':
    t1 = threading.Thread(target=get_data)
    t1.start()
    
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

    window_open = False
    
