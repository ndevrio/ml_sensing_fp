import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import socket
import threading
import struct
import os
import time
import sys

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, Dataset, Subset, DataLoader, random_split

os.environ['KMP_DUPLICATE_LIB_OK']='True'


send_pred_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Internet  # UDP
OUT_UDP_PORT = 7777

HOST = "0.0.0.0"
PORT = 8001
CHUNK = 1000      # buffer size for socket
phone_keys = ['unix_timestamp', 'acc_x', 'acc_y', 'acc_z', 'quart_x', 'quart_y', 'quart_z', 'quart_w', 'grav_x', 'grav_y', 'grav_z', 'roll', 'pitch', 'yaw'] 
classes = ['Front left pocket', 'Front right pocket', 'Back left pocket', 'Back right pocket', 'Tote bag']
current_class = 0

class_names = ['front_left_pocket', 'front_right_pocket', 'back_left_pocket', 'back_right_pocket']
samp_rate = 60
num_win_samples = 128
window_size = int(num_win_samples * (samp_rate/60))
if((window_size % 2) == 1):
    window_size += 1

class_buffer = np.zeros((window_size, 1)) # init with zero
time_buffer = np.zeros((window_size, 1)) # init with zero
raw_acc_buffer = np.zeros((window_size, 3)) # init with zero accel
raw_grav_buffer = np.zeros((window_size, 3)) # init with zero accel
raw_ori_buffer = np.zeros((window_size, 3)) # init with zero accel
raw_quat_buffer = np.array([[0, 0, 0, 1]] * window_size) # init with identity rotations

window_open = True
data_len = 1000

base_colors = [(255, 255, 0, 200), (255, 146, 0, 200), (255, 185, 224, 200), (248, 225, 64, 200)]

in_pocket = False
last_class = 0
start_time = time.time()
class_timer = start_time

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
    global current_class, window_open
    if(evt.key() == 16777234): # left arrow event
        current_class -= 1
        if(current_class < 0):
            current_class = len(classes)-1
        func_txt.setText(classes[current_class])
    if(evt.key() == 16777236): # right arrow event
        current_class += 1
        if(current_class >= len(classes)):
            current_class = 0
        func_txt.setText(classes[current_class])
    if(evt.key() == 32): # space bar event
        pass
    if(evt.key() == 81): # q
        window_open = False
        t1.join()


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
layoutgb.setColumnStretch(1, 5)
wd.setLayout(layoutgb)

func_txt_w = pg.GraphicsLayoutWidget()
vb = func_txt_w.addViewBox()
func_txt = pg.TextItem("Left front pocket", anchor=(0.5,0.5))
func_txt.setColor((150, 150, 150))
func_txt.setFont(QtGui.QFont("Bahnschrift SemiBold", 40, QtGui.QFont.Bold))
func_txt.setPos(0.5, 0.5)
vb.addItem(func_txt, ignoreBounds=True)
layoutgb.addWidget(func_txt_w, 0, 0)

timer_txt_w = pg.GraphicsLayoutWidget()
vb2 = timer_txt_w.addViewBox()
timer_txt = pg.TextItem("0", anchor=(0.5,0.5))
timer_txt.setColor((0, 0, 0))
timer_txt.setFont(QtGui.QFont("Bahnschrift SemiBold", 40, QtGui.QFont.Bold))
timer_txt.setPos(0.5, 0.5)
vb2.addItem(timer_txt, ignoreBounds=True)
layoutgb.addWidget(timer_txt_w, 0, 1)


a1 = win.addPlot(title="x")
a1.setLabel('left', "accel")
a2 = win.addPlot(title="y")
a3 = win.addPlot(title="z")
a1.enableAutoRange('xy', False)
a2.enableAutoRange('xy', False)
a3.enableAutoRange('xy', False)
a1.setXRange(0, window_size, padding=0)
a2.setXRange(0, window_size, padding=0)
a3.setXRange(0, window_size, padding=0)
a1.setYRange(-3, 3, padding=0.1)
a2.setYRange(-3, 3, padding=0.1)
a3.setYRange(-3, 3, padding=0.1)
acc_curve1 = a1.plot(raw_acc_buffer[:, 0], pen=pg.mkPen((0, 255, 0), width=5))
acc_curve2 = a2.plot(raw_acc_buffer[:, 1], pen=pg.mkPen((0, 255, 0), width=5))
acc_curve3 = a3.plot(raw_acc_buffer[:, 2], pen=pg.mkPen((0, 255, 0), width=5))

win.nextRow()

g1 = win.addPlot(title="x")
g1.setLabel('left', "gravity")
g2 = win.addPlot(title="y")
g3 = win.addPlot(title="z")
g1.enableAutoRange('xy', False)
g2.enableAutoRange('xy', False)
g3.enableAutoRange('xy', False)
g1.setXRange(0, window_size, padding=0)
g2.setXRange(0, window_size, padding=0)
g3.setXRange(0, window_size, padding=0)
g1.setYRange(-1, 1, padding=0.1)
g2.setYRange(-1, 1, padding=0.1)
g3.setYRange(-1, 1, padding=0.1)
grav_curve1 = g1.plot(raw_grav_buffer[:, 0], pen=pg.mkPen((255, 0, 255), width=5))
grav_curve2 = g2.plot(raw_grav_buffer[:, 1], pen=pg.mkPen((255, 0, 255), width=5))
grav_curve3 = g3.plot(raw_grav_buffer[:, 2], pen=pg.mkPen((255, 0, 255), width=5))

win.nextRow()

o1 = win.addPlot(title="roll")
o1.setLabel('left', "euler")
o2 = win.addPlot(title="pitch")
o3 = win.addPlot(title="yaw")
o1.enableAutoRange('xy', False)
o2.enableAutoRange('xy', False)
o3.enableAutoRange('xy', False)
o1.setXRange(0, window_size, padding=0)
o2.setXRange(0, window_size, padding=0)
o3.setXRange(0, window_size, padding=0)
o1.setYRange(-150, 150, padding=0.1)
o2.setYRange(-150, 150, padding=0.1)
o3.setYRange(-150, 150, padding=0.1)
ori_curve1 = o1.plot(raw_ori_buffer[:, 0], pen=pg.mkPen((255, 255, 0), width=5))
ori_curve2 = o2.plot(raw_ori_buffer[:, 1], pen=pg.mkPen((255, 255, 0), width=5))
ori_curve3 = o3.plot(raw_ori_buffer[:, 2], pen=pg.mkPen((255, 255, 0), width=5))

layoutgb.addWidget(win, 1, 0, 1, 2)

#############################
####    ML Definitions   ####
#############################

class Motion1DCNN(nn.Module):
    def __init__(self):
        super(Motion1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(32*window_size, 100)
        self.fc2 = nn.Linear(100, len(class_names))
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.dropout(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.dropout(x)

        x = self.fc1(x.view(x.size(0),-1))
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# Set up model and torch properties
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Motion1DCNN().to(device)
model.load_state_dict(torch.load("models/best_" + sys.argv[1]))
model.eval()


#############################
####    Main Functions   ####
#############################

send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Internet  # UDP
OUT_UDP_IP = "127.0.0.1"


def process_data(message):
    global time_buffer, raw_acc_buffer, raw_grav_buffer, raw_quat_buffer, raw_ori_buffer, current_class, in_pocket, start_time, class_timer

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

    if(device_type == "proximity"):
        if(data_str == "true"):
            in_pocket = True
            func_txt.setColor((255, 255, 255))
            timer_txt.setColor((150, 150, 150))
            start_time = time.time()
            class_timer = start_time
        else:
            in_pocket = False
            func_txt.setColor((150, 150, 150))
            timer_txt.setColor((0, 0, 0,))
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

    x = np.expand_dims(np.hstack([raw_acc_buffer, raw_grav_buffer]), axis=0)
    y = np.array(curr_class)
    x = torch.Tensor(np.moveaxis(x, 1, -1))
    y = torch.Tensor(y)

    pred = F.softmax(model(x), dim=1)
    pred = torch.argmax(pred, dim=1)

    current_class = pred[0].item()

    return device_name


def get_data():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:  # UDP
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))

        while window_open:
            try:
                data, addr = s.recvfrom(CHUNK)
                device_id = process_data(data)
            except KeyboardInterrupt:
                print('===== close socket =====')
                s.close()
                os._exit(0)
            except Exception as e:
                print(e)
                pass


class_count = 0
def update():
    global last_class, class_timer, class_count

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

    if in_pocket:
        if(time.time() - class_timer < 3.0):
            func_txt.setText(classes[current_class])
            timer_txt.setText("{:.2f} s".format((time.time() - start_time)))

            if(current_class != last_class):
                class_count += 1

                if(class_count > 5):
                    last_class = current_class
                    class_timer = time.time()
                    class_count = 0
        else:
            func_txt.setColor((50, 75, 255))
    else:
        func_txt.setText("Not in a pocket")

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
