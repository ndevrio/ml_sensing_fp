import numpy as np
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

data_len = 1000


##########################################
####     Misc Function Definitions    ####
##########################################

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


if __name__ == '__main__':
    #t1 = threading.Thread(target=get_data)
    #t1.start()
    
    while True:
        c = input(">")
        print(str(int(c)))

    #t1.join()
    
