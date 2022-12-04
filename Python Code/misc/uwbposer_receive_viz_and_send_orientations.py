import socket
import threading
from collections import deque, defaultdict
import os
from pathlib import Path
import numpy as np

import pygame
import math
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *
from numpy.linalg import inv
import time
from scipy.spatial.transform import Rotation as R

send_pred_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Internet  # UDP
OUT_UDP_PORT = 7777

HOST = "0.0.0.0"
PORT = 8001
CHUNK = 1000      # buffer size for socket
BUFFER_SIZE = 50   # 300/25 secs of data = 12
WATCH_KEYS = ['unix_timestamp', 'quart_x', 'quart_y', 'quart_z', 'quart_w', 'grav_x', 'grav_y', 'grav_z']
#PHONE_KEYS = ['unix_timestamp', 'distance', 'quart_x', 'quart_y', 'quart_z', 'quart_w', 'grav_x', 'grav_y', 'grav_z'] 
PHONE_KEYS = ['unix_timestamp', 'acc_x', 'acc_y', 'acc_z', 'quart_x', 'quart_y', 'quart_z', 'quart_w', 'grav_x', 'grav_y', 'grav_z'] 

DATA = {}

device_ids = {
    "Left_phone": 0,
    "Left_watch": 1,
}
display_names = ["Phone", "Watch"]

data_refreshed = np.zeros((2,), dtype=bool)

time_buffer = {id: np.zeros((BUFFER_SIZE, 1)) for id in device_ids.values()} # init with zero
raw_dist_buffer = {id: np.zeros((BUFFER_SIZE, 1)) for id in device_ids.values()} # init with zero accel
raw_grav_buffer = {id: np.zeros((BUFFER_SIZE, 3)) for id in device_ids.values()} # init with zero accel
raw_ori_buffer = {id: np.array([[0, 0, 0, 1]] * BUFFER_SIZE) for id in device_ids.values()} # init with identity rotations
calibration_quats = {id: np.array([0, 0, 0, 1]) for id in device_ids.values()}
device2bones_quats = {id: np.array([0, 0, 0, 1]) for id in device_ids.values()}
smpl2imu = np.eye(3)

virtual_time = {id: np.array([[0]]) for id in device_ids.values()}
virtual_grav = {id: np.zeros((1, 3)) for id in device_ids.values()} # init with zero accel
virtual_ori = {id: np.array([0, 0, 0, 1]) for id in device_ids.values()} # init with identity rotations
virtual_dist = {id: np.array([[0]]) for id in device_ids.values()} # init with identity rotations

device_positions = [(-1.55, 0, -10.0), (1.55, 0, -10.0), (-2.5, 0, -10.0), (2.5, 0, -10.0), (5, 0, -10.0)]

send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Internet  # UDP
OUT_UDP_IP = "127.0.0.1"

def drawText(position, textString, size):

    font = pygame.font.SysFont("ebrima", size, True)
    textSurface = font.render(textString, True, (0, 0, 0, 255), (255, 255, 255, 255))
    textData = pygame.image.tostring(textSurface, "RGBA", True)
    glRasterPos3d(*position)
    glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)

def draw_cuboid(w=2, h=2, d=0.4, colors=None):
    w = w / 2
    h = h / 2
    d = d / 2

    colors = [(0.0, 1.0, 0.0), (1.0, 0.5, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 1.0)]

    glBegin(GL_QUADS)
    glColor3f(*colors[0])

    glVertex3f(w, d, -h)
    glVertex3f(-w, d, -h)
    glVertex3f(-w, d, h)
    glVertex3f(w, d, h)



    glColor3f(*colors[1])

    glVertex3f(w, -d, h)
    glVertex3f(-w, -d, h)
    glVertex3f(-w, -d, -h)
    glVertex3f(w, -d, -h)

    glColor3f(*colors[2])

    glVertex3f(w, d, h)
    glVertex3f(-w, d, h)
    glVertex3f(-w, -d, h)
    glVertex3f(w, -d, h)

    glColor3f(*colors[3])

    glVertex3f(w, -d, -h)
    glVertex3f(-w, -d, -h)
    glVertex3f(-w, d, -h)
    glVertex3f(w, d, -h)

    glColor3f(*colors[4])

    glVertex3f(-w, d, h)
    glVertex3f(-w, d, -h)
    glVertex3f(-w, -d, -h)
    glVertex3f(-w, -d, h)

    glColor3f(*colors[5])

    glVertex3f(w, d, -h)
    glVertex3f(w, d, h)
    glVertex3f(w, -d, h)
    glVertex3f(w, -d, -h)

    glEnd()

def draw_uwb(distance):
    glLoadIdentity()
    glTranslatef(*device_positions[2])
    drawText((-0.8, -2.7, 0), "UWB distance:   {0:.2f} m away".format(distance), 30)

    # Draw bar graph
    glBegin(GL_QUADS)
    glColor3f(*(0.5, 0.0, 0.5))

    bar_left = -0.8
    bar_top = -3

    max_width = 6.5
    bar_width = (distance / 1.8) * max_width
    if(bar_width > max_width):
        bar_width = max_width

    glVertex3f(bar_left, bar_top, 0)
    glVertex3f(bar_left+bar_width, bar_top, 0)
    glVertex3f(bar_left+bar_width, bar_top-0.2, 0)
    glVertex3f(bar_left, bar_top-0.2, 0)

    glEnd()


def draw(device_id, ori):
    [nx, ny, nz, w] = list(ori)
    glLoadIdentity()
    device_pos = device_positions[device_id] 

    glTranslatef(*device_pos)
    drawText((-0.7, 1.7, 0), display_names[device_id], 30)
    # drawText((-0.7, -1.8, 0), f"a_x: {acc[0]:.3f}", 14)
    # drawText((-0.7, -2.1, 0), f"a_y: {acc[1]:.3f}", 14)
    # drawText((-0.7, -2.4, 0), f"a_z: {acc[2]:.3f}", 14)
    # glRotatef(2 * math.acos(w) * 180.00/math.pi, -1 * nx, nz, ny)
   #  if device_id == 2:
   #      glRotatef(2 * math.acos(w) * 180.00/math.pi, -nx, -nz, -ny)
   #  else:
    glRotatef(2 * math.acos(w) * 180.00/math.pi, nx, nz, ny)
    draw_cuboid(1.5, 1.5, 1.5)

def process_data(message):
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
        
    if device_type == "watch":
        if len(data) != len(WATCH_KEYS):
            print("something is missing...skipping packet")
            return
    elif device_type == "phone":
        if len(data) != len(PHONE_KEYS):
            print("something is missing...skipping packet")
            return

    device_name = device_ids[f"Left_{device_type}"]

    #print(device_type, data)

    # update the buffers
    #if device_type == "watch":
    curr_time = np.array(data[0]).reshape(1, 1)
    curr_acc = np.array(data[1:4]).reshape(1, 3)
    curr_ori = np.array(data[4:8]).reshape(1, 4)
    curr_grav = np.array(data[8:11]).reshape(1, 3)
    # no distance
    curr_dist = np.array([0]).reshape(1, 1)
    """elif device_type == "phone":
        curr_time = np.array(data[0]).reshape(1, 1)
        curr_grav = np.array(data[6:9]).reshape(1, 3)
        curr_ori = np.array(data[2:6]).reshape(1, 4)
        curr_dist = np.array(data[1]).reshape(1, 1)"""

    time_buffer[device_name] = np.concatenate([time_buffer[device_name][1:], curr_time])
    raw_dist_buffer[device_name] = np.concatenate([raw_dist_buffer[device_name][1:], curr_dist])
    raw_grav_buffer[device_name] = np.concatenate([raw_grav_buffer[device_name][1:], curr_grav])
    raw_ori_buffer[device_name] = np.concatenate([raw_ori_buffer[device_name][1:], curr_ori])

    return device_name

def resizewin(width, height):
    """
    For resizing window
    """
    if height == 0:
        height = 1
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-7, 7, -7, 7, 0, 15)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def init():
    glShadeModel(GL_SMOOTH)
    glClearColor(255.0, 255.0, 255.0, 0.0)
    glClearDepth(1.0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)

def read_data(line):
    line = line.replace('\n', '')
                
    w = float(line.split('w')[1])
    nx = float(line.split('a')[1])
    ny = float(line.split('b')[1])
    nz = float(line.split('c')[1])
    return [w, nx, ny, nz]

def sensor2global(ori, acc, device_id):
    # this function works!
    device_mean_quat = calibration_quats[device_id]

    og_mat = R.from_quat(ori).as_matrix()
    global_inertial_frame = R.from_quat(device_mean_quat).as_matrix()
    global_mat = (global_inertial_frame.T).dot(og_mat)
    global_quat = R.from_matrix(global_mat).as_quat()

    sensor_rel_acc = og_mat.dot(acc) # align acc to the sensor frame of ref
    global_acc = (global_inertial_frame.T).dot(sensor_rel_acc) # align acc to the world frame
    return global_quat, global_acc

def send_data_to_transpose():
    time = 0
    ori = []
    dist = []
    grav = []
    for _id in [1, 0]: # send left watch then left phone
        ori.append(virtual_ori[_id][[3, 0, 1, 2]])
        dist.append(virtual_dist[_id])
        grav.append(virtual_grav[_id])
        time += virtual_time[_id][0]

    t = time / 2
    o = np.array(ori)
    d = np.array(dist)
    g = np.array(grav)

    s = str(t) + '@' + \
        ','.join(['%g' % v for v in d.flatten()]) + '#' + \
        ','.join(['%g' % v for v in o.flatten()]) + '$' + \
        ','.join(['%g' % v for v in g.flatten()])

    sensorBytes = bytes(s, encoding="utf8")
    send_pred_sock.sendto(sensorBytes, (OUT_UDP_IP, OUT_UDP_PORT))

if __name__ == '__main__':
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:  # UDP
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))

        video_flags = OPENGL | DOUBLEBUF
        pygame.init()
        screen = pygame.display.set_mode((860, 860), video_flags)
        pygame.display.set_caption("PyTeapot IMU orientation visualization")
        resizewin(860, 860)
        init()
        ticks = pygame.time.get_ticks()

        dist = 0
        while True:
            try:
                data, addr = s.recvfrom(CHUNK)
                device_id = process_data(data)

                event = pygame.event.poll()

                # quit
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    break

                if (event.type == KEYDOWN and event.key == K_c):
                    print("Started calibration!")
                if (event.type == KEYUP and event.key == K_c):
                    # Calc the mean quat 
                    for _id in raw_ori_buffer.keys():
                        mean_quat = np.mean(raw_ori_buffer[_id][-30:], axis=0) # 30 frames @20 Hz is 1.5sec
                        calibration_quats[_id] = mean_quat
                    print("Finished calibration")

                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                t = time_buffer[device_id][-1]
                o = raw_ori_buffer[device_id][-1]
                g = raw_grav_buffer[device_id][-1]
                d = raw_dist_buffer[device_id][-1]

                # convert to a global inertial frame
                global_o, global_g = sensor2global(o, g, device_id)

                virtual_time[device_id] = t
                virtual_grav[device_id] = global_g.reshape(1, 3)
                virtual_ori[device_id] = global_o
                virtual_dist[device_id] = d

                [nx, ny, nz, w] = list(global_o)

                if(float(d) > 0):
                    dist = float(d)
                draw_uwb(dist)
                for id in virtual_ori.keys():
                    # there's a for loop here because we want to re-render the whole frame
                    _o = virtual_ori[id]
                    _g = virtual_grav[id]
                    draw(id, _o)

                # Only send data over socket once both values have been refreshed
                data_refreshed[device_id] = 1
                if(np.all(data_refreshed)):
                    send_data_to_transpose()
                    data_refreshed.fill(0)

                pygame.display.flip()
            except KeyboardInterrupt:
                print('===== close socket =====')
                os._exit(0)
            except Exception as e:
                print(e)
                pass
