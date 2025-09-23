from graphics import *
import time
import numpy as np
import math as mt

def ScalePolygon(polygon_coordinates,Sx,Sy):
    center = np.mean(polygon_coordinates,axis = 0)
    center[2] = 0
    polygon_coordinates = polygon_coordinates - center
    f = np.array([[Sx, 0, 0], [0, Sy, 0], [0, 0, 1]])
    polygon_coordinates = polygon_coordinates.dot(f)
    polygon_coordinates = polygon_coordinates + center
    return polygon_coordinates

def ShiftPolygon(polygon_coordinates,Dx,Dy):
    f = np.array([[1, 0, 0], [0, 1, 0], [Dx, Dy, 1]])
    return polygon_coordinates.dot(f)

xw = 600; yw = 600      # розміри вікна
win = GraphWin("2-D", xw, yw)
win.setBackground('white')
sx = 1.1; sy = 1.1      # масштабування
dx = 50; dy = -50       # переміщення
sx2 = 1; sy2 = 0.7      # масштабування 2

# Правильний пентагон
st = 50                 # розмір сторони правильного пентагону

polygon_coordinates = np.empty((5,3))

R = st / (2 * mt.sin(mt.pi / 5))

for i in range(5):
    angle = 2 * mt.pi * i / 5 + mt.pi / 2
    polygon_coordinates[i,0] = R * mt.cos(angle) + st
    polygon_coordinates[i,1] = yw - st - R * mt.sin(angle)
    polygon_coordinates[i,2] = 1

obj = Polygon(Point(polygon_coordinates[0,0], polygon_coordinates[0,1]), Point(polygon_coordinates[1,0], polygon_coordinates[1,1]), Point(polygon_coordinates[2,0], polygon_coordinates[2,1]), Point(polygon_coordinates[3,0], polygon_coordinates[3,1]), Point(polygon_coordinates[4,0], polygon_coordinates[4,1]))
obj.draw(win)
stop = xw/dx
stop = float(stop)
ii = int(stop)
for i in range(ii):
    time.sleep(0.5)
    obj.setOutline("white")
    polygon_coordinates = ScalePolygon(polygon_coordinates,sx,sy)
    obj = Polygon(Point(polygon_coordinates[0,0], polygon_coordinates[0,1]), Point(polygon_coordinates[1,0], polygon_coordinates[1,1]), Point(polygon_coordinates[2,0], polygon_coordinates[2,1]), Point(polygon_coordinates[3,0], polygon_coordinates[3,1]), Point(polygon_coordinates[4,0], polygon_coordinates[4,1]))
    obj.draw(win)
    time.sleep(0.5)
    polygon_coordinates = ShiftPolygon(polygon_coordinates,dx,dy)
    obj = Polygon(Point(polygon_coordinates[0,0], polygon_coordinates[0,1]), Point(polygon_coordinates[1,0], polygon_coordinates[1,1]), Point(polygon_coordinates[2,0], polygon_coordinates[2,1]), Point(polygon_coordinates[3,0], polygon_coordinates[3,1]), Point(polygon_coordinates[4,0], polygon_coordinates[4,1]))
    obj.draw(win)
    time.sleep(0.5)
    obj.setOutline("white")
    polygon_coordinates = ScalePolygon(polygon_coordinates,sx2,sy2)
    obj = Polygon(Point(polygon_coordinates[0,0], polygon_coordinates[0,1]), Point(polygon_coordinates[1,0], polygon_coordinates[1,1]), Point(polygon_coordinates[2,0], polygon_coordinates[2,1]), Point(polygon_coordinates[3,0], polygon_coordinates[3,1]), Point(polygon_coordinates[4,0], polygon_coordinates[4,1]))
    obj.draw(win)

win.getMouse()
win.close()