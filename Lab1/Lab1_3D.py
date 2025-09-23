from graphics import *
import time
import numpy as np
import math as mt
import random

def SortSides(Prxy):
    sides = []
    
    sides.append(((0,1,5,4), (Prxy[0,2] +  Prxy[1,2] +  Prxy[5,2] +  Prxy[4,2]) / 4))
    sides.append(((0,4,7,3), (Prxy[0,2] +  Prxy[4,2] +  Prxy[7,2] +  Prxy[3,2]) / 4))
    sides.append(((4,5,6,7), (Prxy[4,2] +  Prxy[5,2] +  Prxy[6,2] +  Prxy[7,2]) / 4))
    sides.append(((3,2,6,7), (Prxy[3,2] +  Prxy[2,2] +  Prxy[6,2] +  Prxy[7,2]) / 4))
    sides.append(((0,1,2,3), (Prxy[0,2] +  Prxy[1,2] +  Prxy[2,2] +  Prxy[3,2]) / 4))
    sides.append(((2,6,5,1), (Prxy[2,2] +  Prxy[6,2] +  Prxy[5,2] +  Prxy[1,2]) / 4))
    
    sides.sort(key=lambda x: x[1], reverse=True)

    return sides

def PrlpdWiz(Prxy, outline, fill, sides):
    for i in range(3):
        obj = Polygon(Point(Prxy[sides[i][0][0], 0], Prxy[sides[i][0][0], 1]), Point(Prxy[sides[i][0][1], 0], Prxy[sides[i][0][1], 1]), Point(Prxy[sides[i][0][2], 0], Prxy[sides[i][0][2], 1]), Point(Prxy[sides[i][0][3], 0], Prxy[sides[i][0][3], 1]))
        obj.setOutline(outline)
        obj.setFill(fill)
        obj.draw(win)
    
    return PrlpdWiz

def insertX (Figure, TetaG):
    TetaR = (mt.pi*TetaG)/180
    f = np.array([ [1, 0, 0, 0], [0, mt.cos(TetaR), mt.sin(TetaR), 0], [0, -mt.sin(TetaR),  mt.cos(TetaR), 0], [0, 0, 0, 1]])
    Prxy = Figure.dot(f)
    print('Rotate around X')
    # print(Prxy)
    return Prxy

def insertY (Figure, TetaG):
    TetaR = (mt.pi*TetaG)/180
    f = np.array([ [mt.cos(TetaR), 0, -mt.sin(TetaR), 0],[0, 1, 0, 0], [mt.sin(TetaR), 0,  mt.cos(TetaR), 0], [0, 0, 0, 1]])
    Prxy = Figure.dot(f)
    print('Rotate around Y')
    # print(Prxy)
    return Prxy

def ShiftXYZ (Figure, l, m, n):
   f = np.array([ [1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [l, m, n, 1] ])
   Prxy = Figure.dot(f)
   print('Shifting')
#    print(Prxy)
   return Prxy

def ProjectXY (Figure):
   f = np.array([ [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1] ])
   Prxy = Figure.dot(f)
   print('Proection on XY')
   return Prxy

xw = 600; yw = 600      # розміри вікна
win = GraphWin("3-D", xw, yw)
win.setBackground('white')
st = 300

Prlpd = np.array([ [0, 0, 0, 1],
                  [st, 0, 0, 1],
                  [st, st, 0, 1],
                  [0, st, 0, 1],
                  [0, 0, st, 1],
                  [st, 0, st, 1],
                  [st, st, st, 1],
                  [0, st, st, 1]])

TetaG1 = -30; TetaG2 = 30 # Поворот по X та Y відповідно
l = (xw/4)+st/2; m = (yw/4)-st; n = m # Зміщення

Prlpd1 = ShiftXYZ(Prlpd, l, -m, n)
Prlpd2 = insertX(Prlpd1, TetaG1)
Prlpd3 = insertY(Prlpd2, TetaG2)
sorted_sides = SortSides(Prlpd3)
outline_rgb = [0,0,0]
fill_rgb = [255,255,255]
rand_outline_rgb = [0,0,0]
rand_fill_rgb = [0,0,0]
PrlpdWiz(Prlpd3,color_rgb(outline_rgb[0],outline_rgb[1],outline_rgb[2]),color_rgb(fill_rgb[0],fill_rgb[1],fill_rgb[2]),sorted_sides)
reset = True
while True:
    time.sleep(0.5)
    if reset:
        if outline_rgb[0] != 255:
            outline_rgb[0] += random.randint(0,255-outline_rgb[0])
        if outline_rgb[1] != 255:
            outline_rgb[1] += random.randint(0,255-outline_rgb[1])
        if outline_rgb[2] != 255:
            outline_rgb[2] += random.randint(0,255-outline_rgb[2])
        if fill_rgb[0] != 255:
            fill_rgb[0] += random.randint(0,255-fill_rgb[0])
        if fill_rgb[1] != 255:
            fill_rgb[1] += random.randint(0,255-fill_rgb[1])
        if fill_rgb[2] != 255:
            fill_rgb[2] += random.randint(0,255-fill_rgb[2])
        if outline_rgb == [255,255,255] and fill_rgb == [255,255,255]:
            reset = False
            rand_outline_rgb = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]
            rand_fill_rgb = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]
    else:
        if outline_rgb[0] != rand_outline_rgb[0]:
            outline_rgb[0] -= random.randint(0,outline_rgb[0] - rand_outline_rgb[0])
        if outline_rgb[1] != rand_outline_rgb[1]:
            outline_rgb[1] -= random.randint(0,outline_rgb[1] - rand_outline_rgb[1])
        if outline_rgb[2] != rand_outline_rgb[2]:
            outline_rgb[2] -= random.randint(0,outline_rgb[2] - rand_outline_rgb[2])
        if fill_rgb[0] != rand_fill_rgb[0]:
            fill_rgb[0] -= random.randint(0,fill_rgb[0] - rand_fill_rgb[0])
        if fill_rgb[1] != rand_fill_rgb[1]:
            fill_rgb[1] -= random.randint(0,fill_rgb[1] - rand_fill_rgb[1])
        if fill_rgb[2] != rand_fill_rgb[2]:
            fill_rgb[2] -= random.randint(0,fill_rgb[2] - rand_fill_rgb[2])
        if outline_rgb == rand_outline_rgb and fill_rgb == rand_fill_rgb:
            reset = True

    PrlpdWiz(Prlpd3,color_rgb(outline_rgb[0],outline_rgb[1],outline_rgb[2]),color_rgb(fill_rgb[0],fill_rgb[1],fill_rgb[2]),sorted_sides)