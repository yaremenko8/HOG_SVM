import numpy as np
from scipy import misc, signal
import time

blockRowCells  = 2
blockColCells  = 2
cellRows       = 12
cellCols       = 12
binCount       = 10
WIDTH          = 98 #96 + 2
HEIGHT         = 98 

Sx = [[1, 0, -1],
      [2, 0, -2],
      [1, 0, -1]]
Sy = [[-1,-2,-1],
      [ 0, 0, 0],
      [1,  2, 1]]

Pi = 3.14159265359

def convolve(pic, krl):   #colored
    r = len(krl)
    fm_width  = pic.shape[0] - r + 1
    fm_height = pic.shape[1] - r + 1
    fm = [[0 for j in range(fm_height)] for i in range(fm_width)]
    for i in range(fm_width):
        for j in range(fm_height):    
            fm[i][j] = sum([pic[i + x][j + y] * krl[x][y] for x in range(r) for y in range(r)])   
    return fm

def greyscale(pic_):
    w = pic_.shape[0]
    h = pic_.shape[1]
    pic = np.copy(pic_).tolist()
    for i in range(w):
        for j in range(h):
            pic[i][j] = pic[i][j][0] * 0.299 + pic[i][j][1] * 0.587 + pic[i][j][2] * 0.114
    return np.array(pic, np.dtype('uint8'))

norm = lambda s: np.sqrt(sum(map(lambda x: x**2, s)))

def extract_hog(img):
    #t = time.time()
    pic = misc.imread(img) if type(img) == str else np.copy(img)
    #print(time.time() - t)
    #t = time.time()
    pic = misc.imresize(pic, (WIDTH, HEIGHT))
    pic = greyscale(pic)
    #print(time.time() - t)
    #t = time.time()
    Ix = signal.convolve2d(pic, Sx, mode = "valid").tolist()
    Iy = signal.convolve2d(pic, Sy, mode = "valid").tolist()
    #print(time.time() - t)
    t = time.time()
    ANG = [[np.arctan2(Iy[i][j], Ix[i][j]) for j in range(len(Ix[0]))] for i in range(len(Ix))]
    #print(time.time() - t)
    t = time.time()
    ABS = [[norm((Ix[i][j], Iy[i][j])) for j in range(len(Ix[0]))] for i in range(len(Ix))]
    #print(time.time() - t)
    #t = time.time()
    cell_hgrams = [[[0 for k in range(binCount)] for j in range(len(Ix[0]) // cellCols)] for i in range(len(Ix) // cellRows)]
    for i in range(len(cell_hgrams)):
        for j in range(len(cell_hgrams[0])):
            for a in range(cellRows):
                for b in range(cellCols):
                    x = i * cellRows + a
                    y = j * cellCols + b
                    q = 2 * Pi / binCount
                    cell_hgrams[i][j][int((ANG[x][y] + Pi) // q)] += ABS[x][y]
    #print(time.time() - t)
    #t = time.time()
    blocks = [[[] for j in range(len(cell_hgrams[0]) // blockColCells)] for i in range(len(cell_hgrams) // blockRowCells)]
    for i in range(len(blocks)):
        for j in range(len(blocks[0])):
            current = []
            for a in range(blockRowCells):
                for b in range(blockColCells):    
                    x = i * blockRowCells + a
                    y = j * blockColCells + b
                    current += cell_hgrams[x][y]
            blocks[i][j] = list(map(lambda x: x / np.sqrt(norm(current) ** 2 + 0.1), current))
    #print(time.time() - t)   
    descriptor = []
    for i in blocks:
        for j in i:
            descriptor += j
    return descriptor
    
#print(len(hog_descriptor("//home/grigory/train/00029.png")))
