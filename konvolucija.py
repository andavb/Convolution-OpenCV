from cv2 import *
import numpy as np
from PIL import Image
from scipy import ndimage

img=cv2.imread("nova.png")

A=0
G=0
M=0 #visina slike height
N=0 #dolzina slike  width
NMS=0


def togray():
    global img
    img=cv2.cvtColor(img, COLOR_RGB2GRAY);


def gaussianblurring():
    global img, img2, gauss
    #izvede funkcijo z 3x3 gaussian blut s standardnim odklonom 0
    img = cv2.GaussianBlur(img, (3, 3), 0)
    cv2.imshow("gaussianblurring", img)


def sobelfilter():
    global img, G, A, gradient, gradientX, gradientY
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)#CV_32 za 32bitne slike, ksize je matrika 3x3
    absx = np.absolute(gx)
    cv2.imshow("gx", np.uint8(absx))

    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    absy = np.absolute(gy)
    cv2.imshow("gy", np.uint8(absy))

    G=sqrt(gx**2 + gy**2)

    A=np.arctan2(gx, gy)
    Y, X = img.shape[:2]

    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    Grad = np.hypot(Ix, Iy)
    Grad = Grad / Grad.max() * 255
    theta = np.arctan2(Iy, Ix)

    M, N = img.shape[:2]

    for y in range(M):
        for x in range(N):
            if theta[y][x] < 0:
                print(theta[y][x])


    cv2.imshow("gradient", np.uint8(G))#np.uint8 prtevorimo iz 32bitne v 8bitno


def tansanjerobov():
    global M, N, G, A, NMS, trobov
    M, N = img.shape[:2]

    print(M, N)

    NMS = G

    for i in range(M):
        for j in range(N):

            if (i-1 >= 0 and i+1 != M) and (j-1 >= 0 and j+1 != N):
                if A[i][j] < 0:
                    A[i][j] += 360 #za negativne kote

                # 360-22.5, 350-157.5
                if (0 <= A[i][j] < 22.5) or (157.5 <= A[i][j] <= 180) or (337.5 <= A[i][j]) or (157.5 <= A[i][j] < 202.5):
                    if (G[i][j] <= G[i][j - 1]) or (G[i][j] <= G[i][j + 1]):
                        NMS[i][j] = 0

                # 180+67.5 180+22.5
                if (22.5 <= A[i][j] < 67.5) or (202.5<=A[i][j]<247.5):
                    if (G[i][j] <= G[i - 1][j + 1]) or (G[i][j] <= G[i + 1][j - 1]):
                        NMS[i][j] = 0

                #180+67.5 180+112.5
                if (67.5 <= A[i][j] < 112.5) or (247.5 <= A[i][j] < 292.5):
                    if (G[i][j] <= G[i - 1][j]) or (G[i][j] <= G[i + 1][j]):
                        NMS[i][j] = 0

                #180+112.5 180+157.5
                if (112.5 <= A[i][j] < 157.5) or (292.5 <= A[i][j] < 337.5):
                    if (G[i][j] <= G[i - 1][j - 1]) or (G[i][j] <= G[i + 1][j + 1]):
                        NMS[i][j] = 0

    cv2.imshow("tansanjerobov", np.uint8(NMS))


def dvojnoupragovljanje(low, strong):

    global M, N, NMS, G, A, dvupr
    i=0
    i=0
    nova = np.zeros([M, N])


    for i in range(M - 1):
        for j in range(N - 1):

            if NMS[i][j] <= low:
                if((NMS[i+1][j] >= strong) or (NMS[i-1][j] >= strong) or (NMS[i][j+1] >= strong)
                    or (NMS[i][j-1] >= strong) or (NMS[i-1][j-1] >= strong) or (NMS[i+1][j+1] >= strong)):
                    nova[i][j] = 255
                else:
                    nova[i][j] = 0

    cv2.imshow("dvojnoupragovljanje", np.uint8(nova))


gaussianblurring()

togray()

sobelfilter()

tansanjerobov()

dvojnoupragovljanje(80, 160)

#edges = cv2.Canny(img,100,200)


#cv2.imshow("image", edges)

cv2.waitKey(0)