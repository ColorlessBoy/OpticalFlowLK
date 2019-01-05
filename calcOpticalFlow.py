import numpy as np
from matplotlib import pyplot as plt
import sys, os, cv2, time

def calcOpticalFlowLK(prevImg, nextImg, winSize = 39, threshD = 1e-9):
    prevImg = prevImg/1.
    nextImg = nextImg/1.
    prevDx = cv2.Sobel(prevImg,cv2.CV_64F,1,0, ksize=3)
    prevDy = cv2.Sobel(prevImg,cv2.CV_64F,0,1, ksize=3)
    nextDx = cv2.Sobel(nextImg,cv2.CV_64F,1,0, ksize=3)
    nextDy = cv2.Sobel(nextImg,cv2.CV_64F,0,1, ksize=3)
    sigma1 = prevDx**2 + nextDx**2
    sigma2 = sigma4 = prevDx*prevDy+nextDx*nextDy
    sigma3 = (nextImg-prevImg)*(prevDx+nextDx)
    sigma5 = prevDy**2 + nextDy**2
    sigma6 = (nextImg-prevImg)*(prevDy+nextDy)
    sigma = [sigma1, sigma2, sigma3, sigma4, sigma5, sigma6]
    for i in range(len(sigma)):
        sigma[i] = cv2.blur(sigma[i], (winSize, winSize))
    D = sigma[1]**2 - sigma[0]*sigma[4]
    D[np.abs(D) < threshD] = np.Inf
    D = 1.0/D
    hx = (sigma[4]*sigma[2]-sigma[1]*sigma[5])*D
    hy = (sigma[1]*sigma[2]-sigma[0]*sigma[5])*D
    arrow = np.sqrt(hx**2+hy**2)
    return hx, -hy, arrow

def getCoor(hx, hy, arrow, step = 20, percent = 0.1):
    y, x= np.meshgrid(np.arange(0, hx.shape[0], step), np.arange(0, hx.shape[1], step))
    coor = np.vstack((x.flatten(), y.flatten()))
    arrow = arrow[y, x]
    index = np.argsort(arrow.flatten())
    coor = coor[:, index[int(-len(index)*percent):]]
    return coor

def drawArrow(img, hx, hy, coor, scale=50):
    color = (0, 255, 255)
    mask = np.zeros_like(img)
    m = np.max((np.abs(hx), np.abs(hy)))
    if m < 0.5:
        scale /=0.5
    for i in range(coor.shape[1]):
        x1 = int(coor[0, i])
        y1 = int(coor[1, i])
        x2 = int(coor[0, i]+hx[y1, x1]*scale)
        y2 = int(coor[1, i]+hy[y1, x1]*scale)
        mask = cv2.line(mask, (x1, y1), (x2, y2), color, thickness=1)
        mask = cv2.circle(mask,(x1,y1), 3, color, -1)
    out = cv2.add(img, mask)
    return out

def test():
    prevName = "tsukuba_l.png"; nextName = "tsukuba_r.png"
    prevImg = cv2.imread(prevName)
    nextImg = cv2.imread(nextName)
    prevGray = cv2.cvtColor(prevImg, cv2.COLOR_BGR2GRAY)
    nextGray = cv2.cvtColor(nextImg, cv2.COLOR_BGR2GRAY)
    hx, hy, arrow= calcOpticalFlowLK(prevGray, nextGray)
    print("arrow_max = {}".format(np.max(arrow)))
    coor = getCoor(hx, hy, arrow, step=10, percent=0.2)
    out = drawArrow(prevImg, hx, hy, coor)
    plt.hist(arrow.ravel(), 1000, [0, 1]); plt.show()
    while 1:
        cv2.imshow('frame',out)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

def run(prevImg, nextImg):
    prevGray = cv2.cvtColor(prevImg, cv2.COLOR_BGR2GRAY)
    nextGray = cv2.cvtColor(nextImg, cv2.COLOR_BGR2GRAY)
    hx, hy, arrow= calcOpticalFlowLK(prevGray, nextGray)
    coor = getCoor(hx, hy, arrow)
    out = drawArrow(prevImg, hx, hy, coor)
    return out

def main(folder_name):
    print(folder_name)
    folder_in = './eval-data/'+folder_name+'/'
    assert(os.path.exists(folder_in))
    folder_out = './result/'+folder_name+'/'
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)
    files = os.listdir(folder_in)
    files.sort()
    for i in range(len(files)-1):
        prevImg = cv2.imread(folder_in+files[i])
        nextImg = cv2.imread(folder_in+files[i+1])
        out = run(prevImg, nextImg)
        cv2.imwrite(folder_out+str(i)+'.png', out)

def video(infile, outfile = 'result.avi'):
    assert os.path.exists(infile), "video doesn't exist"
    cap = cv2.VideoCapture(infile)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out_video = cv2.VideoWriter(outfile, fourcc, fps, (int(width), int(height)))
    _, old_frame = cap.read()
    while 1:
        ret, frame = cap.read()
        if ret:
            out = run(old_frame, frame)
            cv2.imshow('frame', out)
            out_video.write(out)
        else: break
    cv2.destroyAllWindows()
    out_video.release()
    cap.release()

def camera(filename = 'camera.avi'):
    cap = cv2.VideoCapture(0)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out_video = cv2.VideoWriter(filename, fourcc, 12, (int(width), int(height)))
    recording = False
    dT = 0; startT = 0; cnt = 0
    _, old_frame = cap.read()
    while(1):
        ret,frame = cap.read()
        if ret :
            out = run(old_frame, frame)
            if(recording):
                cnt += 1
                out_video.write(out)
                out = cv2.putText(out, "REC.", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 5)
            cv2.imshow('frame',out)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            elif k == 32:
                if recording:
                    dT += time.time() - startT
                else:
                    startT = time.time()
                recording = not recording
            old_frame = frame.copy()
        else : break
    fps = np.floor(cnt/dT)
    out_video.set(cv2.CAP_PROP_FPS, fps)
    cv2.destroyAllWindows()
    out_video.release()
    cap.release()

if __name__ == '__main__':
    if (len(sys.argv)<=1):
        print('Please input with python: "test", "eval-data", or "camera"')
    else:
        if (sys.argv[1] == 'test'):
            test()
        elif (sys.argv[1] == 'eval-data'):
            assert os.path.exists('./eval-data'), "eval-data doesn't exist"
            if(not os.path.exists('./result')):
                os.mkdir('./result')
            for folder_name in os.listdir('./eval-data'):
                main(folder_name)
        elif (sys.argv[1] == 'video'):
            filename = 'data.avi'
            if(len(sys.argv) >= 3):
                filename = sys.argv[2]
            video(filename)
        elif (sys.argv[1] == 'camera'):
            camera()