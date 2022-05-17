import os
import shutil

import cv2
from imutils.paths import list_images
import numpy as np
import tkinter.filedialog
import _tkinter
def img_read(file_path,chnnel=3):
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), chnnel)
def pianyimerge(mc,fl,px,py):
    # roi move
    # border = max(abs(px), abs(py))
    # img_add = cv2.copyMakeBorder(fl, border, abs(py), border, abs(px),
    #                              cv2.BORDER_CONSTANT,
    #                              value=(0, 0, 0))
    # h1, w1, _ = mc.shape
    # if (py > 0):
    #     if (px < 0):
    #         imgroi = img_add[0:h1 - (border + abs(py)), border + abs(px):w1]
    #     else:
    #         imgroi = img_add[0:h1 - (border + abs(py)), 0:w1 - (border + abs(px))]
    # elif (px < 0):
    #     imgroi = img_add[border + abs(py):h1, border + abs(px):w1]
    # else:
    #     imgroi = img_add[border + abs(py):h1, 0:w1 - (border + abs(px))]
    #
    # warpAffine
    h1, w1, _ = mc.shape
    print(mc.shape, fl.shape,px,py)
    M = np.float32([[1, 0, -px],[0, 1, -py]])
    imgroi = cv2.warpAffine(fl, M, (w1, h1))
    return cv2.addWeighted(src1=mc, alpha=0.5, src2=imgroi, beta=0.5, gamma=0)

def pianyimergeadd(mc,fl1,fl2,px1,py1,px2,py2):
    h1, w1, _ = mc.shape
    zeros = np.zeros(mc.shape[:2], dtype="uint8")
    print(mc.shape, fl1.shape,px1,py1,fl2.shape,px2,py2)
    M1 = np.float32([[1, 0, -px1],[0, 1, -py1]])
    M2 = np.float32([[1, 0, -px2],[0, 1, -py2]])
    fl1_warp = cv2.warpAffine(fl1, M1, (w1, h1))
    fl2_warp = cv2.warpAffine(fl2, M2, (w1, h1))
    fl1_split = cv2.split(fl1_warp)[1]
    fl2_split = cv2.split(fl2_warp)[2]
    fl=cv2.merge([zeros, fl1_split, fl2_split])
    return cv2.add(mc, fl)
def pianyimergeadd_single(mc,fl,px,py):
    h1, w1, _ = mc.shape
    print(mc.shape, fl.shape,px,py)
    M = np.float32([[1, 0, -px],[0, 1, -py]])
    fl_warp = cv2.warpAffine(fl, M, (w1, h1))
    return cv2.add(mc, fl_warp)

def danjiupian(mc,fl,px,py):
    h1, w1, _ = mc.shape
    print(mc.shape, fl.shape,px,py)
    M = np.float32([[1, 0, -px],[0, 1, -py]])
    fl_warp = cv2.warpAffine(fl, M, (w1, h1))
    return fl_warp

def maksdir(savepath):
    if not os.path.exists(savepath):
        try:
            print("创建文件夹%s"%savepath)
            os.makedirs(savepath)
        except:
            print("%s is exit"%savepath)
    return 0
def merge_image(BF=None,FL1=None,FL2=None,px1=None,py1=None,px2=None,py2=None,image_save_path=None,merge_type="add"):
    if merge_type=="add":
        if FL1==None:
            result = pianyimergeadd_single(img_read(BF), img_read(FL2), px2, py2)
        elif FL2==None:
            result = pianyimergeadd_single(img_read(BF), img_read(FL1), px1, py1)
        else:
            result = pianyimergeadd(img_read(BF), img_read(FL1), img_read(FL2), px1, py1, px2, py2)
    else:
        if FL1==None:
            result = pianyimerge(img_read(BF), img_read(FL2), px2, py2)
        elif FL2==None:
            result = pianyimerge(img_read(BF), img_read(FL1), px1, py1)
        else:
            BFFL1 = pianyimerge(img_read(BF), img_read(FL1), px1, py1)
            result = pianyimerge(BFFL1, img_read(FL2), px2, py2)

    cv2.imencode('.jpg', result)[1].tofile(image_save_path)

if __name__ == '__main__':

    # BF = img_read(r"H:\20210817182857\明场\右上\A_1.jpg")
    # Fl1 = img_read(r"H:\20210817182857\FL1\右上\A_1.jpg")
    # Fl2 = img_read(r"H:\20210817182857\FL2\右上\A_1.jpg")
    # test=pianyimergeadd(BF,Fl1,Fl2,0,-7,2,-6)
    # BFFL1=pianyimerge(BF,Fl1,0,-7)
    # BFFL12=pianyimerge(BFFL1,Fl2,2,-6)
    # # test=pianyimergeadd_single(BF,Fl1,0,-7)
    # cv2.imwrite("tAAAA.jpg",test)
    # cv2.imwrite("tAAAAC.jpg",BFFL12)

    # originpath =r"H:\20210817182857"
    # savepath = r"H:\test"
    # select = 2
    originpath=tkinter.filedialog.askdirectory(title="请选择要处理的文件夹路径")
    savepath=tkinter.filedialog.askdirectory(title="请选择保存路径")
    select = int(input("请选择融合方式：1:权重融合(addWeighted,默认为0.5) 2:叠加融合(add)     "))
    if "明场" in os.listdir(originpath):
        if "FL1" in os.listdir(originpath) and "FL2" in os.listdir(originpath):
            for childdir in os.listdir(os.path.join(originpath, "明场")):
                px1 = float(input("请输入%s FL1的X偏移量："%childdir))
                py1 = float(input("请输入%s FL1的Y偏移量："%childdir))
                px2 = float(input("请输入%s FL2的X偏移量："%childdir))
                py2 = float(input("请输入%s FL2的Y偏移量："%childdir))
                # px1=px2=0
                # py1=py2=-7
                print("你输入%s FL1的X,Y偏移量为 "%childdir,px1,py1," %s FL2的X,Y偏移量为 "%childdir,px2,py2)
                try:
                    for filepath in list(list_images(os.path.join(originpath, "明场",childdir))):
                        filenamedir=os.path.join(savepath, childdir,filepath.split(os.sep)[-1][:-4])
                        maksdir(filenamedir)
                        print(filenamedir,filepath)
                        BF = img_read(filepath)
                        FL1 = img_read(filepath.replace("明场", "FL1"))
                        FL2 = img_read(filepath.replace("明场", "FL2"))
                        if select==1:
                            BFFL1=pianyimerge(BF,FL1,px1,py1)
                            BFFL12=pianyimerge(BFFL1,FL2,px2,py2)
                            shutil.copy2(filepath,os.path.join(filenamedir,filepath.split(os.sep)[-1][:-4]+"_BF.jpg"))
                            cv2.imencode('.jpg', danjiupian(BF,FL1,px1,py1))[1].tofile(
                                os.path.join(filenamedir, filepath.split(os.sep)[-1][:-4]+"_FL1.jpg"))
                            cv2.imencode('.jpg', danjiupian(BF,FL2,px2,py2))[1].tofile(
                                os.path.join(filenamedir, filepath.split(os.sep)[-1][:-4]+"_FL2.jpg"))
                            cv2.imencode('.jpg', BFFL12)[1].tofile(os.path.join(filenamedir,filepath.split(os.sep)[-1][:-4]+"_merge.jpg"))
                        else:
                            BFFL12=pianyimergeadd(BF,FL1,FL2,px1,py1,px2,py2)
                            shutil.copy2(filepath,os.path.join(filenamedir,filepath.split(os.sep)[-1][:-4]+"_BF.jpg"))
                            cv2.imencode('.jpg', danjiupian(BF,FL1,px1,py1))[1].tofile(
                                os.path.join(filenamedir, filepath.split(os.sep)[-1][:-4]+"_FL1.jpg"))
                            cv2.imencode('.jpg', danjiupian(BF,FL2,px2,py2))[1].tofile(
                                os.path.join(filenamedir, filepath.split(os.sep)[-1][:-4]+"_FL2.jpg"))
                            cv2.imencode('.jpg', BFFL12)[1].tofile(os.path.join(filenamedir,filepath.split(os.sep)[-1][:-4]+"_merge.jpg"))
                except Exception as e:
                    print(e)
        elif "FL1" in os.listdir(originpath):
            print("只存在FL1文件夹")
            for childdir in os.listdir(os.path.join(originpath, "明场")):
                px1 = float(input("请输入%s FL1的X偏移量："%childdir))
                py1 = float(input("请输入%s FL1的Y偏移量："%childdir))
                print("你输入%s FL1的X,Y偏移量为"%childdir,px1,py1)
                try:
                    for filepath in list(list_images(os.path.join(originpath, "明场",childdir))):
                        filenamedir=os.path.join(savepath, childdir,filepath.split(os.sep)[-1][:-4])
                        maksdir(filenamedir)
                        BF = img_read(filepath)
                        FL1 = img_read(filepath.replace("明场", "FL1"))
                        if select==1:
                            BFFL1=pianyimerge(BF,FL1,px1,py1)
                            shutil.copy2(filepath,os.path.join(filenamedir,filepath.split(os.sep)[-1][:-4]+"_BF.jpg"))
                            cv2.imencode('.jpg', danjiupian(BF,FL1,px1,py1))[1].tofile(
                                os.path.join(filenamedir, filepath.split(os.sep)[-1][:-4]+"_FL1.jpg"))
                            cv2.imencode('.jpg', BFFL1)[1].tofile(os.path.join(filenamedir,filepath.split(os.sep)[-1][:-4]+"_merge.jpg"))
                        else:
                            BFFL1=pianyimergeadd_single(BF,FL1,px1,py1)
                            shutil.copy2(filepath,os.path.join(filenamedir,filepath.split(os.sep)[-1][:-4]+"_BF.jpg"))
                            cv2.imencode('.jpg', danjiupian(BF,FL1,px1,py1))[1].tofile(
                                os.path.join(filenamedir, filepath.split(os.sep)[-1][:-4]+"_FL1.jpg"))
                            cv2.imencode('.jpg', BFFL1)[1].tofile(os.path.join(filenamedir,filepath.split(os.sep)[-1][:-4]+"_merge.jpg"))
                except Exception as e:
                    print(e)
        elif "FL2" in os.listdir(originpath):
            print("只存在FL2文件夹")
            for childdir in os.listdir(os.path.join(originpath, "明场")):
                px2 = float(input("请输入%s FL2的X偏移量："%childdir))
                py2 = float(input("请输入%s FL2的Y偏移量："%childdir))
                print("你输入%s FL2的X,Y偏移量为"%childdir, px2, py2)
                try:
                    for filepath in list(list_images(os.path.join(originpath, "明场",childdir))):
                        filenamedir=os.path.join(savepath, childdir,filepath.split(os.sep)[-1][:-4])
                        maksdir(filenamedir)
                        BF = img_read(filepath)
                        FL2 = img_read(filepath.replace("明场", "FL2"))
                        if select==1:
                            BFFL2=pianyimerge(BF,FL2,px2,py2)
                            shutil.copy2(filepath,os.path.join(filenamedir,filepath.split(os.sep)[-1][:-4]+"_BF.jpg"))
                            cv2.imencode('.jpg', danjiupian(BF,FL2,px2,py2))[1].tofile(
                                os.path.join(filenamedir, filepath.split(os.sep)[-1][:-4]+"_FL2.jpg"))
                            cv2.imencode('.jpg', BFFL2)[1].tofile(os.path.join(filenamedir,filepath.split(os.sep)[-1][:-4]+"_merge.jpg"))
                        else:
                            BFFL2=pianyimergeadd_single(BF,FL2,px2,py2)
                            shutil.copy2(filepath,os.path.join(filenamedir,filepath.split(os.sep)[-1][:-4]+"_BF.jpg"))
                            cv2.imencode('.jpg', danjiupian(BF,FL2,px2,py2))[1].tofile(
                                os.path.join(filenamedir, filepath.split(os.sep)[-1][:-4]+"_FL2.jpg"))
                            cv2.imencode('.jpg', BFFL2)[1].tofile(os.path.join(filenamedir,filepath.split(os.sep)[-1][:-4]+"_merge.jpg"))
                except Exception as e:
                    print(e)
        else:
            print("没有发现FL1,FL2文件夹")
    else:
        print("没有发现明场文件夹")




