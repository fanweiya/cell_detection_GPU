import os
import tkinter
from tkinter import filedialog
from imutils.paths import list_images
import cv2
import random
from PIL import Image
from scipy.stats import norm
import numpy as np
def mergePIL(LT,RT,LB,RB,drawpaer_w,drawpare_h):
    imgae1 = Image.open(LT)
    imgae2 = Image.open(RT)
    imgae3 = Image.open(LB)
    imgae4 = Image.open(RB)
    w1, h1 = imgae1.size
    w2, h2 = imgae2.size
    w3, h3 = imgae3.size
    w4, h4 = imgae4.size
    target_w = drawpaer_w
    target_h = drawpare_h
    zj_x=int(drawpaer_w/2)
    zj_y=int(drawpare_h/2)
    target1 = Image.new('RGB', (target_w, target_h))
    target2 = Image.new('RGB', (target_w, target_h))
    target3 = Image.new('RGB', (target_w, target_h))
    target4 = Image.new('RGB', (target_w, target_h))
    target5 = Image.new('RGB', (target_w, target_h))
    target1.paste(imgae1, (0, 0, w1, h1))
    target2.paste(imgae2, (target_w - w2, 0, target_w, h2))
    target3.paste(imgae3, (0, target_h - h3, w3, target_h))
    target4.paste(imgae4, (target_w - w4, target_h - h4, target_w, target_h))
    target5.paste(target1.crop([0,0,zj_x,zj_y]),(0, 0, zj_x, zj_y))
    target5.paste(target2.crop([zj_x,0,target_w,zj_y]),(zj_x, 0, target_w, zj_y))
    target5.paste(target3.crop([0,zj_y,zj_x,target_h]),(0, zj_y, zj_x, target_h))
    target5.paste(target4.crop([zj_x,zj_y,target_w,target_h]),(zj_x, zj_y, target_w, target_h))
    im3 = Image.blend(target1, target2, 0.5)
    im4 = Image.blend(target3, target4, 0.5)
    im5 = Image.blend(im3, im4, 0.5)
    #target5.show()
    return im5,target5
def getWeightsMatrix(images,isColorMode):
    '''
    功能：获取权值矩阵
    :param images:  输入两个相同区域的图像
    :return: weigthA,weightB
    '''
    (imageA, imageB) = images
    weightMatA = np.ones(imageA.shape, dtype=np.float32)
    weightMatB = np.ones(imageA.shape, dtype=np.float32)
    row, col = imageA.shape[:2]
    weightMatB_1 = weightMatB.copy()
    weightMatB_2 = weightMatB.copy()
    # 获取四条线的相加和，判断属于哪种模式
    compareList = []
    compareList.append(np.count_nonzero(imageA[0: row // 2, 0: col // 2] > 0))
    compareList.append(np.count_nonzero(imageA[row // 2: row, 0: col // 2] > 0))
    compareList.append(np.count_nonzero(imageA[row // 2: row, col // 2: col] > 0))
    compareList.append(np.count_nonzero(imageA[0: row // 2, col // 2: col] > 0))
    # self.printAndWrite("    compareList:" + str(compareList))
    index = compareList.index(min(compareList))
    # print("index:", index)
    if index == 2:
        # 重合区域在imageA的上左部分
        # self.printAndWrite("上左")
        rowIndex = 0
        colIndex = 0
        for j in range(1, col):
            for i in range(row - 1, -1, -1):
                # tempSum = imageA[i, col - j].sum()
                if (isColorMode and imageA[i, col - j].sum() != -3) or (
                        isColorMode is False and imageA[i, col - j] != -1):
                    # if imageA[i, col - j] != -1:
                    rowIndex = i + 1
                    break
            if rowIndex != 0:
                break
        for i in range(col - 1, -1, -1):
            # tempSum = imageA[rowIndex, i].sum()
            if (isColorMode and imageA[rowIndex, i].sum() != -3) or (
                    isColorMode is False and imageA[rowIndex, i] != -1):
                # if imageA[rowIndex, i] != -1:
                colIndex = i + 1
                break
        # 赋值
        for i in range(rowIndex + 1):
            if rowIndex == 0:
                rowIndex = 1
            weightMatB_1[rowIndex - i, :] = (rowIndex - i) * 1 / rowIndex
        for i in range(colIndex + 1):
            if colIndex == 0:
                colIndex = 1
            weightMatB_2[:, colIndex - i] = (colIndex - i) * 1 / colIndex
        weightMatB = weightMatB_1 * weightMatB_2
        weightMatA = 1 - weightMatB
    # elif leftCenter != 0 and bottomCenter != 0 and upCenter == 0 and rightCenter == 0:
    elif index == 3:
        # 重合区域在imageA的下左部分
        # self.printAndWrite("下左")
        rowIndex = 0
        colIndex = 0
        for j in range(1, col):
            for i in range(row):
                # tempSum = imageA[i, col - j].sum()
                if (isColorMode and imageA[i, col - j].sum() != -3) or (
                        isColorMode is False and imageA[i, col - j] != -1):
                    # if imageA[i, col - j] != -1:
                    rowIndex = i - 1
                    break
            if rowIndex != 0:
                break
        for i in range(col - 1, -1, -1):
            # tempSum = imageA[rowIndex, i].sum()
            if (isColorMode and imageA[rowIndex, i].sum() != -3) or (
                    isColorMode is False and imageA[rowIndex, i] != -1):
                # if imageA[rowIndex, i] != -1:
                colIndex = i + 1
                break
        # 赋值
        for i in range(rowIndex, row):
            if rowIndex == 0:
                rowIndex = 1
            weightMatB_1[i, :] = (row - i - 1) * 1 / (row - rowIndex - 1)
        for i in range(colIndex + 1):
            if colIndex == 0:
                colIndex = 1
            weightMatB_2[:, colIndex - i] = (colIndex - i) * 1 / colIndex
        weightMatB = weightMatB_1 * weightMatB_2
        weightMatA = 1 - weightMatB
    # elif rightCenter != 0 and bottomCenter != 0 and upCenter == 0 and leftCenter == 0:
    elif index == 0:
        # 重合区域在imageA的下右部分
        # self.printAndWrite("下右")
        rowIndex = 0
        colIndex = 0
        for j in range(0, col):
            for i in range(row):
                # tempSum = imageA[i, j].sum()
                if (isColorMode and imageA[i, j].sum() != -3) or (
                        isColorMode is False and imageA[i, j] != -1):
                    # if imageA[i, j] != -1:
                    rowIndex = i - 1
                    break
            if rowIndex != 0:
                break
        for i in range(col):
            # tempSum = imageA[rowIndex, i].sum()
            if (isColorMode and imageA[rowIndex, i].sum() != -3) or (
                    isColorMode is False and imageA[rowIndex, i] != -1):
                # if imageA[rowIndex, i] != -1:
                colIndex = i - 1
                break
        # 赋值
        for i in range(rowIndex, row):
            if rowIndex == 0:
                rowIndex = 1
            weightMatB_1[i, :] = (row - i - 1) * 1 / (row - rowIndex - 1)
        for i in range(colIndex, col):
            if colIndex == 0:
                colIndex = 1
            weightMatB_2[:, i] = (col - i - 1) * 1 / (col - colIndex - 1)
        weightMatB = weightMatB_1 * weightMatB_2
        weightMatA = 1 - weightMatB
    # elif upCenter != 0 and rightCenter != 0 and leftCenter == 0 and bottomCenter == 0:
    elif index == 1:
        # 重合区域在imageA的上右部分
        # self.printAndWrite("上右")
        rowIndex = 0
        colIndex = 0
        for j in range(0, col):
            for i in range(row - 1, -1, -1):
                # tempSum = imageA[i, j].sum()
                if (isColorMode and imageA[i, j].sum() != -3) or (
                        (isColorMode is False) and imageA[i, j] != -1):
                    rowIndex = i + 1
                    break
            if rowIndex != 0:
                break
        for i in range(col):
            # tempSum = imageA[rowIndex, i].sum()
            if (isColorMode and imageA[rowIndex, i].sum() != -3) or (
                    (isColorMode is False) and imageA[rowIndex, i] != -1):
                colIndex = i - 1
                break
        for i in range(rowIndex + 1):
            if rowIndex == 0:
                rowIndex = 1
            weightMatB_1[rowIndex - i, :] = (rowIndex - i) * 1 / rowIndex
        for i in range(colIndex, col):
            if colIndex == 0:
                colIndex = 1
            weightMatB_2[:, i] = (col - i - 1) * 1 / (col - colIndex - 1)
        weightMatB = weightMatB_1 * weightMatB_2
        weightMatA = 1 - weightMatB
    # print(weightMatA)
    # print(weightMatB)
    return (weightMatA, weightMatB)

def fuseByFadeInAndFadeOut(images, dx, dy,isColorMode):
    '''
    功能：渐入渐出融合
    :param images:输入两个相同区域的图像
    :param direction: 横向拼接还是纵向拼接
    :return:融合后的图像
    '''
    # print("dx=", dx, "dy=", dy)
    (imageA, imageB) = images
    # cv2.imshow("A", imageA.astype(np.uint8))
    # cv2.imshow("B", imageB.astype(np.uint8))
    # cv2.waitKey(0)
    # self.printAndWrite("dx={}, dy={}".format(dx, dy))
    row, col = imageA.shape[:2]
    weightMatA = np.ones(imageA.shape, dtype=np.float32)
    weightMatB = np.ones(imageA.shape, dtype=np.float32)
    # self.printAndWrite("    ratio: "  + str(np.count_nonzero(imageA > -1) / imageA.size))
    if np.count_nonzero(imageA > -1) / imageA.size > 0.65:
        # self.printAndWrite("直接融合")
        # 如果对于imageA中，非0值占比例比较大，则认为是普通融合
        # 根据区域的行列大小来判断，如果行数大于列数，是水平方向
        if col <= row:
            # self.printAndWrite("普通融合-水平方向")
            for i in range(0, col):
                if dy >= 0:
                    weightMatA[:, col - i - 1] = weightMatA[:, col - i - 1] * i * 1.0 / col
                    weightMatB[:, i] = weightMatB[:, i] * i * 1.0 / col
                    # weightMatA[:, i] = weightMatA[:, i] * i * 1.0 / col
                    # weightMatB[:, col - i - 1] = weightMatB[:, col - i - 1] * i * 1.0 / col
                elif dy < 0:
                    weightMatA[:, col - i - 1] = weightMatA[:, col - i - 1] * (col - i) * 1.0 / col
                    weightMatB[:, i] = weightMatB[:, i] * (col - i) * 1.0 / col
                    # weightMatA[:, i] = weightMatA[:, i] * (col - i) * 1.0 / col
                    # weightMatB[:, col - i - 1] = weightMatB[:, col - i - 1] * (col - i) * 1.0 / col
        # 根据区域的行列大小来判断，如果列数大于行数，是竖直方向
        elif row < col:
            # self.printAndWrite("普通融合-竖直方向")
            for i in range(0, row):
                if dx <= 0:
                    weightMatA[i, :] = weightMatA[i, :] * i * 1.0 / row
                    weightMatB[row - i - 1, :] = weightMatB[row - i - 1, :] * i * 1.0 / row
                elif dx > 0:
                    weightMatA[i, :] = weightMatA[i, :] * (row - i) * 1.0 / row
                    weightMatB[row - i - 1, :] = weightMatB[row - i - 1, :] * (row - i) * 1.0 / row
    else:
        # 如果对于imageA中，非0值占比例比较小，则认为是拐角融合
        weightMatA, weightMatB = getWeightsMatrix(images,isColorMode)
    imageA[imageA < 0] = imageB[imageA < 0]
    result = weightMatA * imageA.astype(int) + weightMatB * imageB.astype(int)
    result[result < 0] = 0
    result[result > 255] = 255
    fuseRegion = np.uint8(result)
    return fuseRegion
def fuseImage(images, dx, dy,isColorMode):
    """
    功能：融合图像
    :param images: [imageA, imageB]
    :param dx: x方向偏移量
    :param dy: y方向偏移量
    :return:
    """
    fuseRegion = fuseByFadeInAndFadeOut(images, dx, dy,isColorMode)
    return fuseRegion

def resizeimg(img):
    result=cv2.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
    return result

def left_rigit_fuse(draw_H,draw_W,img1,img2,iscolermode):
    stitchResult = np.ones((draw_H, draw_W, 3), dtype="uint8")
    h, w, _ = img1.shape
    stitchResult[0:h,0:w,:]=img1
    roi1=stitchResult[:,draw_W-w:w,:].copy()
    stitchResult[0:h,draw_W-w:draw_W,:]=img2
    roi2=stitchResult[:,draw_W-w:w,:]
    test=fuseImage([roi1,roi2],0,draw_W-w,iscolermode)
    stitchResult[:,draw_W-w:w,:]=test
    return stitchResult

def top_botem_fuse(draw_H,draw_W,img1,img2,iscolermode):
    stitchResult = np.ones((draw_H, draw_W, 3), dtype="uint8")
    h, w, _ = img1.shape
    stitchResult[0:h,0:w,:]=img1
    roi1=stitchResult[draw_H-h:h,:,:].copy()
    stitchResult[draw_H-h:draw_H,:,:]=img2
    roi2=stitchResult[draw_H-h:h,:,:]
    test=fuseImage([roi1,roi2],draw_H-h,0,iscolermode)
    stitchResult[draw_H-h:h,:,:]=test
    return stitchResult

def left_rigit_fuse_center(draw_H,draw_W,img1,img2,overlap_thred,iscloermode=True):
    stitchResult = np.ones((draw_H, draw_W, 3), dtype="uint8")
    h, w, _ = img1.shape
    center_w=int(draw_W/2)
    stitchResult[0:h,0:center_w,:]=img1[0:h,0:center_w,:]

    roi1=img1[:,center_w-overlap_thred:center_w+overlap_thred,:]
    stitchResult[0:h,center_w:draw_W,:]=img2[0:h,w-center_w:w,:]
    roi2=img2[:,w-center_w-overlap_thred:w-center_w+overlap_thred,:]
    test=fuseImage([roi1,roi2],0,draw_W-w+overlap_thred,iscloermode)
    stitchResult[:,center_w-overlap_thred:center_w+overlap_thred,:]=test
    return stitchResult

def top_botem_fuse_center(draw_H,draw_W,img1,img2,overlap_thred,iscloermode=True):
    stitchResult = np.ones((draw_H, draw_W, 3), dtype="uint8")
    h, w, _ = img1.shape
    center_h=int(draw_H/2)
    stitchResult[0:h,0:w,:]=img1
    while h-center_h-overlap_thred<0:
        overlap_thred=overlap_thred-1
    roi1 = img1[center_h - overlap_thred:center_h + overlap_thred, :, :]
    stitchResult[center_h:draw_H, 0:w, :] = img2[h - center_h:h, :, :]
    roi2 = img2[h - center_h - overlap_thred:h - center_h + overlap_thred, :, :]
    #print(overlap_thred,roi1.shape,roi2.shape)
    print("overlap_thred use %i"%overlap_thred)
    test=fuseImage([roi1,roi2],draw_H-h+overlap_thred,0,iscloermode)
    stitchResult[center_h-overlap_thred:center_h+overlap_thred,:,:]=test
    return stitchResult

def xiaofeng(LT,RT,LB,RB,drawpaer_w,drawpare_h,iscolermode=True):
    LTimg= cv2.imdecode(np.fromfile(LT, dtype=np.uint8), cv2.IMREAD_COLOR)
    RTimg= cv2.imdecode(np.fromfile(RT, dtype=np.uint8), cv2.IMREAD_COLOR)
    LBimg= cv2.imdecode(np.fromfile(LB, dtype=np.uint8), cv2.IMREAD_COLOR)
    RBimg= cv2.imdecode(np.fromfile(RB, dtype=np.uint8), cv2.IMREAD_COLOR)
    Th=max(LTimg.shape[0],RTimg.shape[0])
    Bh=max(LBimg.shape[0],RBimg.shape[0])
    TLR=left_rigit_fuse(Th,drawpaer_w,LTimg,RTimg,iscolermode)
    BLR=left_rigit_fuse(Bh,drawpaer_w,LBimg,RBimg,iscolermode)
    result=top_botem_fuse(drawpare_h,drawpaer_w,TLR,BLR,iscolermode)
    return result

def xiaofeng_center(LT,RT,LB,RB,drawpaer_w,drawpare_h,ovelap_thread,iscolermode=True):
    LTimg= cv2.imdecode(np.fromfile(LT, dtype=np.uint8), cv2.IMREAD_COLOR)
    RTimg= cv2.imdecode(np.fromfile(RT, dtype=np.uint8), cv2.IMREAD_COLOR)
    LBimg= cv2.imdecode(np.fromfile(LB, dtype=np.uint8), cv2.IMREAD_COLOR)
    RBimg= cv2.imdecode(np.fromfile(RB, dtype=np.uint8), cv2.IMREAD_COLOR)
    Th=max(LTimg.shape[0],RTimg.shape[0])
    Bh=max(LBimg.shape[0],RBimg.shape[0])
    TLR=left_rigit_fuse_center(Th,drawpaer_w,LTimg,RTimg,ovelap_thread,iscolermode)
    BLR=left_rigit_fuse_center(Bh,drawpaer_w,LBimg,RBimg,ovelap_thread,iscolermode)
    result=top_botem_fuse_center(drawpare_h,drawpaer_w,TLR,BLR,ovelap_thread,iscolermode)
    return result

##投射灯效果（提亮度）
def generate_spot_light_mask(mask_size,
                             position=None,
                             max_brightness=255,
                             min_brightness=0,
                             mode="gaussian",
                             linear_decay_rate=None,
                             speedup=False):
    """
    Generate decayed light mask generated by spot light given position, direction. Multiple spotlights are accepted.
    Args:
        mask_size: tuple of integers (w, h) defining generated mask size
        position: list of tuple of integers (x, y) defining the center of spotlight light position,
                  which is the reference point during rotating
        max_brightness: integer that max brightness in the mask
        min_brightness: integer that min brightness in the mask
        mode: the way that brightness decay from max to min: linear or gaussian
        linear_decay_rate: only valid in linear_static mode. Suggested value is within [0.2, 2]
        speedup: use `shrinkage then expansion` strategy to speed up vale calculation
    Return:
        light_mask: ndarray in float type consisting value from max_brightness to min_brightness. If in 'linear' mode
                    minimum value could be smaller than given min_brightness.
    """
    if position is None:
        position = [(int(mask_size[0]/2), int(mask_size[1]/2))]

    if linear_decay_rate is None:
        if mode == "linear_static":
            linear_decay_rate = random.uniform(0.25, 1)
    assert mode in ["linear_static", "gaussian"], \
        "mode must be linear_dynamic, linear_static or gaussian"
    mask = np.zeros(shape=(mask_size[1], mask_size[0]), dtype=np.float32)
    if mode == "gaussian":
        mu = np.sqrt(mask.shape[0]**2+mask.shape[1]**2)
        dev = mu / 3.5
        mask = _decay_value_radically_norm_in_matrix(mask_size, position, max_brightness, min_brightness, dev)
    mask = np.asarray(mask, dtype=np.uint8)
    # add median blur
    mask = cv2.medianBlur(mask, 5)
    mask = 255 - mask
    return mask
def _decay_value_radically_norm_in_matrix(mask_size, centers, max_value, min_value, dev):
    """
    _decay_value_radically_norm function in matrix format
    """
    center_prob = norm.pdf(0, 0, dev)
    x_value_rate = np.zeros((mask_size[1], mask_size[0]))
    for center in centers:
        coord_x = np.arange(mask_size[0])
        coord_y = np.arange(mask_size[1])
        xv, yv = np.meshgrid(coord_x, coord_y)
        dist_x = xv - center[0]
        dist_y = yv - center[1]
        dist = np.sqrt(np.power(dist_x, 2) + np.power(dist_y, 2))
        x_value_rate += norm.pdf(dist, 0, dev) / center_prob
    mask = x_value_rate * (max_value - min_value) + min_value
    mask[mask > 255] = 255
    return mask
def add_spot_light(frame, light_position=None, max_brightness=255, min_brightness=0,
                   mode='gaussian', linear_decay_rate=None, transparency=0.5):
    """
    Add mask generated from spot light to given image
    """
    # if transparency is None:
    #     transparency = random.uniform(0.5, 0.85)
    # transparency=0.6
    #frame = cv2.imread(image)
    height, width, _ = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = generate_spot_light_mask(mask_size=(width, height),
                                    position=light_position,
                                    max_brightness=max_brightness,
                                    min_brightness=min_brightness,
                                    mode=mode,
                                    linear_decay_rate=linear_decay_rate)
    hsv[:, :, 2] = hsv[:, :, 2]*(1-transparency+0.2) + mask*transparency
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    frame[frame > 255] = 255
    frame = np.asarray(frame, dtype=np.uint8)
    return frame
def hard_stitching(stitching_type,LT,RT,LB,RB,image_save_path,drawpaer_w,drawpare_h,light_thread):
    if light_thread != 0:
        if stitching_type =="smoothing_stitching":
            xiaofengresult = xiaofeng(LT, RT, LB, RB, drawpaer_w, drawpare_h, True)
            cv2.imencode('.jpg', add_spot_light(xiaofengresult, transparency=light_thread))[1].tofile(image_save_path)
        else:
            output1, output2 = mergePIL(LT, RT, LB, RB, drawpaer_w, drawpare_h)
            cv2.imencode('.jpg', add_spot_light(cv2.cvtColor(np.asarray(output2), cv2.COLOR_RGB2BGR),
                                                transparency=light_thread))[1].tofile(image_save_path)
            # xiaofengresult2 = xiaofeng_center(LT, RT, LB, RB, drawpaer_w, drawpare_h, ovlapthred, True)
            # cv2.imencode('.jpg', add_spot_light(xiaofengresult2, transparency=light_thread))[1].tofile(result_path4)
    else:
        if stitching_type=="smoothing_stitching":
            xiaofengresult = xiaofeng(LT, RT, LB, RB, drawpaer_w, drawpare_h, True)
            cv2.imencode('.jpg', xiaofengresult)[1].tofile(image_save_path)
        else:
            output1, output2 = mergePIL(LT, RT, LB, RB, drawpaer_w, drawpare_h)
            output2.save(image_save_path)
            #cv2.imencode('.jpg', xiaofengresult2)[1].tofile(result_path4)
if __name__ == '__main__':
    originpath = tkinter.filedialog.askdirectory(title="请选择要处理的文件夹路径")
    savepath = tkinter.filedialog.askdirectory(title="请选择保存路径")
    drawpaer_w=int(input("请输入画布的宽度"))
    drawpare_h=int(input("请输入画布的长度"))
    ovlapthred=int(input("请输入需要消缝宽度(中线两边各留多少像素)"))
    light_thread=float(input("请输入四角亮度强度值，默认值0.5,输入0为不启用该功能"))
    # originpath=r"H:\B项目\111"
    # savepath=r"H:\test"
    # drawpaer_w=6614
    # drawpare_h=5378
    # ovlapthred=150
    # light_thread=0
    if drawpare_h%2!=0 or drawpaer_w%2!=0:
        print("宽度或长度不是偶数，请重新输入")
        drawpaer_w = int(input("请输入画布的宽度"))
        drawpare_h = int(input("请输入画布的长度"))
    for idir in ['右上', '右下', '左上', '左下']:
        if idir not in os.listdir(originpath):
            print("%s文件夹不存在",idir)
    for LT in list(list_images(os.path.join(originpath,"左上"))):
        if max(Image.open(LT).size)>max(drawpaer_w,drawpare_h):
            print("图片大小超过画布大小")
        else:
            print(LT,drawpaer_w,drawpare_h)
            RT=LT.replace("左上","右上")
            LB=LT.replace("左上","左下")
            RB=LT.replace("左上","右下")
            result_path1=os.path.join(savepath,LT.split(os.sep)[-1][:-4]+"_blend.jpg")
            result_path2 = os.path.join(savepath, LT.split(os.sep)[-1][:-4] +"_stitching.jpg")
            result_path3 = os.path.join(savepath, LT.split(os.sep)[-1][:-4] +"_xiaofeng_pianyi.jpg")
            result_path4 = os.path.join(savepath, LT.split(os.sep)[-1][:-4] +"_xiaofeng_center.jpg")
            output1,output2=mergePIL(LT,RT,LB,RB,drawpaer_w,drawpare_h)
            xiaofengresult = xiaofeng(LT, RT, LB, RB, drawpaer_w, drawpare_h,True)
            xiaofengresult2=xiaofeng_center(LT,RT,LB,RB,drawpaer_w,drawpare_h,ovlapthred,True)
            if light_thread!=0:
                output1.save(result_path1)
                cv2.imencode('.jpg', add_spot_light(cv2.cvtColor(np.asarray(output2), cv2.COLOR_RGB2BGR),transparency=light_thread))[1].tofile(result_path2)
                cv2.imencode('.jpg', add_spot_light(xiaofengresult,transparency=light_thread))[1].tofile(result_path3)
                cv2.imencode('.jpg', add_spot_light(xiaofengresult2,transparency=light_thread))[1].tofile(result_path4)
            else:
                output1.save(result_path1)
                output2.save(result_path2)
                cv2.imencode('.jpg', xiaofengresult)[1].tofile(result_path3)
                cv2.imencode('.jpg', xiaofengresult2)[1].tofile(result_path4)


# if __name__ == '__main__':
    # img1=cv2.imread("A_1.jpg")
    # img2=cv2.imread("A_2.jpg")
    # img3=cv2.imread("A_3.jpg")
    # img4=cv2.imread("A_4.jpg")
    # # img2=cv2.imread("A_2.jpg")
    # TLR=left_rigit_fuse(2808,6534,img1,img2)
    # BLR=left_rigit_fuse(2808,6534,img3,img4)
    # result=top_botem_fuse(5290,6534,TLR,BLR)
    # cv2.imwrite("TLR.jpg",TLR)
    # cv2.imwrite("BLR.jpg",BLR)
    # cv2.imwrite("end.jpg",result)
