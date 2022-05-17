import math
import os
import sys
import time
from pathlib import Path

import cv2
import imutils
import numpy as np
from lsnms import nms
from sahi.slicing import slice_image

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())
from scipy.spatial.distance import pdist, squareform
from utils.general import scale_coords, letterbox, \
    get_centerpoint, plot_cricle, \
    convert_from_image_to_cv2, convert_from_cv2_to_image, use_lnms_nms,save_resultimage
from openvino.inference_engine import IECore
from utils.general import cell_struct,cell_list,channel_struct
def get_openvino_core_net_exec(model_xml_path, target_device="CPU"):
    if not os.path.isfile(model_xml_path):
        print(f'{model_xml_path} does not exist')
        return None
    model_bin_path = Path(model_xml_path).with_suffix('.bin').as_posix()
    # load IECore object
    OVIE = IECore()
    # load CPU extensions if availabel
    # lib_ext_path = '/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension.so'
    # if 'CPU' in target_device and os.path.exists(lib_ext_path):
    #     print(f"Loading CPU extensions from {lib_ext_path}")
    #     OVIE.add_extension(lib_ext_path, "CPU")
    # load openVINO network
    OVNet = OVIE.read_network(model=model_xml_path, weights=model_bin_path)
    # create executable network
    # if "GPU" in OVIE.available_devices:
    #     print(OVIE.available_devices)
    #     target_device="GPU"
    #     print("Use GPU")
    OVExec = OVIE.load_network(network=OVNet, device_name=target_device)
    return OVIE, OVNet, OVExec

def get_prediction(image,
    detection_model,
    image_size: int = None,
    shift_amount=None,
    conf_thres: float = 0.5,  # confidence threshold
    iou_thres: float = 0.45,  # NMS IOU threshold
    full_shape=None,
    usedevice="CUDA"):
    if shift_amount is None:
        shift_amount = [0, 0]
    img = letterbox(convert_from_image_to_cv2(image), image_size, stride=64, auto=False)[0]
    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = img.astype('float32')
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    if usedevice == "CUDA":
        detections = detection_model.infer(img)
    else:
        results = detection_model["OVExec"].infer(inputs={detection_model["InputLayer"]: img})
        detections = results[detection_model["OutputLayer"]]
    det=use_lnms_nms(detections,conf_thres=conf_thres,iou_thres=iou_thres,cutoff_distance=8)
    if len(det):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
        det[:, 0] = det[:, 0] + shift_amount[0]
        det[:, 1] = det[:, 1] + shift_amount[1]
        det[:, 2] = det[:, 2] + shift_amount[0]
        det[:, 3] = det[:, 3] + shift_amount[1]
        return det
    return None
def load_ovdet_models(weights,device="CPU"):
    OVIE, OVNet, OVExec = get_openvino_core_net_exec(weights, device)
    InputLayer = next(iter(OVNet.input_info))
    OutputLayer = list(OVNet.outputs)[-1]
    # print("Available Devices: ", OVIE.available_devices)
    # print("Input Layer: ", InputLayer)
    # print("Output Layer: ", OutputLayer)
    print(weights.split("/")[-2]," Model Input Shape: ", OVNet.input_info[InputLayer].input_data.shape)
    print(weights.split("/")[-2]," Model Output Shape: ", OVNet.outputs[OutputLayer].shape)
    imgsz = max(OVNet.input_info[InputLayer].input_data.shape)
    return {
        "OVExec": OVExec,
        "InputLayer": InputLayer,
        "OutputLayer": OutputLayer
    }
def warm_loadnms():
    s_time=time.time()
    # Create boxes: approx 30 pixels wide / high
    image_size = 10_00
    n_predictions = 10_0
    topleft = np.random.uniform(0.0, high=image_size, size=(n_predictions, 2))
    wh = np.random.uniform(15, 45, size=topleft.shape)
    boxes = np.concatenate([topleft, topleft + wh], axis=1).astype(np.float64)
    # Create scores
    scores = np.random.uniform(0., 1., size=(len(boxes),))
    # Apply NMS
    # During the process, overlapping boxes are queried using a R-Tree, ensuring a log-time search
    keep = nms(boxes, scores, iou_threshold=0.5)
    print("warm load nms cost %.2f s"%(time.time()-s_time))

def compute_type(tongdao,group_type,cellx,celly,max_aix,min_aix,image,xyxy_0,xyxy_1,xyxy_2,xyxy_3):
    img=image[int(xyxy_1):int(xyxy_3), int(xyxy_0):int(xyxy_2)]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_crop=crop_box(image,xyxy_0,xyxy_1,xyxy_2,xyxy_3,pading=3)
    crop_gray=cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
    liangdu = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,2])
    sharpne = cv2.Laplacian(gray, cv2.CV_64F).var()
    # lower_green = np.array([10, 25, 25])
    # upper_green = np.array([99, 255, 255])
    # mask_G = cv2.inRange(hsv, lower_green, upper_green)
    # lower_red = np.array([0, 25, 25])
    # upper_red = np.array([10, 255, 255])
    # mask_R = cv2.inRange(hsv, lower_red, upper_red)
    # mask = mask_G + mask_R
    tongdao_value=int(tongdao)
    cellx_value="%.2f"%(cellx)
    celly_value="%.2f"%(celly)
    group_type_value=int(group_type)
    avg_gray=np.mean(np.array(gray))
    max_gray=np.max(np.array(gray))
    min_gray=np.min(np.array(gray))
    sum_gray=np.sum(np.array(gray))
    diff_gray=max_gray-min_gray
    if max_gray!=0:
        edge_diff="%.2f"%((max_gray-min_gray)/max_gray)
    else:
        edge_diff = 0
    bg_diff = abs(int(avg_gray)-int(min_gray))
    grad_gray=cv2.morphologyEx(crop_gray, cv2.MORPH_GRADIENT, kernel=np.ones((3, 3), np.uint8))
    ret, binary = cv2.threshold(grad_gray, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(contours)
    # '细胞编号' + "," + '属性值' + "," + '通道' + "," + 'X坐标' + "," + 'Y坐标' + "," + '团属性'  + "," \
    # '面积' + "," + '周长' + "," + '长轴' + "," + '短轴' + "," + '圆度' + "," \
    # + '边缘锐利度' + "," + '边缘差异率' + "," + '平均灰度' + "," + '累计灰度' + "," + '最大灰度' + "," + '最小灰度' + "," + '灰度差' + "," + '背景差异'  + "," +"正圆率"+ "," + "亮度"+"\n"
    if len(cnts)== 0:
        area=max_aix*min_aix
        zhouchang=(max_aix+min_aix)*2
        if max(max_aix,min_aix)==0 or min(max_aix,min_aix)==0:
            zhengyuan_rate = 0
            roundness = 0
        else:
            zhengyuan_rate = min(max_aix, min_aix) / max(max_aix, min_aix)
            roundness = max(max_aix, min_aix) / min(max_aix, min_aix)
        max_aix_value = max_aix
        min_aix_value = min_aix
        return [tongdao_value,cellx_value,celly_value,group_type_value,area,zhouchang,max_aix_value,
                min_aix_value,roundness,sharpne,edge_diff,avg_gray,sum_gray,max_gray,min_gray,diff_gray,bg_diff,zhengyuan_rate,liangdu]
    elif cv2.contourArea(max(cnts, key=cv2.contourArea))<20:
        area=max_aix*min_aix
        zhouchang=(max_aix+min_aix)*2
        if max(max_aix,min_aix)==0 or min(max_aix,min_aix)==0:
            zhengyuan_rate = 0
            roundness = 0
        else:
            zhengyuan_rate = min(max_aix, min_aix) / max(max_aix, min_aix)
            roundness = max(max_aix, min_aix) / min(max_aix, min_aix)
        max_aix_value = max_aix
        min_aix_value = min_aix
        return [tongdao_value,cellx_value,celly_value,group_type_value,area,zhouchang,max_aix_value,
                min_aix_value,roundness,sharpne,edge_diff,avg_gray,sum_gray,max_gray,min_gray,diff_gray,bg_diff,zhengyuan_rate,liangdu]
    else:
        maxc = max(cnts, key=cv2.contourArea)
        zhouchang=cv2.arcLength(maxc, True)
        area=cv2.contourArea(maxc)
        _, _, w, h = cv2.boundingRect(maxc)
        max_aix_value = max(w,h)
        min_aix_value = min(w,h)
        if area==None:
            area=0
        if zhouchang==None:
            zhouchang=0
        a = cv2.contourArea(maxc) * 4 * math.pi
        b = math.pow(cv2.arcLength(maxc, True), 2)
        if b == 0:
            roundness= 0
        else:
            roundness= a / b
        zhengyuan_rate = min(max_aix_value,min_aix_value) / max(max_aix_value,min_aix_value)
        return [tongdao_value,cellx_value,celly_value,group_type_value,area,zhouchang,max_aix_value,
                min_aix_value,roundness,sharpne,edge_diff,avg_gray,sum_gray,max_gray,min_gray,diff_gray,bg_diff,zhengyuan_rate,liangdu]

def pianyimergeadd(mc, fl1, fl2, px1=None, py1=None, px2=None, py2=None):
    if px1 != None:
        h1, w1, _ = mc.shape
        zeros = np.zeros(mc.shape[:2], dtype="uint8")
        # print(mc.shape, fl1.shape,px1,py1,fl2.shape,px2,py2)
        M1 = np.float32([[1, 0, -px1], [0, 1, -py1]])
        M2 = np.float32([[1, 0, -px2], [0, 1, -py2]])
        fl1_warp = cv2.warpAffine(fl1, M1, (w1, h1))
        fl2_warp = cv2.warpAffine(fl2, M2, (w1, h1))
        fl1_split = cv2.split(fl1_warp)[1]
        fl2_split = cv2.split(fl2_warp)[2]
        fl = cv2.merge([zeros, fl1_split, fl2_split])
        return cv2.add(mc, fl)
    else:
        zeros = np.zeros(mc.shape[:2], dtype="uint8")
        fl1_split = cv2.split(fl1)[1]
        fl2_split = cv2.split(fl2)[2]
        fl = cv2.merge([zeros, fl1_split, fl2_split])
        return cv2.add(mc, fl)

def auto_imagejiupian(mc, fl1, fl2):
    img_bf = mc
    img_green = fl1
    img_red =fl2
    img_bf_gray = cv2
    img_bf_gray = cv2.cvtColor(img_bf, cv2.COLOR_BGR2GRAY)
    img_green_gray = cv2.cvtColor(img_green, cv2.COLOR_BGR2GRAY)
    img_red_gray = cv2.cvtColor(img_red, cv2.COLOR_BGR2GRAY)
    window_size = 1000
    if max(img_bf_gray.shape) < 1050:
        window_size = 500
    padding = 50
    grdimage = cv2.morphologyEx(img_red_gray, cv2.MORPH_GRADIENT, np.ones((15, 15), np.uint8))
    select_image = grdimage
    select_windows = sorted(
        [[j, j + window_size, i, i + window_size, np.sum(select_image[j:j + window_size, i:i + window_size])] for i in
         range(padding, select_image.shape[1], window_size) for j in
         range(padding, select_image.shape[0], window_size)], key=lambda x: x[-1], reverse=True)[0]
    # print(img_bf_gray.shape)
    # print(select_windows)
    # print(select_windows)
    # [50, 1050, 1050, 2050, 2572612]
    ystart, yend, xstart, xend = select_windows[0], select_windows[1], select_windows[2], select_windows[3]
    # cv2.rectangle(img_bf,(xstart,ystart),(xend,yend),(0))
    src_img = img_bf_gray[ystart - padding:yend + padding, xstart - padding:xend + padding]
    green_template = img_green_gray[ystart:yend, xstart:xend]
    red_template = img_red_gray[ystart:yend, xstart:xend]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient_src = cv2.morphologyEx(src_img, cv2.MORPH_GRADIENT, kernel)
    green_gradient_tem = cv2.morphologyEx(green_template, cv2.MORPH_GRADIENT, kernel)
    red_gradient_tem = cv2.morphologyEx(red_template, cv2.MORPH_GRADIENT, kernel)
    # 相关系数匹配方法：cv2.TM_CCOEFF
    g_res = cv2.matchTemplate(gradient_src, green_gradient_tem, cv2.TM_CCOEFF)
    g_max_loc = cv2.minMaxLoc(g_res)[-1]
    g_dw = g_max_loc[0] - padding
    g_dh = g_max_loc[1] - padding
    print("FL1 X", g_dw, " Y ", g_dh)  # 偏移值
    r_res = cv2.matchTemplate(gradient_src, red_gradient_tem, cv2.TM_CCOEFF)
    r_max_loc = cv2.minMaxLoc(r_res)[-1]
    r_dw = r_max_loc[0] - padding
    r_dh = r_max_loc[1] - padding
    print("FL2 X", r_dw, " Y ", r_dh)  # 偏移值
    h1, w1, _ = img_bf.shape
    M1 = np.float32([[1, 0, g_dw], [0, 1, g_dh]])
    jiaozhengreen = cv2.warpAffine(img_green, M1, (w1, h1))
    M1 = np.float32([[1, 0, r_dw], [0, 1, r_dh]])
    jiaozhenred = cv2.warpAffine(img_red, M1, (w1, h1))
    zeros = np.zeros(img_bf.shape[:2], dtype="uint8")
    fl1_split = cv2.split(jiaozhengreen)[1]
    fl2_split = cv2.split(jiaozhenred)[2]
    fl = cv2.merge([zeros, fl1_split, fl2_split])
    add_result = cv2.add(img_bf, fl)
    #addweight_result = cv2.addWeighted(src1=img_bf, alpha=0.5, src2=fl, beta=0.5, gamma=0)
    #g_dw, g_dh, r_dw, r_dh,
    return jiaozhengreen,jiaozhenred,add_result

def hsv_greed_and_red(cropImg):
    fl1_hsv=cv2.cvtColor(cropImg,cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 43, 25])
    upper_green = np.array([99, 255, 255])
    mask_G = cv2.inRange(fl1_hsv, lower_green, upper_green)
    fl2_hsv=cv2.cvtColor(cropImg,cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])
    mask_R = cv2.inRange(fl2_hsv, lower_red, upper_red)
    if np.sum(mask_R)>np.sum(mask_G):
        return 1
    else:
        return 0
def convet_str(list):
    return ",".join([str(i) for i in list])
def crop_box(im,x1,y1,x2,y2,pading=2):
    x1=max(x1-pading,0)
    y1=max(y1-pading,0)
    x2=min(x2+pading,im.shape[1])
    y2=min(y2+pading,im.shape[0])
    return im[y1:y2,x1:x2]

def sice_detection(im0,cut_size,detection_model,conf_thres,iou_thres):
    durations_in_seconds={}
    time_start = time.time()
    slice_image_result = slice_image(
        image=convert_from_cv2_to_image(im0),
        slice_height=cut_size,
        slice_width=cut_size,
        overlap_height_ratio=0.1,
        overlap_width_ratio=0.1,
    )
    num_slices = len(slice_image_result)
    durations_in_seconds["slice"] = time.time() - time_start
    # create prediction input
    num_batch = 1
    num_group = int(num_slices / num_batch)
    print("Number of slices:", num_slices)
    # perform sliced prediction
    pred_time = time.time()
    object_prediction_list = []
    for group_ind in range(num_group):
        # prepare batch (currently supports only 1 batch)
        image_list = []
        shift_amount_list = []
        for image_ind in range(num_batch):
            image_list.append(slice_image_result.images[group_ind * num_batch + image_ind])
            shift_amount_list.append(slice_image_result.starting_pixels[group_ind * num_batch + image_ind])
        # perform batch prediction
        prediction_result = get_prediction(
            image=image_list[0],
            detection_model=detection_model,
            image_size=640,
            shift_amount=shift_amount_list[0],
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            full_shape=[
                slice_image_result.original_image_height,
                slice_image_result.original_image_width,
            ]
        )
        object_prediction_list.append(prediction_result)
    # if num_slices > 1 and perform_standard_pred:
    #     # perform standard prediction
    #     prediction_result = get_prediction(
    #         image=im0,
    #         detection_model=detection_model,
    #         image_size=image_size,
    #         shift_amount=[0, 0],
    #         full_shape=None,
    #     )
    #     object_prediction_list.append(prediction_result)
    object_prediction_list = [object_prediction for object_prediction in object_prediction_list if
                              isinstance(object_prediction, np.ndarray)]
    if len(object_prediction_list)>0:
        det = np.concatenate(object_prediction_list, axis=0)
        durations_in_seconds["prediction"] = time.time() - pred_time
        nms_start = time.time()
        keep = nms(det[:, :4], det[:, 4],iou_threshold=iou_thres, score_threshold=conf_thres,cutoff_distance=64)
        det = det[keep]
        print("Number of Cell:", det.shape[0])
        # WBweighted box clustering
        # parrl_xyxy, label, cluster_indices = wbc(det[:,:4].cpu().numpy().astype(np.float64), det[:,4].cpu().numpy().astype(np.float64), iou_threshold=iou_thres, score_threshold=conf_thres,
        #                                                    cutoff_distance=cutoff_distance)
        durations_in_seconds["NMS"] = time.time() - nms_start
        array_xyxy = det[:, :4]
        cell_info = np.empty([len(det), 13])
        cell_info[:, 0] = ((array_xyxy[:, 0] + array_xyxy[:, 2]) / 2).astype(int)  # center x
        cell_info[:, 1] = ((array_xyxy[:, 1] + array_xyxy[:, 3]) / 2).astype(int)  # center y
        cell_info[:, 2] = ((array_xyxy[:, 2] - array_xyxy[:, 0]) + (array_xyxy[:, 3] - array_xyxy[:, 1])) / 4  # 半径
        cell_info[:, 3] = np.ones(len(det))  # 团属性
        cell_info[:, 4] = det[:, -1].astype(int)  # 标签
        cell_info[:, 5] = ((array_xyxy[:, 2] - array_xyxy[:, 0]) + (array_xyxy[:, 3] - array_xyxy[:, 1])) / 2  # 直径
        cell_info[:, 6] = array_xyxy[:, 0]  # tx
        cell_info[:, 7] = array_xyxy[:, 1]  # ty
        cell_info[:, 8] = array_xyxy[:, 2]  # bx
        cell_info[:, 9] = array_xyxy[:, 3]  # by
        cell_info[:, 10] = np.multiply((array_xyxy[:, 2] - array_xyxy[:, 0]),
                                       (array_xyxy[:, 3] - array_xyxy[:, 1]))  # 面积
        cell_info[:, 11] = np.maximum((array_xyxy[:, 2] - array_xyxy[:, 0]),
                                      (array_xyxy[:, 3] - array_xyxy[:, 1]))  # 长轴
        cell_info[:, 12] = np.minimum((array_xyxy[:, 2] - array_xyxy[:, 0]),
                                      (array_xyxy[:, 3] - array_xyxy[:, 1]))  # 短轴
        print(durations_in_seconds)
        return  cell_info
    return []
def compute_IOU(rec1,rec2):
    """
    计算两个矩形框的交并比。
    :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
    :param rec2: (x0,y0,x1,y1)
    :return: 交并比IOU.
    """
    left_column_max  = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max       = max(rec1[1],rec2[1])
    down_row_min     = min(rec1[3],rec2[3])
    #两矩形无相交区域的情况
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        return S_cross/(S1+S2-S_cross)
def get_bbox(cell_info,im_shape,padding=0):
    left_column_max=max(cell_info[6]-padding,0) #tx
    right_column_min=min(cell_info[8]+padding,im_shape[1])#bx
    up_row_max=max(cell_info[7]-padding,0) #ty
    down_row_min=min(cell_info[9]+padding,im_shape[0]) #by
    return list(map(int, [left_column_max,up_row_max,right_column_min,down_row_min]))
def get_shuxing(bf0,BFall_cell_info,FL1all_cell_info,FL2all_cell_info):
    compare_map=np.zeros([bf0.shape[1],bf0.shape[0],bf0.shape[2]])
    growh_num=1
    for cell_num, (cellx, celly, redius, group_type, label_num, diam, xyxy_0, xyxy_1, xyxy_2, xyxy_3, Area, max_aix,
                   min_aix) in enumerate(BFall_cell_info):
        compare_map[int(cellx),int(celly),0]=cell_num+growh_num
    for cell_num, (cellx, celly, redius, group_type, label_num, diam, xyxy_0, xyxy_1, xyxy_2, xyxy_3, Area, max_aix,
                   min_aix) in enumerate(FL1all_cell_info):
        compare_map[int(cellx),int(celly),1]=cell_num+growh_num
    for cell_num, (cellx, celly, redius, group_type, label_num, diam, xyxy_0, xyxy_1, xyxy_2, xyxy_3, Area, max_aix,
                   min_aix) in enumerate(FL2all_cell_info):
        compare_map[int(cellx),int(celly),2]=cell_num+growh_num
    BF_FL1_FL2_array=compare_map[np.where((compare_map[:, :, 0] > 0)&(compare_map[:, :, 1] > 0)&(compare_map[:, :, 2] > 0))]
    BF_FL1_FL2=(BF_FL1_FL2_array-growh_num).tolist()
    BF,BF_FL1,BF_FL2,FL1,FL2,FL1_FL2=[],[],[],[],[],[]
    compare_map[np.where((compare_map[:,:,0]>0)&(compare_map[:,:,1]>0)&(compare_map[:,:,2]>0))]=0
    compare_map=compare_map.astype(int)
    iou_thred=0.8
    for bfchannel_index in compare_map[np.nonzero(compare_map[:,:,0])]:
        bf_cell_info=BFall_cell_info[bfchannel_index[0]-growh_num].astype(int)
        bf_roi_extend_box=get_bbox(bf_cell_info,bf0.shape,padding=bf_cell_info[2])
        bf_box=get_bbox(bf_cell_info,bf0.shape,padding=0)
        channelroi=compare_map[bf_roi_extend_box[0]:bf_roi_extend_box[2],bf_roi_extend_box[1]:bf_roi_extend_box[3]]
        exist_fl1,exist_fl2 = False,False
        fl1_index,fl2_index=-1,-1
        for FL1_channel in channelroi[np.nonzero(channelroi[:, :, 1])]:
            fl1_cell_info=FL1all_cell_info[FL1_channel[1]-growh_num]
            fl1_box = get_bbox(fl1_cell_info, bf0.shape, padding=0)
            if compute_IOU(bf_box,fl1_box)>iou_thred:
                compare_map[int(fl1_cell_info[0]),int(fl1_cell_info[1]),1]=0
                exist_fl1=True
                fl1_index=FL1_channel[1]-growh_num
                break
        for FL2_channel in channelroi[np.nonzero(channelroi[:, :, 2])]:
            fl2_cell_info=FL2all_cell_info[FL2_channel[2]-growh_num]
            fl2_box = get_bbox(fl2_cell_info, bf0.shape, padding=0)
            if compute_IOU(bf_box,fl2_box)>iou_thred:
                compare_map[int(fl2_cell_info[0]),int(fl2_cell_info[1]),2]=0
                exist_fl2 = True
                fl2_index = FL2_channel[2]-growh_num
                break
        compare_map[int(bf_cell_info[0]),int(bf_cell_info[1]),0]=0
        if exist_fl1==True and exist_fl2==True:
            BF_FL1_FL2.append([bfchannel_index[0]-growh_num, fl1_index, fl2_index])
        elif exist_fl1:
            BF_FL1.append([bfchannel_index[0]-growh_num, fl1_index])
        elif exist_fl2:
            BF_FL2.append([bfchannel_index[0]-growh_num, fl2_index])
        else:
            BF.extend([bfchannel_index[0]-growh_num])

    for fl1channel_index in compare_map[np.nonzero(compare_map[:,:,1])]:
        fl1_cell_info=FL1all_cell_info[fl1channel_index[1]-growh_num]
        fl1_roi_extend_box=get_bbox(fl1_cell_info,bf0.shape,padding=fl1_cell_info[2])
        fl1_box=get_bbox(fl1_cell_info,bf0.shape,padding=0)
        channelroi=compare_map[fl1_roi_extend_box[0]:fl1_roi_extend_box[2],fl1_roi_extend_box[1]:fl1_roi_extend_box[3]]
        exist_fl2 = False
        fl2_index=-1
        for FL2_channel in channelroi[np.nonzero(channelroi[:, :, 2])]:
            fl2_cell_info=FL2all_cell_info[FL2_channel[2]-growh_num]
            fl2_box = get_bbox(fl2_cell_info, bf0.shape, padding=0)
            if compute_IOU(fl1_box,fl2_box)>iou_thred:
                compare_map[int(fl2_cell_info[0]),int(fl2_cell_info[1]),2]=0
                exist_fl2 = True
                fl2_index = FL2_channel[2]-growh_num
                break
        compare_map[int(fl1_cell_info[0]),int(fl1_cell_info[1]),1]=0
        if exist_fl2==True:
            FL1_FL2.append([fl1channel_index[1]-growh_num,fl2_index])
        else:
            FL1.extend([fl1channel_index[1]-growh_num])
    BF.extend((compare_map[:,:,0][np.nonzero(compare_map[:,:,0])]-growh_num).tolist())
    FL1.extend((compare_map[:,:,1][np.nonzero(compare_map[:,:,1])]-growh_num).tolist())
    FL2.extend((compare_map[:,:,2][np.nonzero(compare_map[:,:,2])]-growh_num).tolist())
    return BF, BF_FL1, BF_FL2, BF_FL1_FL2, FL1, FL2, FL1_FL2

def sigle_computer(input_im_path,
                   image_save_path,
                   data_save_path,
                   cut_size: int = 640,
                   overlap_height_ratio: float = 0.1,
                   overlap_width_ratio: float = 0.1,
                   perform_standard_pred: bool = True,
                   conf_thres: float = 0.2,  # confidence threshold
                   iou_thres: float = 0.1,  # NMS IOU threshold
                   cutoff_distance: int = 64,
                   BF_detection_model=None,
                   FL_detection_model=None,
                   distance_tred=0,
                   line_thickness=1):
    print("detmodel_PID", os.getpid())
    start_time = time.time()
    Label = ["BF", "BF_FL1", "BF_FL2", "BF_FL1_FL2", "FL1", "FL2","FL1_FL2"]
    shuxing = [1, 6, 7, 16, 2, 3,10]
    custom_colors = [(211, 211, 211), (34, 139, 34), (0, 0, 255), (0, 215, 255), (124, 205, 124), (85, 85, 205),(0,238,238)]
    bf0 = cv2.imdecode(np.fromfile(input_im_path[0], dtype=np.uint8), cv2.IMREAD_COLOR)
    fl10 = cv2.imdecode(np.fromfile(input_im_path[1], dtype=np.uint8), cv2.IMREAD_COLOR)
    fl20 = cv2.imdecode(np.fromfile(input_im_path[2], dtype=np.uint8), cv2.IMREAD_COLOR)  # BGR
    fl10, fl20, im0 = auto_imagejiupian(bf0, fl10, fl20)
    anysis_result_title = '细胞编号' + "," + '属性值' + "," + '通道' + "," + 'X坐标' + "," + 'Y坐标' + "," + '团属性'  + "," \
                          '面积' + "," + '周长' + "," + '长轴' + "," + '短轴' + "," + '圆度' + "," \
                          + '边缘锐利度' + "," + '边缘差异率' + "," + '平均灰度' + "," + '累计灰度' + "," + '最大灰度' + "," + '最小灰度' + "," + '灰度差' + "," + '背景差异'  + "," +"正圆率"+ "," + "亮度"+"\n"
    anysis_result = ""
    BFgroup_list, BFall_cell_info=sice_detection(bf0,cut_size,BF_detection_model,conf_thres,iou_thres)
    FL1group_list, FL1all_cell_info=sice_detection(fl10,cut_size,FL_detection_model,conf_thres,iou_thres)
    FL2group_list, FL2all_cell_info=sice_detection(fl20,cut_size,FL_detection_model,conf_thres,iou_thres)
    shuxing_cost_time=time.time()
    BF, BF_FL1, BF_FL2, BF_FL1_FL2, FL1, FL2, FL1_FL2=get_shuxing(bf0,BFall_cell_info,FL1all_cell_info,FL2all_cell_info)
    print("属性计算耗时", time.time() - shuxing_cost_time)
    pr_cell_num=0
    for cell_number in BF:
        cell_shuxing=1
        cellx, celly, redius, group_type, label_num, diam, xyxy_0, xyxy_1, xyxy_2, xyxy_3, Area, max_aix,min_aix=BFall_cell_info[int(cell_number)]
        cell_result = compute_type(0, group_type, cellx, celly, max_aix, min_aix, bf0, int(xyxy_0),int(xyxy_1), int(xyxy_2), int(xyxy_3))
        plot_cricle((int(cellx), int(celly)), int((cell_result[6] + cell_result[7]) / 4), bf0,color=custom_colors[0],line_thickness=line_thickness)
        plot_cricle((int(cellx), int(celly)), int((cell_result[6] + cell_result[7]) / 4), im0,color=custom_colors[shuxing.index(cell_shuxing)],line_thickness=line_thickness)
        pr_cell_num+=1
        head = str(pr_cell_num) + "," + str(cell_shuxing) + ","
        anysis_result += head + convet_str(cell_result) + "\n"
    for cell_number in FL1:
        cell_shuxing=2
        cellx, celly, redius, group_type, label_num, diam, xyxy_0, xyxy_1, xyxy_2, xyxy_3, Area, max_aix,min_aix=FL1all_cell_info[int(cell_number)]
        cell_result = compute_type(1, group_type, cellx, celly, max_aix, min_aix, fl10, int(xyxy_0),int(xyxy_1), int(xyxy_2), int(xyxy_3))
        plot_cricle((int(cellx), int(celly)), int((cell_result[6] + cell_result[7]) / 4), fl10,color=custom_colors[4],line_thickness=line_thickness)
        plot_cricle((int(cellx), int(celly)), int((cell_result[6] + cell_result[7]) / 4), im0,color=custom_colors[shuxing.index(cell_shuxing)],line_thickness=line_thickness)
        pr_cell_num += 1
        head = str(pr_cell_num) + "," + str(cell_shuxing) + ","
        anysis_result += head + convet_str(cell_result) + "\n"
    for cell_number in FL2:
        cell_shuxing=3
        cellx, celly, redius, group_type, label_num, diam, xyxy_0, xyxy_1, xyxy_2, xyxy_3, Area, max_aix,min_aix=FL2all_cell_info[int(cell_number)]
        cell_result = compute_type(2, group_type, cellx, celly, max_aix, min_aix, fl20, int(xyxy_0),int(xyxy_1), int(xyxy_2), int(xyxy_3))
        plot_cricle((int(cellx), int(celly)), int((cell_result[6] + cell_result[7]) / 4), fl20,color=custom_colors[5],line_thickness=line_thickness)
        plot_cricle((int(cellx), int(celly)), int((cell_result[6] + cell_result[7]) / 4), im0,color=custom_colors[shuxing.index(cell_shuxing)],line_thickness=line_thickness)
        pr_cell_num += 1
        head = str(pr_cell_num) + "," + str(cell_shuxing) + ","
        anysis_result += head + convet_str(cell_result) + "\n"
    for cell_list in BF_FL1:
        cell_shuxing=6
        cellx, celly, redius, group_type, label_num, diam, xyxy_0, xyxy_1, xyxy_2, xyxy_3, Area, max_aix,min_aix=BFall_cell_info[int(cell_list[0])]
        BF_cell_result = compute_type(0, group_type, cellx, celly, max_aix, min_aix, bf0, int(xyxy_0), int(xyxy_1),int(xyxy_2), int(xyxy_3))
        plot_cricle((int(cellx), int(celly)), int((BF_cell_result[6] + BF_cell_result[7]) / 4), bf0,color=custom_colors[0], line_thickness=line_thickness)
        cellx, celly, redius, group_type, label_num, diam, xyxy_0, xyxy_1, xyxy_2, xyxy_3, Area, max_aix,min_aix=FL1all_cell_info[int(cell_list[1])]
        FL1_cell_result = compute_type(1, group_type, cellx, celly, max_aix, min_aix, fl10, int(xyxy_0), int(xyxy_1),int(xyxy_2), int(xyxy_3))
        plot_cricle((int(cellx), int(celly)), int((FL1_cell_result[6] + FL1_cell_result[7]) / 4), fl10,color=custom_colors[4], line_thickness=line_thickness)
        plot_cricle((int(cellx), int(celly)), int((BF_cell_result[6] + BF_cell_result[7]) / 4), im0,color=custom_colors[shuxing.index(cell_shuxing)], line_thickness=line_thickness)
        pr_cell_num += 1
        head = str(pr_cell_num) + "," + str(cell_shuxing) + ","
        anysis_result += head + convet_str(BF_cell_result) + "\n" + head + convet_str(FL1_cell_result) + "\n"
    for cell_list in BF_FL2:
        cell_shuxing=7
        cellx, celly, redius, group_type, label_num, diam, xyxy_0, xyxy_1, xyxy_2, xyxy_3, Area, max_aix,min_aix=BFall_cell_info[int(cell_list[0])]
        BF_cell_result = compute_type(0, group_type, cellx, celly, max_aix, min_aix, bf0, int(xyxy_0), int(xyxy_1),int(xyxy_2), int(xyxy_3))
        plot_cricle((int(cellx), int(celly)), int((BF_cell_result[6] + BF_cell_result[7]) / 4), bf0,color=custom_colors[0], line_thickness=line_thickness)
        cellx, celly, redius, group_type, label_num, diam, xyxy_0, xyxy_1, xyxy_2, xyxy_3, Area, max_aix,min_aix=FL2all_cell_info[int(cell_list[1])]
        FL2_cell_result = compute_type(2, group_type, cellx, celly, max_aix, min_aix, fl20, int(xyxy_0), int(xyxy_1),int(xyxy_2), int(xyxy_3))
        plot_cricle((int(cellx), int(celly)), int((FL2_cell_result[6] + FL2_cell_result[7]) / 4), fl20,color=custom_colors[5], line_thickness=line_thickness)
        plot_cricle((int(cellx), int(celly)), int((BF_cell_result[6] + BF_cell_result[7]) / 4), im0,color=custom_colors[shuxing.index(cell_shuxing)], line_thickness=line_thickness)
        pr_cell_num += 1
        head = str(pr_cell_num) + "," + str(cell_shuxing) + ","
        anysis_result += head + convet_str(BF_cell_result) + "\n" + head + convet_str(FL2_cell_result) + "\n"
    for cell_list in FL1_FL2:
        cell_shuxing=10
        cellx, celly, redius, group_type, label_num, diam, xyxy_0, xyxy_1, xyxy_2, xyxy_3, Area, max_aix,min_aix=FL1all_cell_info[int(cell_list[0])]
        FL1_cell_result = compute_type(1, group_type, cellx, celly, max_aix, min_aix, fl10, int(xyxy_0), int(xyxy_1),int(xyxy_2), int(xyxy_3))
        plot_cricle((int(cellx), int(celly)), int((FL1_cell_result[6] + FL1_cell_result[7]) / 4), fl10,color=custom_colors[4], line_thickness=line_thickness)
        cellx, celly, redius, group_type, label_num, diam, xyxy_0, xyxy_1, xyxy_2, xyxy_3, Area, max_aix,min_aix=FL2all_cell_info[int(cell_list[1])]
        FL2_cell_result = compute_type(2, group_type, cellx, celly, max_aix, min_aix, fl20, int(xyxy_0), int(xyxy_1),int(xyxy_2), int(xyxy_3))
        plot_cricle((int(cellx), int(celly)), int((FL2_cell_result[6] + FL2_cell_result[7]) / 4), fl20,color=custom_colors[5], line_thickness=line_thickness)
        plot_cricle((int(cellx), int(celly)), int((FL1_cell_result[6] + FL1_cell_result[7]) / 4), im0,color=custom_colors[shuxing.index(cell_shuxing)], line_thickness=line_thickness)
        pr_cell_num += 1
        head = str(pr_cell_num) + "," + str(cell_shuxing) + ","
        anysis_result += head + convet_str(FL1_cell_result) + "\n" + head + convet_str(FL2_cell_result) + "\n"
    for cell_list in BF_FL1_FL2:
        cell_shuxing=16
        cellx, celly, redius, group_type, label_num, diam, xyxy_0, xyxy_1, xyxy_2, xyxy_3, Area, max_aix,min_aix=BFall_cell_info[int(cell_list[0])]
        BF_cell_result = compute_type(0, group_type, cellx, celly, max_aix, min_aix, bf0, int(xyxy_0), int(xyxy_1),int(xyxy_2), int(xyxy_3))
        plot_cricle((int(cellx), int(celly)), int((BF_cell_result[6] + BF_cell_result[7]) / 4), bf0,color=custom_colors[0], line_thickness=line_thickness)
        cellx, celly, redius, group_type, label_num, diam, xyxy_0, xyxy_1, xyxy_2, xyxy_3, Area, max_aix,min_aix=FL1all_cell_info[int(cell_list[1])]
        FL1_cell_result = compute_type(1, group_type, cellx, celly, max_aix, min_aix, fl10, int(xyxy_0), int(xyxy_1),int(xyxy_2), int(xyxy_3))
        plot_cricle((int(cellx), int(celly)), int((FL1_cell_result[6] + FL1_cell_result[7]) / 4), fl10,color=custom_colors[4], line_thickness=line_thickness)
        cellx, celly, redius, group_type, label_num, diam, xyxy_0, xyxy_1, xyxy_2, xyxy_3, Area, max_aix,min_aix=FL2all_cell_info[int(cell_list[2])]
        FL2_cell_result = compute_type(2, group_type, cellx, celly, max_aix, min_aix, fl20, int(xyxy_0), int(xyxy_1),int(xyxy_2), int(xyxy_3))
        plot_cricle((int(cellx), int(celly)), int((FL2_cell_result[6] + FL2_cell_result[7]) / 4), fl20,color=custom_colors[5], line_thickness=line_thickness)
        plot_cricle((int(cellx), int(celly)), int((BF_cell_result[6] + BF_cell_result[7]) / 4), im0,color=custom_colors[shuxing.index(cell_shuxing)], line_thickness=line_thickness)
        pr_cell_num += 1
        head = str(pr_cell_num) + "," + str(cell_shuxing) + ","
        anysis_result += head + convet_str(BF_cell_result) + "\n"+head + convet_str(FL1_cell_result) + "\n" + head + convet_str(FL2_cell_result) + "\n"

    for group_list_plot in zip([bf0,fl10,fl20],[BFgroup_list,FL1group_list,FL2group_list]):
        for cell_group in group_list_plot[1]:
            reduis = np.max(squareform(pdist(np.array(cell_group)))) + 1
            if len(cell_group) == 2:
                roundceter = (round((cell_group[0][0] + cell_group[1][0]) / 2),
                              round((cell_group[0][1] + cell_group[1][1]) / 2))
                # cv2.line(im0, ctentsers[0], ctentsers[1], (255, 255, 0), 1, cv2.LINE_AA)
                cv2.circle(group_list_plot[0], roundceter, int(reduis), (250, 250, 255), line_thickness, cv2.LINE_AA)
            else:
                px, py = get_centerpoint(cell_group)
                cv2.circle(group_list_plot[0], (px, py), int(reduis), (250, 250, 255), line_thickness, cv2.LINE_AA)
    print("总耗时:",time.time()-start_time)
    try:
        if os.path.exists(data_save_path):
            os.remove(data_save_path)
        with open(data_save_path, 'a+', encoding='utf-8') as f:
            f.write(anysis_result_title + anysis_result)
    except Exception as e:
        print(e)
        # traceback.print_exc(file=open("Error_LOG.txt", 'a+'))
    try:
        save_resultimage(bf0,image_save_path[0])
        save_resultimage(fl10,image_save_path[1])
        save_resultimage(fl20,image_save_path[2])
        save_resultimage(im0,image_save_path[3])
    except Exception as e:
        print(e)
        # traceback.print_exc(file=open("Error_LOG.txt", 'a+'))
def typeconvert(conf_thres):
    if isinstance(conf_thres,str):
        print("%s类型错误，强制类型转换"%conf_thres)
        conf_thres=float(conf_thres)
    return conf_thres
def detmodel_iference(
    input_im_path,
    image_save_path,
    data_save_path,
    fcs_save_path,
    cut_size: int = 640,
    overlap_height_ratio: float = 0.1,
    overlap_width_ratio: float = 0.1,
    perform_standard_pred: bool = True,
    conf_thres: float = 0.2,  # confidence threshold
    iou_thres: float = 0.1,  # NMS IOU threshold
    cutoff_distance: int = 64,
    typeblue_model=None,
    BFdetection_model=None,
    FLdeteciton_model=None,
    distance_tred=0,
    line_thickness=1,
    BF_conf_thrd=0.1,
    BF_iou_thrd=0.1,
    FL1_conf_thrd=0.1,
    FL1_iou_thrd=0.1,
    FL2_conf_thrd=0.1,
    FL2_iou_thrd=0.1,
    extype="trypanblue"):
    print(input_im_path,image_save_path,cut_size,type(cut_size))
    print("detmodel_PID", os.getpid())
    conf_thres=typeconvert(conf_thres)
    iou_thres=typeconvert(iou_thres)
    BF_conf_thrd=typeconvert(BF_conf_thrd)
    BF_iou_thrd=typeconvert(BF_iou_thrd)
    FL1_conf_thrd=typeconvert(FL1_conf_thrd)
    FL1_iou_thrd=typeconvert(FL1_iou_thrd)
    FL2_conf_thrd=typeconvert(FL2_conf_thrd)
    FL2_iou_thrd=typeconvert(FL2_iou_thrd)
    SumLabel=["BF","BF_FL1","BF_FL2","BF_FL1_FL2","FL1","FL2","FL1_FL2","dead","live"]
    shuxing = [1, 6, 7, 16, 2, 3,10,35,34]
    # currently only 1 batch supported
    time_start=time.time()
    if extype == "trypanblue":
        cell_infos = cell_list()
        custom_Label=["dead","live"]
        im0 = cv2.imdecode(np.fromfile(input_im_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        all_cell_info = sice_detection(im0, cut_size, typeblue_model, conf_thres, iou_thres)
        if len(all_cell_info) > 0:
            for cell_num, (cellx, celly, redius, group_type, label_num, diam, xyxy_0, xyxy_1, xyxy_2, xyxy_3, Area, max_aix,
                      min_aix) in enumerate(all_cell_info):
                cell=cell_struct(channel_struct(group_type=group_type,type=shuxing[SumLabel.index(custom_Label[int(label_num)])]).get_det_info(img=im0,xyxy_0=int(xyxy_0),xyxy_1=int(xyxy_1),xyxy_2=int(xyxy_2),xyxy_3=int(xyxy_3)),None,None)
                cell_infos.push_cell(cell)
        cell_infos.group_computer(0,distance_tred=0)
        im0 = cell_infos.draw_cell_info(im0, 0)
        save_resultimage(im0,image_save_path)
        cell_infos.to_csv(data_save_path)
        cell_infos.to_csv(fcs_save_path)
    if extype == "suspension_BF":
        im0 = cv2.imdecode(np.fromfile(input_im_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        all_cell_info = sice_detection(im0, cut_size, BFdetection_model, conf_thres, iou_thres)
        cell_infos = cell_list()
        if len(all_cell_info) > 0:
            for cell_num, (cellx, celly, redius, group_type, label_num, diam, xyxy_0, xyxy_1, xyxy_2, xyxy_3, Area, max_aix,min_aix) in enumerate(all_cell_info):
                cell=cell_struct(channel_struct(group_type=group_type,type=shuxing[SumLabel.index("BF")]).get_det_info(img=im0,xyxy_0=int(xyxy_0),xyxy_1=int(xyxy_1),xyxy_2=int(xyxy_2),xyxy_3=int(xyxy_3)),None,None)
                cell_infos.push_cell(cell)
        cell_infos.group_computer(0,distance_tred=1.1)
        im0 = cell_infos.draw_cell_info(im0, 0)
        save_resultimage(im0,image_save_path)
        cell_infos.to_csv(data_save_path)
        cell_infos.to_csv(fcs_save_path)
    if extype == "suspension_FL":
        cell_infos = cell_list()
        if input_im_path[0]=="" or input_im_path[1]=="":
            if input_im_path[0]=="":
                fl20 = cv2.imdecode(np.fromfile(input_im_path[1], dtype=np.uint8), cv2.IMREAD_COLOR)
                fl20_cell_info = sice_detection(fl20, cut_size, FLdeteciton_model, FL2_conf_thrd,FL2_iou_thrd)
                for cell_num, (cellx, celly, redius, group_type, label_num, diam, xyxy_0, xyxy_1, xyxy_2, xyxy_3, Area, max_aix,min_aix) in enumerate(fl20_cell_info):
                    cell = cell_struct(BF=None,FL1=None,FL2=channel_struct(group_type=group_type, local_channel=2, type=shuxing[SumLabel.index("FL2")]).get_det_info(img=fl20, xyxy_0=int(xyxy_0),xyxy_1=int(xyxy_1),xyxy_2=int(xyxy_2),xyxy_3=int(xyxy_3)))
                    cell_infos.push_cell(cell)
                cell_infos.group_computer(2)
                fl20=cell_infos.draw_group(fl20,2)
                im0 = cell_infos.draw_cell_info(fl20, 2)
                save_resultimage(im0,image_save_path[1])
                cell_infos.to_csv(data_save_path)
                cell_infos.to_csv(fcs_save_path)
            if input_im_path[1]=="":
                fl10 = cv2.imdecode(np.fromfile(input_im_path[0], dtype=np.uint8), cv2.IMREAD_COLOR)
                fl10_cell_info = sice_detection(fl10, cut_size, FLdeteciton_model, FL1_conf_thrd,FL1_iou_thrd)
                for cell_num, (cellx, celly, redius, group_type, label_num, diam, xyxy_0, xyxy_1, xyxy_2, xyxy_3, Area, max_aix,min_aix) in enumerate(fl10_cell_info):
                    cell = cell_struct(BF=None,FL2=None,FL1=channel_struct(group_type=group_type, local_channel=1, type=shuxing[SumLabel.index("FL1")]).get_det_info(img=fl10,xyxy_0=int(xyxy_0),xyxy_1=int(xyxy_1),xyxy_2=int(xyxy_2),xyxy_3=int(xyxy_3)))
                    cell_infos.push_cell(cell)
                cell_infos.group_computer(1)
                fl10 = cell_infos.draw_group(fl10, 1)
                im0 = cell_infos.draw_cell_info(fl10, 1)
                save_resultimage(im0,image_save_path[0])
                cell_infos.to_csv(data_save_path)
                cell_infos.to_csv(fcs_save_path)
        else:
            fl10 = cv2.imdecode(np.fromfile(input_im_path[0], dtype=np.uint8), cv2.IMREAD_COLOR)
            fl20 = cv2.imdecode(np.fromfile(input_im_path[1], dtype=np.uint8), cv2.IMREAD_COLOR)  # BGR
            fl1copy = fl10.copy()
            fl2copy = fl20.copy()
            zeros = np.zeros(fl10.shape[:2], dtype="uint8")
            fl1_split = cv2.split(fl10)[1]
            fl2_split = cv2.split(fl20)[2]
            merge_image = cv2.merge([zeros, fl1_split, fl2_split])
            fl10_cell_info = sice_detection(fl10, cut_size, FLdeteciton_model, FL1_conf_thrd,FL1_iou_thrd)
            fl20_cell_info = sice_detection(fl20, cut_size, FLdeteciton_model, FL2_conf_thrd,FL2_iou_thrd)
            fl1cell_list = cell_list()
            for cell_num, (cellx, celly, redius, group_type, label_num, diam, xyxy_0, xyxy_1, xyxy_2, xyxy_3, Area, max_aix,min_aix) in enumerate(fl10_cell_info):
                cell = cell_struct(BF=None, FL2=None, FL1=channel_struct(group_type=group_type, local_channel=1,type=shuxing[SumLabel.index("FL1")]).get_det_info(img=fl10,xyxy_0=int(xyxy_0), xyxy_1=int(xyxy_1), xyxy_2=int(xyxy_2),xyxy_3=int(xyxy_3)))
                fl1cell_list.push_cell(cell)
            fl1cell_list.group_computer(1)
            fl10 = cell_infos.draw_group(fl10, 1)
            fl10 = fl1cell_list.draw_cell_info(fl10, 1)
            save_resultimage(fl10,image_save_path[0])
            #save_resultimage(cell_infos.draw_cell_info(merge_image.copy(), 1),'H:\\test\\A1_01_01_01_fl1_m.jpg')
            fl2cell_list = cell_list()
            for cell_num, (cellx, celly, redius, group_type, label_num, diam, xyxy_0, xyxy_1, xyxy_2, xyxy_3, Area, max_aix,min_aix) in enumerate(fl20_cell_info):
                cell = cell_struct(BF=None, FL1=None, FL2=channel_struct(group_type=group_type, local_channel=2,type=shuxing[SumLabel.index("FL2")]).get_det_info(img=fl20, xyxy_0=int(xyxy_0), xyxy_1=int(xyxy_1), xyxy_2=int(xyxy_2),xyxy_3=int(xyxy_3)))
                fl2cell_list.push_cell(cell)
            fl2cell_list.group_computer(2)
            fl20 = cell_infos.draw_group(fl20, 2)
            cell_infos=fl1cell_list.merge_cell_list(fl2cell_list)
            fl20 = cell_infos.draw_cell_info(fl20, 2)
            #save_resultimage(cell_infos.draw_cell_info(merge_image.copy(), 2),'H:\\test\\A1_01_01_01_fl2_m.jpg')
            save_resultimage(fl20,image_save_path[1])
            cell_infos=cell_infos.get_shuxing(fl10.shape)
            cell_infos.draw_cell_info(merge_image,"all")
            save_resultimage(merge_image,image_save_path[2])
            cell_infos.to_csv(data_save_path)
            fcs_infos=cell_infos.get_shuxing(fl10.shape,bf_img=None,fl1_img=fl1copy,fl2_img=fl2copy,fcs_output=True)
            fcs_infos.to_csv(fcs_save_path)

    if extype == "AOPI_merge":
        bf0 = cv2.imdecode(np.fromfile(input_im_path[0], dtype=np.uint8), cv2.IMREAD_COLOR)
        bfcopy=bf0.copy()
        bf0_cell_info = sice_detection(bf0, cut_size, BFdetection_model,BF_conf_thrd,BF_iou_thrd)
        bfcell_list = cell_list()
        for cell_num, (cellx, celly, redius, group_type, label_num, diam, xyxy_0, xyxy_1, xyxy_2, xyxy_3, Area, max_aix,min_aix) in enumerate(bf0_cell_info):
            cell = cell_struct(FL2=None, FL1=None, BF=channel_struct(group_type=group_type, local_channel=0,
                                                                     type=shuxing[SumLabel.index("BF")]).get_det_info(
                img=bf0, xyxy_0=int(xyxy_0), xyxy_1=int(xyxy_1), xyxy_2=int(xyxy_2), xyxy_3=int(xyxy_3)))
            bfcell_list.push_cell(cell)
        bfcell_list.group_computer(0)
        bf0_im_reulst = bfcell_list.draw_group(bf0, 0)
        bf0_im_reulst = bfcell_list.draw_cell_info(bf0_im_reulst, 0)
        save_resultimage(bf0_im_reulst,image_save_path[0])
        if input_im_path[1]=="" or input_im_path[2]=="":
            if input_im_path[1]=="":
                fl2cell_list = cell_list()
                fl20 = cv2.imdecode(np.fromfile(input_im_path[2], dtype=np.uint8), cv2.IMREAD_COLOR)
                zeros = np.zeros(bf0.shape[:2], dtype="uint8")
                fl2_split = cv2.split(fl20)[2]
                fl = cv2.merge([zeros, zeros, fl2_split])
                merge_image = cv2.add(bf0, fl)
                fl20_cell_info = sice_detection(fl20, cut_size, FLdeteciton_model, FL2_conf_thrd,FL2_iou_thrd)
                for cell_num, (cellx, celly, redius, group_type, label_num, diam, xyxy_0, xyxy_1, xyxy_2, xyxy_3, Area, max_aix,min_aix) in enumerate(fl20_cell_info):
                    cell = cell_struct(BF=None,FL1=None,FL2=channel_struct(group_type=group_type, local_channel=2, type=shuxing[SumLabel.index("FL2")]).get_det_info(img=fl20, xyxy_0=int(xyxy_0),xyxy_1=int(xyxy_1),xyxy_2=int(xyxy_2),xyxy_3=int(xyxy_3)))
                    fl2cell_list.push_cell(cell)
                fl2cell_list.group_computer(2)
                fl20=fl2cell_list.draw_group(fl20,2)
                fl20_im_reulst = fl2cell_list.draw_cell_info(fl20, 2)
                save_resultimage(fl20_im_reulst,image_save_path[2])
                cell_infos=bfcell_list.merge_cell_list(fl2cell_list)
                cell_infos = cell_infos.get_shuxing(bf0.shape)
                cell_infos.draw_cell_info(merge_image, "all")
                save_resultimage(merge_image,image_save_path[3])
                cell_infos.to_csv(data_save_path)
                cell_infos.to_csv(fcs_save_path)
            if input_im_path[2]=="":
                fl1cell_list = cell_list()
                fl10 = cv2.imdecode(np.fromfile(input_im_path[1], dtype=np.uint8), cv2.IMREAD_COLOR)
                zeros = np.zeros(bf0.shape[:2], dtype="uint8")
                fl1_split = cv2.split(fl10)[1]
                fl = cv2.merge([zeros, fl1_split, zeros])
                merge_image = cv2.add(bf0, fl)
                fl10_cell_info = sice_detection(fl10, cut_size, FLdeteciton_model, FL1_conf_thrd,FL1_iou_thrd)
                for cell_num, (cellx, celly, redius, group_type, label_num, diam, xyxy_0, xyxy_1, xyxy_2, xyxy_3, Area, max_aix,min_aix) in enumerate(fl10_cell_info):
                    cell = cell_struct(BF=None,FL2=None,FL1=channel_struct(group_type=group_type, local_channel=1, type=shuxing[SumLabel.index("FL1")]).get_det_info(img=fl10,xyxy_0=int(xyxy_0),xyxy_1=int(xyxy_1),xyxy_2=int(xyxy_2),xyxy_3=int(xyxy_3)))
                    fl1cell_list.push_cell(cell)
                fl1cell_list.group_computer(1)
                fl10 = fl1cell_list.draw_group(fl10, 1)
                fl10_im_reulst = fl1cell_list.draw_cell_info(fl10, 1)
                save_resultimage(fl10_im_reulst,image_save_path[1])
                cell_infos=bfcell_list.merge_cell_list(fl1cell_list)
                cell_infos = cell_infos.get_shuxing(bf0.shape)
                cell_infos.draw_cell_info(merge_image, "all")
                save_resultimage(merge_image,image_save_path[3])
                cell_infos.to_csv(data_save_path)
                cell_infos.to_csv(fcs_save_path)
        elif input_im_path[1]=="" and input_im_path[2]=="":
            print("请至少包含一个荧光通道")
            assert "请至少包含一个荧光通道"
        else:
            fl10 = cv2.imdecode(np.fromfile(input_im_path[1], dtype=np.uint8), cv2.IMREAD_COLOR)
            fl20 = cv2.imdecode(np.fromfile(input_im_path[2], dtype=np.uint8), cv2.IMREAD_COLOR)  # BGR
            fl1copy = fl10.copy()
            fl2copy = fl20.copy()
            zeros = np.zeros(fl10.shape[:2], dtype="uint8")
            fl1_split = cv2.split(fl10)[1]
            fl2_split = cv2.split(fl20)[2]
            fl = cv2.merge([zeros, fl1_split, fl2_split])
            merge_image=cv2.add(bf0,fl)
            fl10_cell_info = sice_detection(fl10, cut_size, FLdeteciton_model, FL1_conf_thrd,FL1_iou_thrd)
            fl20_cell_info = sice_detection(fl20, cut_size, FLdeteciton_model, FL2_conf_thrd,FL2_iou_thrd)
            fl1cell_list = cell_list()
            for cell_num, (cellx, celly, redius, group_type, label_num, diam, xyxy_0, xyxy_1, xyxy_2, xyxy_3, Area, max_aix,min_aix) in enumerate(fl10_cell_info):
                cell = cell_struct(BF=None, FL1=channel_struct(group_type=group_type, local_channel=1,type=shuxing[SumLabel.index("FL1")]).get_det_info(img=fl10,xyxy_0=int(xyxy_0), xyxy_1=int(xyxy_1), xyxy_2=int(xyxy_2),xyxy_3=int(xyxy_3)), FL2=None)
                fl1cell_list.push_cell(cell)
            fl1cell_list.group_computer(1)
            fl10 = fl1cell_list.draw_group(fl10, 1)
            fl10 = fl1cell_list.draw_cell_info(fl10, 1)
            save_resultimage(fl10,image_save_path[1])
            fl2cell_list = cell_list()
            #save_resultimage(cell_infos.draw_cell_info(merge_image.copy(), 1),'H:\\test\\A1_01_01_01_fl1_m.jpg')
            for cell_num, (cellx, celly, redius, group_type, label_num, diam, xyxy_0, xyxy_1, xyxy_2, xyxy_3, Area, max_aix,min_aix) in enumerate(fl20_cell_info):
                cell = cell_struct(BF=None, FL1=None, FL2=channel_struct(group_type=group_type, local_channel=2,type=shuxing[SumLabel.index("FL2")]).get_det_info(img=fl20, xyxy_0=int(xyxy_0), xyxy_1=int(xyxy_1), xyxy_2=int(xyxy_2),xyxy_3=int(xyxy_3)))
                fl2cell_list.push_cell(cell)
            fl2cell_list.group_computer(2)
            fl20 = fl2cell_list.draw_group(fl20, 2)
            fl20 = fl2cell_list.draw_cell_info(fl20, 2)
            #save_resultimage(cell_infos.draw_cell_info(merge_image.copy(), 2),'H:\\test\\A1_01_01_01_fl2_m.jpg')
            save_resultimage(fl20,image_save_path[2])
            cell_infos=bfcell_list.merge_cell_list(fl1cell_list)
            cell_infos=cell_infos.merge_cell_list(fl2cell_list)
            cell_infos=cell_infos.get_shuxing(fl10.shape)
            cell_infos.draw_cell_info(merge_image,"all")
            save_resultimage(merge_image,image_save_path[3])
            cell_infos.to_csv(data_save_path)
            fcs_infos=cell_infos.get_shuxing(fl10.shape,bf_img=bfcopy,fl1_img=fl1copy,fl2_img=fl2copy,fcs_output=True)
            fcs_infos.to_csv(fcs_save_path)
    print("总处理时间", time.time() - time_start)


