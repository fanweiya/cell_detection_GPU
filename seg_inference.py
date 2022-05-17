import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from utils.general import *
def waterseg(img, imgmask, data_save_path, fcs_save_path, cell_type="single_clone", fast_mode=True, min_distance=120):
    stime = time.time()
    im0s_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if cell_type == "single_clone":
        # 移除噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opening = cv2.morphologyEx(imgmask, cv2.MORPH_OPEN, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        local_max = peak_local_max(dist_transform, indices=False, min_distance=min_distance, labels=imgmask)
        markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
        markersend = watershed(-dist_transform, markers, mask=imgmask)
        cell_num = len(np.unique(markersend)) - 1
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        opening = cv2.morphologyEx(imgmask, cv2.MORPH_OPEN, kernel)
        dist_transform = cv2.distanceTransform(imgmask, cv2.DIST_L2, 3)
        sure_bg = cv2.dilate(opening, kernel)
        ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
        # 获得未知区域
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        # 标记
        ret, markers1 = cv2.connectedComponents(sure_fg)
        # 确保背景是1不是0
        markers = markers1 + 1
        # 未知区域标记为0
        markers[unknown == 255] = 0
        markersend = cv2.watershed(img, markers)
        cell_num = len(np.unique(markersend)) - 2
    cell_infos = cell_list()
    print("%s waterseg cost time" % cell_num, time.time() - stime)
    if cell_num > 0:
        statime3 = time.time()
        seg_object_Thrend_list = []
        if fast_mode:
            print("fast mode True")
            #print("markersend area ",np.sum(imgmask),np.sum(markersend>=2),cell_num)
            area = "%.2f" % (cv2.countNonZero(imgmask) / cell_num)
            for num in range(cell_num):
                cell_infos.push_cell(
                    cell_struct(BF=channel_struct(local_channel=0, area=area, type=1), FL1=None, FL2=None))
        else:
            # mask = np.zeros(imgmask.shape, dtype="uint8")
            # imgmarkers=[]
            # stime2=time.time()
            # for label in np.unique(markersend)[1:]:
            #     if cell_type != "single_clone" and label == 1:
            #         continue
            #     else:
            #         mask[markersend == label] = 1
            #         maskinv = np.ones(imgmask.shape, dtype="uint8")
            #         maskinv[markersend == label] = 0
            #         cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #         cnts = imutils.grab_contours(cnts)
            #         maxc = max(cnts, key=cv2.contourArea)
            #         x, y, w, h = cv2.boundingRect(maxc)
            #         bg_gray = cv2.bitwise_and(im0s_gray, im0s_gray, mask=maskinv)[y:y + h, x:x + w]
            #         gray = cv2.bitwise_and(im0s_gray, im0s_gray, mask=mask)[y:y + h, x:x + w]
            #         imgroi = cv2.bitwise_and(img, img, mask=mask)[y:y + h, x:x + w]
            #         imgmarkers.append([imgroi, gray, bg_gray])
            # print("markerer:",time.time()-stime2)
            obj_list = []
            Threadnum=2
            # if os.cpu_count()-2>2:
            #     Threadnum=os.cpu_count()-2
            print("Threadnum",Threadnum)
            with ThreadPoolExecutor(max_workers=Threadnum) as t:
                for label in np.unique(markersend)[1:]:
                    if cell_type != "single_clone" and label == 1:
                        continue
                    z=[img, im0s_gray, imgmask, markersend, label]
                    obj = t.submit(lambda p: opencv_computer_type(*p), z)
                    obj_list.append(obj)
            for future in as_completed(obj_list):
                cell_infos.push_cell(future.result())
            # for label in np.unique(markersend)[1:]:
            #     if cell_type != "single_clone" and label == 1:
            #         continue
            #     t = Muti_Thread_get_result(opencv_computer_type, (img, im0s_gray, imgmask, markersend, label,))
            #     seg_object_Thrend_list.append(t)
            #     t.start()
            #     # print("getmask", time.time() - statime5)
            #     # cv2.imwrite("%s.jpg"%str(label),seg_object)
            #     # result=opencv_computer_other(label-1,seg_object)
            # for t in seg_object_Thrend_list:
            #     t.join()  # 一定要join，不然主线程比子线程跑的快，会拿不到结果
            #     # anysis_result = anysis_result+t.get_result()+'\n'
            #     cell_infos.push_cell(t.get_result())
            statime4 = time.time()
            print("属性计算cost_time", statime4 - statime3)
            cell_infos.group_computer(0)
            print("computer type and out costtime", time.time() - statime3)
    cell_infos.to_csv(data_save_path,group_computer=False)
    cell_infos.to_csv(fcs_save_path,group_computer=False)
    resultimg = mask_image(img, imgmask)
    return resultimg, cell_num


def transfection_computer(SEG_result, data_savepath, fcs_save_path, fast_mode=False):
    cell_infos = cell_list()
    bf_count = np.sum(SEG_result["BF_segresult"] == 1)
    BF_gray = cv2.cvtColor(SEG_result["BF_img"], cv2.COLOR_BGR2GRAY)
    bf_cropgray = cv2.bitwise_and(BF_gray,BF_gray, mask=SEG_result["BF_segresult"])
    try:
        bf_avg_temp_gray = int(np.sum(bf_cropgray)/bf_count)
    except:
        bf_avg_temp_gray = int(np.sum(bf_cropgray))
    try:
        bf_min_temp_gray = int(np.min(bf_cropgray[np.nonzero(bf_cropgray)]))
    except:
        bf_min_temp_gray = 0
    bfrsult = channel_struct(local_channel=0, type=1, group_type=1, area=int(bf_count),avg_gray=bf_avg_temp_gray,sum_gray=int(np.sum(bf_cropgray)),
                             max_gray=int(np.max(bf_cropgray)),min_gray=bf_min_temp_gray,center_x=int(BF_gray.shape[1]/2),center_y=int(BF_gray.shape[0]/2))
    w, h = SEG_result["BF_img"].shape[:2]
    channel = [np.zeros((w, h), dtype="uint8")]
    #bf_counter = counter_image(SEG_result["BF_img"], SEG_result["BF_segresult"])
    bf_mask_img =mask_image(SEG_result["BF_img"], SEG_result["BF_segresult"])
    save_resultimage(bf_mask_img,SEG_result["BF_save_path"])
    if "FL1_img" in SEG_result:
        fl1_gray = cv2.cvtColor(SEG_result["FL1_img"], cv2.COLOR_BGR2GRAY)
        fl1_cropgray = cv2.bitwise_and(fl1_gray, fl1_gray, mask=SEG_result["FL1_segresult"])
        fl1_cropimg = cv2.bitwise_and(SEG_result["FL1_img"], SEG_result["FL1_img"], mask=SEG_result["FL1_segresult"])
        fl1_count = np.sum(fl1_cropgray > 0)
        try:
            fl1_avg_temp_gray=int(np.sum(fl1_cropgray) / fl1_count)
        except:
            fl1_avg_temp_gray=int(np.sum(fl1_cropgray))
        try:
            fl1_min_temp_gray=int(np.min(fl1_cropgray[np.nonzero(fl1_cropgray)]))
        except:
            fl1_min_temp_gray=0
        fl1rsult = channel_struct(local_channel=1, type=6, group_type=1, area=int(fl1_count),
                                  avg_gray=fl1_avg_temp_gray, sum_gray=int(np.sum(fl1_cropgray)),
                                  min_gray=fl1_min_temp_gray, max_gray=int(np.max(fl1_cropgray)),
                                  center_x=int(fl1_gray.shape[1] / 2), center_y=int(fl1_gray.shape[0] / 2))
        fl1_split = cv2.split(cv2.resize(fl1_cropimg, (h, w)))[1]
        channel.append(fl1_split)
        fl1_mask_img = mask_image(SEG_result["FL1_img"], SEG_result["FL1_segresult"])
        #fl1_counter = counter_image(fl1_mask_img, SEG_result["FL1_segresult"])
        save_resultimage(fl1_mask_img,SEG_result["FL1_save_path"])
    else:
        fl1rsult = None
        channel.append(np.zeros((w, h), dtype="uint8"))
    if "FL2_img" in SEG_result:
        fl2_gray = cv2.cvtColor(SEG_result["FL2_img"], cv2.COLOR_BGR2GRAY)
        fl2_cropgray = cv2.bitwise_and(fl2_gray, fl2_gray, mask=SEG_result["FL2_segresult"])
        fl2_cropimg = cv2.bitwise_and(SEG_result["FL2_img"], SEG_result["FL2_img"], mask=SEG_result["FL2_segresult"])
        fl2_count = np.sum(fl2_cropgray > 0)
        try:
            fl2_avg_temp_gray=int(np.sum(fl2_cropgray) / fl2_count)
        except:
            fl2_avg_temp_gray=int(np.sum(fl2_cropgray))
        try:
            fl2_min_temp_gray=int(np.min(fl2_cropgray[np.nonzero(fl2_cropgray)]))
        except:
            fl2_min_temp_gray=0
        fl2rsult = channel_struct(local_channel=2, type=7, group_type=1, area=int(fl2_count),
                                  avg_gray=fl2_avg_temp_gray, sum_gray=int(np.sum(fl2_cropgray)),
                                  min_gray=fl2_min_temp_gray, max_gray=int(np.max(fl2_cropgray)),
                                  center_x=int(fl2_gray.shape[1] / 2), center_y=int(fl2_gray.shape[0] / 2))
        fl2_split = cv2.split(cv2.resize(fl2_cropimg, (h, w)))[2]
        channel.append(fl2_split)
        fl2_mask_img = mask_image(SEG_result["FL2_img"], SEG_result["FL2_segresult"])
        # fl2_counter = counter_image(fl2_mask_img, SEG_result["FL2_segresult"])
        save_resultimage(fl2_mask_img,SEG_result["FL2_save_path"])
    else:
        fl2rsult = None
        channel.append(np.zeros((w, h), dtype="uint8"))
    cell = cell_struct(BF=bfrsult, FL1=fl1rsult, FL2=fl2rsult)
    cell_infos.push_cell(cell)
    cell_infos.to_csv(data_save_path=data_savepath)
    cell_infos.to_csv(data_save_path=fcs_save_path)
    fl = cv2.merge(channel)
    merge_image = cv2.add(SEG_result["BF_img"], fl)
    return merge_image


def segmodel_iference(kuangpredictor, grouppredictor, chopredictor, flpredictor, processing_type, input_image_path,
                      image_save_path, data_save_path, fcs_save_path, cut_patch, fastmode,mab_min_distance=120):
    print("segmodel_PID", os.getpid())
    if processing_type == "single_clone":
        im = cv2.imdecode(np.fromfile(input_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        stime = time.time()
        if sum(np.array(im[10, 10])) == 0:
            # result = grouppredictor.overlap_tile_predict(img_file=im, tile_size=(512, 512), pad_size=[16, 16])
            result = grouppredictor.predict(im)
            result = get_roimask(result, cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        else:
            segresult = kuangpredictor.predict(input_image_path)
            #segobject = get_roi(im, segresult)
            result = grouppredictor.predict(im)
            result = get_roimask(result, segresult['label_map'])
        print("推理耗时", time.time() - stime)
        print("mab_min_distance", mab_min_distance)
        resultimg, cell_num = waterseg(im, result['label_map'], data_save_path=data_save_path,
                                       fcs_save_path=fcs_save_path, cell_type="Mab", fast_mode=False,
                                       min_distance=mab_min_distance)
        save_resultimage(resultimg,image_save_path)
        print("总耗时:", time.time() - stime)

    elif processing_type == "confluence":
        if cut_patch == None:
            cut_patch = 3
        else:
            cut_patch = int(cut_patch)
        # predict
        print(input_image_path)
        im = cv2.imdecode(np.fromfile(input_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        print(im.shape)
        stime = time.time()
        if sum(np.array(im[10, 10])) == 0:
            result = chopredictor.overlap_tile_predict(img_file=im, tile_size=(
            int(im.shape[1] / cut_patch), int(im.shape[1] / cut_patch)))
            result = get_roimask(result, cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        else:
            segresult = kuangpredictor.predict(input_image_path)
            # segobject = get_roi(im, segresult)
            result = chopredictor.overlap_tile_predict(img_file=im, tile_size=(
            int(im.shape[1] / cut_patch), int(im.shape[1] / cut_patch)))
            result = get_roimask(result, segresult['label_map'])
        print("推理耗时", time.time() - stime)
        resultimg, cell_num = waterseg(im, result['label_map'], data_save_path=data_save_path,
                                       fcs_save_path=fcs_save_path, cell_type="confluence", fast_mode=fastmode)
        save_resultimage(resultimg,image_save_path)
        print("总耗时:", time.time() - stime)

    elif processing_type == "transfection":
        if cut_patch == None:
            cut_patch = 3
        else:
            cut_patch = int(cut_patch)
        # predict
        print(input_image_path)
        SEG_result = {}
        bf_im = cv2.imdecode(np.fromfile(input_image_path[0], dtype=np.uint8), cv2.IMREAD_COLOR)
        print(bf_im.shape)
        if sum(np.array(bf_im[10, 10])) == 0:
            result = chopredictor.overlap_tile_predict(img_file=bf_im, tile_size=(
            int(bf_im.shape[1] / cut_patch), int(bf_im.shape[1] / cut_patch)))
            result = get_roimask(result, cv2.cvtColor(bf_im, cv2.COLOR_BGR2GRAY))
            SEG_result["BF_img"] = bf_im
            SEG_result["BF_segresult"] = result['label_map']
            SEG_result["BF_save_path"] = image_save_path[0]
        else:
            #segresult = kuangpredictor.predict(input_image_path[0])
            # segobject = get_roi(bf_im, segresult)
            result = chopredictor.overlap_tile_predict(img_file=bf_im, tile_size=(
            int(bf_im.shape[1] / cut_patch), int(bf_im.shape[1] / cut_patch)))
            #result = get_roimask(result, segresult['label_map'])
            SEG_result["BF_img"] = bf_im
            SEG_result["BF_segresult"] = result['label_map']
            SEG_result["BF_save_path"] = image_save_path[0]
        if input_image_path[1] != "":
            fl1_im = cv2.imdecode(np.fromfile(input_image_path[1], dtype=np.uint8), cv2.IMREAD_COLOR)
            fl1result = flpredictor.overlap_tile_predict(img_file=fl1_im, tile_size=(
            int(fl1_im.shape[1] / (cut_patch + 2)), int(fl1_im.shape[1] / (cut_patch + 2))))
            fl1result = get_roimask(fl1result, cv2.cvtColor(bf_im, cv2.COLOR_BGR2GRAY))
            SEG_result["FL1_img"] = fl1_im
            SEG_result["FL1_segresult"] = fl1result['label_map']
            SEG_result["FL1_save_path"] = image_save_path[1]
        if input_image_path[2] != "":
            fl2_im = cv2.imdecode(np.fromfile(input_image_path[2], dtype=np.uint8), cv2.IMREAD_COLOR)
            fl2result = flpredictor.overlap_tile_predict(img_file=fl2_im, tile_size=(
            int(fl2_im.shape[1] / (cut_patch + 2)), int(fl2_im.shape[1] / (cut_patch + 2))))
            fl2result = get_roimask(fl2result, cv2.cvtColor(bf_im, cv2.COLOR_BGR2GRAY))
            SEG_result["FL2_img"] = fl2_im
            SEG_result["FL2_segresult"] = fl2result['label_map']
            SEG_result["FL2_save_path"] = image_save_path[2]
        stime = time.time()
        merge_image = transfection_computer(SEG_result, data_savepath=data_save_path, fcs_save_path=fcs_save_path)
        print("推理耗时", time.time() - stime)
        save_resultimage(merge_image,image_save_path[3])
        print("总耗时:", time.time() - stime)
    else:
        print("extype is not exits!")
        return 0
# qpt.exe -f ./SEG_openvino -p ./SEG_openvino/seg_inference.py -s ./SEG_openvino_qpt -h False

# 推理耗时 6.902669668197632
# waterseg cost time 0.42223668098449707
# computer type and out costtime 44.80705785751343
# 属性输出耗时 45.33993411064148
# 总耗时: 52.94390392303467
