# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
General utils
"""
import colorsys
import random
import threading
import time
from pathlib import Path

# Settings
import networkx
import numpy as np
from PIL import Image
from imantics import Mask
from lsnms import nms
from networkx.algorithms.components.connected import connected_components
from scipy.spatial.distance import squareform, pdist
from skimage import img_as_ubyte

# from numba_kdtree import KDTree
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
import math
import os
from sklearn.neighbors import KDTree
import cv2
import imutils


class channel_struct:
    def __init__(self, cell_img=np.zeros((1, 1, 3), np.uint8), cell_gray=np.zeros((1, 1), np.uint8), type=0,
                 local_channel=0, group_type=1, area=0, perimeter=0, major_axis_length=0, minor_axis_length=0,
                 roundness=0, sharpne=0, edge_diff=0, avg_gray: float = 0, sum_gray=0, max_gray=0, min_gray=0,
                 background_diff=0, cricle_rate=0, luminance=0, center_x=0, center_y=0):
        self.cell_img = cell_img  # 彩色图
        self.cell_gray = cell_gray  # 灰度图
        self.type = type  # 属性值
        self.local_channel = local_channel  # 通道
        self.center_x = center_x  # 中心点X坐标
        self.center_y = center_y  # 中心点Y坐标
        self.group_type = group_type  # 团属性
        self.major_axis_length = major_axis_length  # 主轴长度
        self.minor_axis_length = minor_axis_length  # 次轴长度
        self.area = area  # 面积
        self.perimeter = perimeter  # 周长
        self.roundness = roundness  # 圆度
        self.sharpne = sharpne  # 边缘锐利度
        self.avg_gray = avg_gray  # 平均灰度
        self.sum_gray = sum_gray  # 累计灰度
        self.min_gray = min_gray  # 最小灰度
        self.max_gray = max_gray  # 最大灰度
        self.diff_gray = self.max_gray - self.min_gray  # 灰度差
        self.edge_diff = edge_diff  # 边缘差异率
        self.radius = (self.major_axis_length + self.minor_axis_length) / 4
        self.background_diff = background_diff  # 背景差异
        self.cricle_rate = cricle_rate  # 正圆率
        self.luminance = luminance  # 光度
        self.struct_infos = {"属性值": self.type, "通道": self.local_channel, "中心点X坐标": self.center_x,
                             "中心点Y坐标": self.center_y, "团属性": self.group_type, "主轴长度": self.major_axis_length,
                             "次轴长度": self.minor_axis_length, "面积": self.area, "周长": self.perimeter,
                             "圆度": self.roundness, "边缘锐利度": self.sharpne, "平均灰度": self.avg_gray, "累计灰度": self.sum_gray,
                             "最小灰度": self.min_gray, "最大灰度": self.max_gray, "灰度差": self.diff_gray,
                             "边缘差异率": self.edge_diff, "背景差异": self.background_diff, "正圆率": self.cricle_rate,
                             "光度": self.luminance}

    def get_seg_info(self, cell_img=np.zeros((1, 1, 3), np.uint8), cell_gray=np.zeros((1, 1), np.uint8),
                     bg_gray=np.zeros((1, 1), np.uint8), center_x=0, center_y=0,mask=None):
        self.cell_img = cell_img  # 彩色图
        self.cell_gray = cell_gray  # 灰度图
        self.bg_gray = bg_gray
        self.conuter = self.find_conuter_seg(self.cell_gray)  # 轮廓
        self.center_x, self.center_y = center_x, center_y  # 中心点
        self.major_axis_length, self.minor_axis_length = self.compute_boudingbox(self.conuter)  # 长短轴
        self.area = cv2.countNonZero(cell_gray)  # 面积
        self.perimeter = cv2.arcLength(self.conuter, True)  # 周长
        self.roundness = (4 * math.pi * self.area) / self.perimeter ** 2 if self.perimeter != 0 else 0  # 圆度
        self.sharpne = cv2.Laplacian(self.cell_gray, cv2.CV_64F).var()  # 边缘锐利度
        self.avg_gray = np.mean(self.cell_gray)  # 平均灰度
        self.sum_gray = np.sum(self.cell_gray)  # 累计灰度
        self.min_gray = int(np.min(self.cell_gray[np.nonzero(self.cell_gray)]))  # 最小灰度
        self.max_gray = np.max(self.cell_gray)  # 最大灰度
        self.diff_gray = self.max_gray - self.min_gray  # 灰度差
        self.edge_diff = self.max_gray - int(np.min(bg_gray))
        self.background_diff = abs(int(self.avg_gray) - int(self.min_gray))  # 背景差异
        self.cricle_rate = self.minor_axis_length / self.major_axis_length if self.major_axis_length != 0 else 0  # 正圆率
        self.luminance = np.mean(cv2.cvtColor(self.cell_img, cv2.COLOR_BGR2HSV)[:, :, 2])
        self.radius = (self.major_axis_length + self.minor_axis_length) / 4
        self.struct_infos = {"属性值": self.type, "通道": self.local_channel, "中心点X坐标": self.center_x,
                             "中心点Y坐标": self.center_y,
                             "团属性": self.group_type, "主轴长度": self.major_axis_length, "次轴长度": self.minor_axis_length,
                             "面积": self.area, "周长": self.perimeter, "圆度": self.roundness, "边缘锐利度": self.sharpne,
                             "平均灰度": self.avg_gray, "累计灰度": self.sum_gray, "最小灰度": self.min_gray, "最大灰度": self.max_gray,
                             "灰度差": self.diff_gray, "边缘差异率": self.edge_diff, "背景差异": self.background_diff,
                             "正圆率": self.cricle_rate, "光度": self.luminance}
        del self.cell_img
        del self.cell_gray
        del self.bg_gray
        del self.conuter
        return self

    def get_det_info(self, img, group_type=1, xyxy_0=0, xyxy_1=0, xyxy_2=0, xyxy_3=0, padding=3):
        self.tx = xyxy_0
        self.ty = xyxy_1
        self.bx = xyxy_2
        self.by = xyxy_3
        self.cell_img = img[int(xyxy_1):int(xyxy_3), int(xyxy_0):int(xyxy_2)]  # 彩色图
        self.cell_gray = cv2.cvtColor(self.cell_img, cv2.COLOR_BGR2GRAY)  # 灰度图
        self.extend_img = self.crop_box(img, xyxy_0, xyxy_1, xyxy_2, xyxy_3, pading=padding)  # 扩展灰度图
        self.conuter = self.find_conuter_det(self.extend_img)  # 轮廓
        self.major_axis_length = max(xyxy_2 - xyxy_0, xyxy_3 - xyxy_1)
        self.minor_axis_length = min(xyxy_2 - xyxy_0, xyxy_3 - xyxy_1)  # 长短轴
        self.area = self.major_axis_length * self.minor_axis_length
        self.perimeter = 2 * (self.major_axis_length + self.minor_axis_length)
        self.center_x = int((xyxy_0 + xyxy_2) / 2)
        self.center_y = int((xyxy_1 + xyxy_3) / 2)  # 中心点
        self.group_type = group_type  # 团属性
        if len(self.conuter) > 0:
            if cv2.contourArea(self.conuter) > (2 * self.area) / 3:
                self.major_axis_length, self.minor_axis_length = self.compute_boudingbox(self.conuter)  # 长短轴
                self.area = cv2.contourArea(self.conuter)
                self.perimeter = cv2.arcLength(self.conuter, True)  # 周长
        self.roundness = (4 * math.pi * self.area) / self.perimeter ** 2 if self.perimeter != 0 else 0  # 圆度
        self.sharpne = cv2.Laplacian(self.cell_gray, cv2.CV_64F).var()  # 边缘锐利度
        self.avg_gray = np.mean(self.cell_gray)  # 平均灰度
        self.sum_gray = np.sum(self.cell_gray)  # 累计灰度
        self.min_gray = np.min(self.cell_gray)  # 最小灰度
        self.max_gray = np.max(self.cell_gray)  # 最大灰度
        self.diff_gray = self.max_gray - self.min_gray  # 灰度差
        self.edge_diff = (self.max_gray - self.min_gray) / self.max_gray if self.max_gray != 0 else 0
        self.background_diff = abs(int(self.avg_gray) - int(self.min_gray))  # 背景差异
        self.cricle_rate = self.minor_axis_length / self.major_axis_length if self.major_axis_length != 0 else 0  # 正圆率
        self.luminance = np.mean(cv2.cvtColor(self.cell_img, cv2.COLOR_BGR2HSV)[:, :, 2])
        #self.radius=int((self.major_axis_length+self.minor_axis_length)/4)-2 if int((self.major_axis_length+self.minor_axis_length)/4)-1>0 else int((self.major_axis_length+self.minor_axis_length)/4)
        self.radius = int((self.major_axis_length + self.minor_axis_length) / 4)
        self.struct_infos = {"属性值": self.type, "通道": self.local_channel, "中心点X坐标": self.center_x,
                             "中心点Y坐标": self.center_y,
                             "团属性": self.group_type, "主轴长度": self.major_axis_length, "次轴长度": self.minor_axis_length,
                             "面积": self.area, "周长": self.perimeter, "圆度": self.roundness, "边缘锐利度": self.sharpne,
                             "平均灰度": self.avg_gray, "累计灰度": self.sum_gray, "最小灰度": self.min_gray, "最大灰度": self.max_gray,
                             "灰度差": self.diff_gray, "边缘差异率": self.edge_diff, "背景差异": self.background_diff,
                             "正圆率": self.cricle_rate, "光度": self.luminance}
        del self.cell_img
        del self.cell_gray
        del self.extend_img
        del self.conuter
        return self

    def copy(self):
        import copy
        dup = copy.deepcopy(self)
        return dup

    def crop_box(self, im, x1, y1, x2, y2, pading=2):
        x1 = max(x1 - pading, 0)
        y1 = max(y1 - pading, 0)
        x2 = min(x2 + pading, im.shape[1])
        y2 = min(y2 + pading, im.shape[0])
        return im[y1:y2, x1:x2]

    def compute_boudingbox(self, conuter):
        try:
            x, y, w, h = cv2.boundingRect(conuter)
            axis = [max(w, h), min(w, h)]
        except:
            axis = [0, 0]
        return axis

    def delet_contours(self,contours, delete_list):
        #  自定义函数：用于删除列表指定序号的轮廓
        #  输入 1：contours：原始轮廓
        #  输入 2：delete_list：待删除轮廓序号列表
        #  返回值：contours：筛选后轮廓
        delta = 0
        for i in range(len(delete_list)):
            del contours[delete_list[i] - delta]
            delta = delta + 1
        return contours
    def hierarchy_contours(self,contours, hierarchy):
        # 筛选轮廓
        # 使用层级结构筛选轮廓
        # hierarchy[i]: [Next，Previous，First_Child，Parent]
        # 要求有父级轮廓
        #print("hierarchy",hierarchy)
        delete_list = []  # 新建待删除的轮廓序号列表
        c, row, col = hierarchy.shape
        for i in range(row):
            if hierarchy[0, i, 3]<0:  # 没有父轮廓或子轮廓
                delete_list.append(i)
        # 根据列表序号删除不符合要求的轮廓
        end_contours = self.delet_contours(contours, delete_list)
        # print(len(end_contours), "contours left after hierarchy filter")
        # drawMyContours("contours after hierarchy filtering", originimg, contours, False)
        return end_contours
    def find_conuter_det(self, crop_img):
        try:
            extend_gray=cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
            grad_gray = cv2.morphologyEx(extend_gray, cv2.MORPH_GRADIENT, kernel=np.ones((3, 3), np.uint8))
            ret, binary = cv2.threshold(grad_gray, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            #name=random.randint(100,999)
            #cv2.imwrite(r"H:\test\cut\%s_img" % str(name) + ".jpg", crop_img)
            #cv2.imwrite(r"H:\test\cut\%s_grad" % str(name) + ".jpg", grad_gray)
            #cv2.imwrite(r"H:\test\cut\%s_gray" % str(name) + ".jpg", extend_gray)
            #cv2.imwrite(r"H:\test\cut\%s_bin"%str(name)+".jpg",binary)
            cnts,hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            #cv2.imwrite(r"H:\test\cut\%s_cou"%str(name)+".jpg",cv2.drawContours(crop_img,cnts,-1,(255,0,0)))
            #hierarchy_cont = self.hierarchy_contours(cnts, hierarchy)
            #cnts = imutils.grab_contours(cnts)
            maxc = max(cnts, key=cv2.contourArea)
            #cv2.imwrite(r"H:\test\cut\%s_end"%str(name)+".jpg",cv2.drawContours(crop_img,maxc,-1,(0,255,0)))
            #cv2.imwrite(r"H:\test\cut\gg_%s" % str(name) + ".jpg", cv2.drawContours(gray, cnts, -1, (255, 0, 255)))

        except:
            maxc = []
        return maxc

    def find_conuter_seg(self, gray):
        try:
            cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            maxc = max(cnts, key=cv2.contourArea)
        except:
            maxc = []
        return maxc

    def print_cell_infos(self):
        print(self.struct_infos)

    def output(self):
        return [self.type, self.local_channel, self.center_x, self.center_y, self.group_type, self.area, self.perimeter,
                self.major_axis_length, self.minor_axis_length, self.roundness, self.sharpne, self.edge_diff,
                self.avg_gray, self.sum_gray, self.max_gray,
                self.min_gray, self.diff_gray, self.background_diff, self.cricle_rate, self.luminance]

    def output_group_info(self):
        return [self.center_x, self.center_y, self.group_type, self.area, self.perimeter, self.major_axis_length,
                self.minor_axis_length, self.roundness, self.sharpne, self.edge_diff, self.avg_gray, self.sum_gray,
                self.max_gray,
                self.min_gray, self.diff_gray, self.background_diff, self.cricle_rate, self.luminance]

    def to_file(self, path):
        cv2.imencode('.jpg', self.cell_img)[1].tofile(path)
        return True

def coutoure_jypue(firest_countour,process_countours):
    # 判断两个轮廓是否相交,如果相交，则剔除第一个轮廓集合中的轮廓
    del_list = []
    for two_co in process_countours:
        for index in range(len(firest_countour)):
            if cv2.pointPolygonTest(firest_countour[index],tuple([int(i) for i in two_co[0][0]]),False)>=0:
                del_list.append(index)
    for index in set(del_list[::-1]):
        firest_countour = firest_countour[:index] + firest_countour[index + 1:]
    process_countours+=(firest_countour)
    return process_countours

class cell_struct:
    def __init__(self, BF=None, FL1=None, FL2=None):
        self.cell_info = [BF, FL1, FL2]

    def get_channel_info(self, channel):
        if channel == 0 or channel == "BF" or channel == "bf":
            return self.cell_info[0]
        elif channel == 1 or channel == "FL1" or channel == "fl1":
            return self.cell_info[1]
        elif channel == 2 or channel == "FL2" or channel == "fl2":
            return self.cell_info[2]

    def get_all_channel_info(self, id=1):
        result = ""
        for channel in self.cell_info:
            if channel != None:
                result += str(id) + "," + ",".join([str(i) for i in channel.output()]) + "\n"
        return result


class cell_list():
    def __init__(self):
        self.cell_infos = []
        self.group_list = [[], [], []]
        self.group_falg = False

    def merge_cell_list(self, cell_list):
        for cell in cell_list.cell_infos:
            self.cell_infos.append(cell)
        return self

    def push_cell(self, cell_struct):
        self.cell_infos.append(cell_struct)

    def infos_output(self):
        anysis_reuslt_title = ','.join(['细胞编号', '属性值', '通道', 'X坐标', 'Y坐标', '团属性', '面积',
                                        '周长', '长轴', '短轴', '圆度', '边缘锐利度', '边缘差异率', '平均灰度', '累计灰度',
                                        '最大灰度', '最小灰度', '灰度差', '背景差异', '正圆率', '亮度']) + "\n"
        if len(self.cell_infos) > 0:
            result = ""
            for num, cell in enumerate(self.cell_infos):
                result += cell.get_all_channel_info(num + 1)
            return anysis_reuslt_title + result
        else:
            return anysis_reuslt_title

    def get_length(self):
        return len(self.cell_infos)

    def get_sum_channel(self):
        temp = []
        for cell in self.cell_infos:
            for channel in cell:
                if channel != None:
                    temp.append(channel.local_channel)
        return len(set(temp))

    def group_computer(self, channel, distance_tred=0):
        group_starttime = time.time()
        self.group_falg = True
        if len(self.cell_infos) < 2:
            print("one cell not make up a group")
            return False
        else:
            centerlist = []
            rediuslist = []
            for cell_num, cell in enumerate(self.cell_infos):
                if cell.get_channel_info(channel) != None:
                    centerlist.append(
                        [cell.get_channel_info(channel).center_x, cell.get_channel_info(channel).center_y])
                    rediuslist.append(cell.get_channel_info(channel).radius)
            # print("centerlist",len(centerlist))
            # print("rediuslist",len(rediuslist))
            if len(centerlist) > 0:
                redius = np.array(rediuslist)
                centerlist = np.array(centerlist)
                tree = KDTree(centerlist)
                centerindex = []
                topk = min(centerlist.shape[0], 50)
                for ind in range(centerlist.shape[0]):
                    dist, index = tree.query([centerlist[ind]], k=topk)
                    for num in range(1, topk):
                        if dist[0][num] <= redius[ind] + redius[index[0][num]] + distance_tred:
                            centerindex.append([index[0][0], index[0][num]])
                sortedlist = merge_lists(centerindex)
                len_list = [len(i) for i in sortedlist]
                # print("sortedlist",sortedlist)
                for list in sortedlist:
                    temp = [(int(centerlist[k][0]), int(centerlist[k][1])) for k in list]
                    self.group_list[channel].append(temp)
                    for k in list:
                        self.cell_infos[k].get_channel_info(channel).group_type = len(temp)
                print("channel ", channel, " 团计算cost_time", time.time() - group_starttime)
                return sum(len_list), self.group_list
            return False

    def draw_group(self, img, channel, line_thickness=1, color=(250, 250, 255)):#250, 250, 255
        if len(self.group_list[channel]) > 0:
            for cell_group in self.group_list[channel]:
                if len(cell_group) == 2:
                    raduis = np.max(squareform(pdist(np.array(cell_group)))) + 1
                    roundceter = (round((cell_group[0][0] + cell_group[1][0]) / 2),
                                  round((cell_group[0][1] + cell_group[1][1]) / 2))
                    # cv2.line(im0, ctentsers[0], ctentsers[1], (255, 255, 0), 1, cv2.LINE_AA)
                    cv2.circle(img, roundceter, int(raduis), color, line_thickness, cv2.LINE_AA)
                else:
                    raduis = np.max(squareform(pdist(np.array(cell_group))))-(0.75*len(cell_group))
                    px, py = get_centerpoint(cell_group)
                    cv2.circle(img, (px, py), int(raduis), color, line_thickness, cv2.LINE_AA)
        return img

    def draw_cell_info(self, img, channel, line_thickness=1):
        shuxing = [1, 6, 7, 16, 2, 3, 10, 35, 34]
        custom_colors = [(211, 211, 211), (34, 139, 34), (0, 0, 255), (0, 215, 255), (124, 205, 124), (85, 85, 205),
                         (0, 238, 238), (0, 0, 255), (0, 255, 0)]
        if len(self.cell_infos) > 0:
            tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
            #tl=2
            if channel != "all":
                img = self.draw_group(img, channel, line_thickness=line_thickness)
                for cell in self.cell_infos:
                    if cell.get_channel_info(channel) != None:
                        # cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
                        # cv2.ellipse(img,
                        #            (cell.get_channel_info(channel).center_x, cell.get_channel_info(channel).center_y),
                        #             (int(cell.get_channel_info(channel).major_axis_length/2),
                        #              int(cell.get_channel_info(channel).minor_axis_length/2)),0,0,360,color=custom_colors[shuxing.index(cell.get_channel_info(channel).type)], thickness=tl,lineType=cv2.LINE_AA)
                        cv2.circle(img,
                                   (cell.get_channel_info(channel).center_x, cell.get_channel_info(channel).center_y),
                                   cell.get_channel_info(channel).radius,
                                   custom_colors[shuxing.index(cell.get_channel_info(channel).type)], tl, cv2.LINE_AA)
            else:
                for cell in self.cell_infos:
                    if cell.get_channel_info(0) != None:
                        channel = 0
                    elif cell.get_channel_info(1) != None:
                        channel = 1
                    else:
                        channel = 2
                    # cv2.ellipse(img,
                    #             (cell.get_channel_info(channel).center_x, cell.get_channel_info(channel).center_y),
                    #             (int(cell.get_channel_info(channel).major_axis_length/2),
                    #              int(cell.get_channel_info(channel).minor_axis_length/2)), 0, 0, 360,
                    #             color=custom_colors[shuxing.index(cell.get_channel_info(channel).type)], thickness=tl,
                    #             lineType=cv2.LINE_AA)
                    cv2.circle(img,
                               (cell.get_channel_info(channel).center_x, cell.get_channel_info(channel).center_y),
                               cell.get_channel_info(channel).radius,
                               custom_colors[shuxing.index(cell.get_channel_info(channel).type)], tl, cv2.LINE_AA)

        return img

    def to_csv(self, data_save_path, group_computer=True):
        try:
            if os.path.exists(data_save_path):
                os.remove(data_save_path)
            if self.cell_infos == []:
                print("cell_infos is empty")
            if group_computer:
                if (self.group_falg == False):
                    print("group_computer is not run")
                    self.group_computer(channel=0)
            with open(data_save_path, 'a+', encoding='utf-8') as f:
                f.write(self.infos_output())
        except Exception as e:
            print("data_save_path ", data_save_path, " not exists", e)
        return True

    def get_RECT(self, infos, im_shape, padding=0):
        left_column_max = max(infos.tx - padding, 0)  # tx
        right_column_min = min(infos.bx + padding, im_shape[1])  # bx
        up_row_max = max(infos.ty - padding, 0)  # ty
        down_row_min = min(infos.by + padding, im_shape[0])  # by
        return list(map(int, [left_column_max, up_row_max, right_column_min, down_row_min]))

    def compute_IOU(self, rec1, rec2):
        """
        计算两个矩形框的交并比。
        :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
        :param rec2: (x0,y0,x1,y1)
        :return: 交并比IOU.
        """
        left_column_max = max(rec1[0], rec2[0])
        right_column_min = min(rec1[2], rec2[2])
        up_row_max = max(rec1[1], rec2[1])
        down_row_min = min(rec1[3], rec2[3])
        # 两矩形无相交区域的情况
        if left_column_max >= right_column_min or down_row_min <= up_row_max:
            return 0
        # 两矩形有相交区域的情况
        else:
            S1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
            S2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
            S_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
            return S_cross / (S1 + S2 - S_cross)
    def replace(self,cellinfo,replacinfo):
        cellinfo.area = replacinfo.area
        cellinfo.perimeter = replacinfo.perimeter
        cellinfo.major_axis_length = replacinfo.major_axis_length
        cellinfo.minor_axis_length = replacinfo.minor_axis_length
        cellinfo.roundness = replacinfo.roundness
        cellinfo.cricle_rate = replacinfo.cricle_rate
        return cellinfo
    def replace_zero(self,cellinfo):
        cellinfo.center_x=0
        cellinfo.center_y=0
        cellinfo.group_type=0
        cellinfo.area = 0
        cellinfo.perimeter = 0
        cellinfo.major_axis_length = 0
        cellinfo.minor_axis_length = 0
        cellinfo.roundness = 0
        cellinfo.cricle_rate = 0
        cellinfo.sharpne = 0
        cellinfo.edge_diff = 0
        cellinfo.avg_gray = 0
        cellinfo.sum_gray = 0
        cellinfo.max_gray = 0
        cellinfo.min_gray = 0
        cellinfo.diff_gray = 0
        cellinfo.background_diff = 0
        cellinfo.cricle_rate = 0
        cellinfo.luminance = 0
        return cellinfo
    def get_shuxing(self, im_shape, same_cell_iou_thred=0.3, bf_img=None, fl1_img=None, fl2_img=None, fcs_output=False):
        compare_map = np.zeros([im_shape[1], im_shape[0], 3])
        growh_num = 17
        for cell_num, cell in enumerate(self.cell_infos):
            for channel in cell.cell_info:
                if channel != None:
                    compare_map[channel.center_x, channel.center_y, channel.local_channel] = cell_num + growh_num
        iou_thred = same_cell_iou_thred
        BF_FL1_FL2_array = compare_map[
            np.where((compare_map[:, :, 0] > 0) & (compare_map[:, :, 1] > 0) & (compare_map[:, :, 2] > 0))]
        BF_FL1_FL2 = (BF_FL1_FL2_array - growh_num).tolist()
        BF, BF_FL1, BF_FL2, FL1, FL2, FL1_FL2 = [], [], [], [], [], []
        compare_map[np.where((compare_map[:, :, 0] > 0) & (compare_map[:, :, 1] > 0) & (compare_map[:, :, 2] > 0))] = 0
        compare_map = compare_map.astype(int)
        for bfchannel_index in compare_map[np.nonzero(compare_map[:, :, 0])]:
            bf_cell_info = self.cell_infos[int(bfchannel_index[0] - growh_num)].get_channel_info("BF")
            bf_roi_extend_box = self.get_RECT(bf_cell_info, im_shape, padding=bf_cell_info.radius)
            bf_box = self.get_RECT(bf_cell_info, im_shape, padding=0)
            channelroi = compare_map[bf_roi_extend_box[0]:bf_roi_extend_box[2],
                         bf_roi_extend_box[1]:bf_roi_extend_box[3]]
            exist_fl1, exist_fl2 = False, False
            fl1_index, fl2_index = -1, -1
            for FL1_channel in channelroi[np.nonzero(channelroi[:, :, 1])]:
                fl1_cell_info = self.cell_infos[FL1_channel[1] - growh_num].get_channel_info("FL1")
                fl1_box = self.get_RECT(fl1_cell_info, im_shape, padding=0)
                if self.compute_IOU(bf_box, fl1_box) > iou_thred:
                    compare_map[int(fl1_cell_info.center_x), int(fl1_cell_info.center_y), 1] = 0
                    exist_fl1 = True
                    fl1_index = FL1_channel[1] - growh_num
                    break
            for FL2_channel in channelroi[np.nonzero(channelroi[:, :, 2])]:
                fl2_cell_info = self.cell_infos[FL2_channel[2] - growh_num].get_channel_info("FL2")
                fl2_box = self.get_RECT(fl2_cell_info, im_shape, padding=0)
                if self.compute_IOU(bf_box, fl2_box) > iou_thred:
                    compare_map[int(fl2_cell_info.center_x), int(fl2_cell_info.center_y), 2] = 0
                    exist_fl2 = True
                    fl2_index = FL2_channel[2] - growh_num
                    break
            compare_map[int(bf_cell_info.center_x), int(bf_cell_info.center_y), 0] = 0
            if exist_fl1 == True and exist_fl2 == True:
                BF_FL1_FL2.append([bfchannel_index[0] - growh_num, fl1_index, fl2_index])
            elif exist_fl1:
                BF_FL1.append([bfchannel_index[0] - growh_num, fl1_index])
            elif exist_fl2:
                BF_FL2.append([bfchannel_index[0] - growh_num, fl2_index])
            else:
                BF.extend([bfchannel_index[0] - growh_num])

        for fl1channel_index in compare_map[np.nonzero(compare_map[:, :, 1])]:
            fl1_cell_info = self.cell_infos[fl1channel_index[1] - growh_num].get_channel_info("FL1")
            fl1_roi_extend_box = self.get_RECT(fl1_cell_info, im_shape, padding=fl1_cell_info.radius)
            fl1_box = self.get_RECT(fl1_cell_info, im_shape, padding=0)
            # cv2.rectangle(im, (fl1_roi_extend_box[0], fl1_roi_extend_box[1]), (fl1_roi_extend_box[2], fl1_roi_extend_box[3]), (0, 0, 255), 2)
            # cv2.rectangle(im, (fl1_box[0], fl1_box[1]), (fl1_box[2], fl1_box[3]), (255, 0, 0), 2)
            channelroi = compare_map[fl1_roi_extend_box[0]:fl1_roi_extend_box[2],
                         fl1_roi_extend_box[1]:fl1_roi_extend_box[3]]
            exist_fl2 = False
            fl2_index = -1
            for FL2_channel in channelroi[np.nonzero(channelroi[:, :, 2])]:
                fl2_cell_info = self.cell_infos[FL2_channel[2] - growh_num].get_channel_info("FL2")
                fl2_box = self.get_RECT(fl2_cell_info, im_shape, padding=0)
                if self.compute_IOU(fl1_box, fl2_box) > iou_thred:
                    compare_map[int(fl2_cell_info.center_x), int(fl2_cell_info.center_y), 2] = 0
                    exist_fl2 = True
                    fl2_index = FL2_channel[2] - growh_num
                    break
            compare_map[int(fl1_cell_info.center_x), int(fl1_cell_info.center_y), 1] = 0
            if exist_fl2 == True:
                FL1_FL2.append([fl1channel_index[1] - growh_num, fl2_index])
            else:
                FL1.extend([fl1channel_index[1] - growh_num])
        BF.extend((compare_map[:, :, 0][np.nonzero(compare_map[:, :, 0])] - growh_num).tolist())
        FL1.extend((compare_map[:, :, 1][np.nonzero(compare_map[:, :, 1])] - growh_num).tolist())
        FL2.extend((compare_map[:, :, 2][np.nonzero(compare_map[:, :, 2])] - growh_num).tolist())
        # print("BF",BF,"FL1",FL1,"FL2",FL2,"BF_FL1",BF_FL1,"BF_FL2",BF_FL2,"FL1_FL2",FL1_FL2)
        # cv2.imwrite(os.path.join(r"H:\test", "compare_map.jpg"), im)
        if fcs_output:
            new_cell_infos = []
            for cell_number in BF:
                cell_shuxing = 1
                bfinfo = self.cell_infos[int(cell_number)].get_channel_info("BF")
                bfinfo.type = cell_shuxing
                fl1info = (bfinfo.copy()).get_det_info(img=fl1_img, xyxy_0=int(bfinfo.tx), xyxy_1=int(bfinfo.ty),
                                                       xyxy_2=int(bfinfo.bx), xyxy_3=int(bfinfo.by))
                fl1info.local_channel = 1
                fl1info=self.replace(fl1info,bfinfo)
                fl2info = (bfinfo.copy()).get_det_info(img=fl2_img, xyxy_0=int(bfinfo.tx), xyxy_1=int(bfinfo.ty),
                                                       xyxy_2=int(bfinfo.bx), xyxy_3=int(bfinfo.by))
                fl2info.local_channel = 2
                fl2info = self.replace(fl2info, bfinfo)
                new_cell_infos.append(cell_struct(BF=bfinfo, FL1=fl1info, FL2=fl2info))
            for cell_number in FL1:
                cell_shuxing = 2
                fl1info = self.cell_infos[int(cell_number)].get_channel_info("FL1")
                fl1info.type = cell_shuxing
                if isinstance(bf_img, np.ndarray):
                    bfinfo = (fl1info.copy()).get_det_info(img=bf_img, xyxy_0=int(fl1info.tx), xyxy_1=int(fl1info.ty),
                                                           xyxy_2=int(fl1info.bx), xyxy_3=int(fl1info.by))
                    bfinfo.local_channel = 0
                    bfinfo = self.replace_zero(bfinfo)
                else:
                    bfinfo = None
                fl2info = (fl1info.copy()).get_det_info(img=fl2_img, xyxy_0=int(fl1info.tx), xyxy_1=int(fl1info.ty),
                                                        xyxy_2=int(fl1info.bx), xyxy_3=int(fl1info.by))
                fl2info.local_channel = 2
                fl2info = self.replace(fl2info, fl1info)
                new_cell_infos.append(cell_struct(BF=bfinfo, FL1=fl1info, FL2=fl2info))
            for cell_number in FL2:
                cell_shuxing = 3
                fl2info = self.cell_infos[int(cell_number)].get_channel_info("FL2")
                fl2info.type = cell_shuxing
                if isinstance(bf_img, np.ndarray):
                    bfinfo = (fl2info.copy()).get_det_info(img=bf_img, xyxy_0=int(fl2info.tx), xyxy_1=int(fl2info.ty),
                                                           xyxy_2=int(fl2info.bx), xyxy_3=int(fl2info.by))
                    bfinfo.local_channel = 0
                    bfinfo = self.replace_zero(bfinfo)
                else:
                    bfinfo = None
                fl1info = (fl2info.copy()).get_det_info(img=fl1_img, xyxy_0=int(fl2info.tx), xyxy_1=int(fl2info.ty),
                                                        xyxy_2=int(fl2info.bx), xyxy_3=int(fl2info.by))
                fl1info.local_channel = 1
                fl1info = self.replace(fl1info, fl2info)
                new_cell_infos.append(cell_struct(BF=bfinfo, FL1=fl1info, FL2=fl2info))
            for cell_list in BF_FL1:
                cell_shuxing = 6
                self.cell_infos[int(cell_list[0])].get_channel_info("BF").type = cell_shuxing
                self.cell_infos[int(cell_list[1])].get_channel_info("FL1").type = cell_shuxing
                fl1info = self.cell_infos[int(cell_list[1])].get_channel_info("FL1")
                fl2info = (fl1info.copy()).get_det_info(img=fl2_img, xyxy_0=int(fl1info.tx), xyxy_1=int(fl1info.ty),
                                                       xyxy_2=int(fl1info.bx), xyxy_3=int(fl1info.by))
                fl2info.local_channel = 2
                fl2info = self.replace(fl2info, fl1info)
                new_cell_infos.append(cell_struct(BF=self.cell_infos[int(cell_list[0])].get_channel_info("BF"),
                                                  FL1=self.cell_infos[int(cell_list[1])].get_channel_info("FL1")
                                                  , FL2=fl2info))
            for cell_list in BF_FL2:
                cell_shuxing = 7
                self.cell_infos[int(cell_list[0])].get_channel_info("BF").type = cell_shuxing
                self.cell_infos[int(cell_list[1])].get_channel_info("FL2").type = cell_shuxing
                fl2info = self.cell_infos[int(cell_list[1])].get_channel_info("FL2")
                fl1info = (fl2info.copy()).get_det_info(img=fl1_img, xyxy_0=int(fl2info.tx), xyxy_1=int(fl2info.ty),
                                                       xyxy_2=int(fl2info.bx), xyxy_3=int(fl2info.by))
                fl1info.local_channel = 1
                fl1info = self.replace(fl1info, fl2info)
                new_cell_infos.append(cell_struct(BF=self.cell_infos[int(cell_list[0])].get_channel_info("BF"),
                                                  FL2=self.cell_infos[int(cell_list[1])].get_channel_info("FL2")
                                                  , FL1=fl1info))
            for cell_list in FL1_FL2:
                cell_shuxing = 10
                self.cell_infos[int(cell_list[0])].get_channel_info("FL1").type = cell_shuxing
                self.cell_infos[int(cell_list[1])].get_channel_info("FL2").type = cell_shuxing
                fl1info = self.cell_infos[int(cell_list[0])].get_channel_info("FL1")
                if isinstance(bf_img, np.ndarray):
                    bfinfo = (fl1info.copy()).get_det_info(img=bf_img, xyxy_0=int(fl1info.tx), xyxy_1=int(fl1info.ty),
                                                           xyxy_2=int(fl1info.bx), xyxy_3=int(fl1info.by))
                    bfinfo.local_channel = 0
                    bfinfo = self.replace_zero(bfinfo)
                else:
                    bfinfo = None
                new_cell_infos.append(cell_struct(FL1=self.cell_infos[int(cell_list[0])].get_channel_info("FL1"),
                                                  FL2=self.cell_infos[int(cell_list[1])].get_channel_info("FL2")
                                                  , BF=bfinfo))
            for cell_list in BF_FL1_FL2:
                cell_shuxing = 16
                self.cell_infos[int(cell_list[0])].get_channel_info("BF").type = cell_shuxing
                self.cell_infos[int(cell_list[1])].get_channel_info("FL1").type = cell_shuxing
                self.cell_infos[int(cell_list[2])].get_channel_info("FL2").type = cell_shuxing
                new_cell_infos.append(cell_struct(FL1=self.cell_infos[int(cell_list[1])].get_channel_info("FL1"),
                                                  FL2=self.cell_infos[int(cell_list[2])].get_channel_info("FL2")
                                                  , BF=self.cell_infos[int(cell_list[0])].get_channel_info("BF")))
        else:
            new_cell_infos = []
            for cell_number in BF:
                cell_shuxing = 1
                self.cell_infos[int(cell_number)].get_channel_info("BF").type = cell_shuxing
                new_cell_infos.append(self.cell_infos[int(cell_number)])
            for cell_number in FL1:
                cell_shuxing = 2
                self.cell_infos[int(cell_number)].get_channel_info("FL1").type = cell_shuxing
                new_cell_infos.append(self.cell_infos[int(cell_number)])
            for cell_number in FL2:
                cell_shuxing = 3
                self.cell_infos[int(cell_number)].get_channel_info("FL2").type = cell_shuxing
                new_cell_infos.append(self.cell_infos[int(cell_number)])
            for cell_list in BF_FL1:
                cell_shuxing = 6
                self.cell_infos[int(cell_list[0])].get_channel_info("BF").type = cell_shuxing
                self.cell_infos[int(cell_list[1])].get_channel_info("FL1").type = cell_shuxing
                new_cell_infos.append(cell_struct(BF=self.cell_infos[int(cell_list[0])].get_channel_info("BF"),
                                                  FL1=self.cell_infos[int(cell_list[1])].get_channel_info("FL1")
                                                  , FL2=None))
            for cell_list in BF_FL2:
                cell_shuxing = 7
                self.cell_infos[int(cell_list[0])].get_channel_info("BF").type = cell_shuxing
                self.cell_infos[int(cell_list[1])].get_channel_info("FL2").type = cell_shuxing
                new_cell_infos.append(cell_struct(BF=self.cell_infos[int(cell_list[0])].get_channel_info("BF"),
                                                  FL2=self.cell_infos[int(cell_list[1])].get_channel_info("FL2")
                                                  , FL1=None))
            for cell_list in FL1_FL2:
                cell_shuxing = 10
                self.cell_infos[int(cell_list[0])].get_channel_info("FL1").type = cell_shuxing
                self.cell_infos[int(cell_list[1])].get_channel_info("FL2").type = cell_shuxing
                new_cell_infos.append(cell_struct(FL1=self.cell_infos[int(cell_list[0])].get_channel_info("FL1"),
                                                  FL2=self.cell_infos[int(cell_list[1])].get_channel_info("FL2")
                                                  , BF=None))
            for cell_list in BF_FL1_FL2:
                cell_shuxing = 16
                self.cell_infos[int(cell_list[0])].get_channel_info("BF").type = cell_shuxing
                self.cell_infos[int(cell_list[1])].get_channel_info("FL1").type = cell_shuxing
                self.cell_infos[int(cell_list[2])].get_channel_info("FL2").type = cell_shuxing
                new_cell_infos.append(cell_struct(FL1=self.cell_infos[int(cell_list[1])].get_channel_info("FL1"),
                                                  FL2=self.cell_infos[int(cell_list[2])].get_channel_info("FL2")
                                                  , BF=self.cell_infos[int(cell_list[0])].get_channel_info("BF")))
        self.cell_infos = new_cell_infos
        return self


## 合并团列表
def write_log(data):
    with open("Run_LOG.txt", "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " " * 5 + str(data) + "\n")


def merge_lists(l):
    def to_graph(l):
        G = networkx.Graph()
        for part in l:
            # each sublist is a bunch of nodes
            G.add_nodes_from(part)
            # it also imlies a number of edges:
            G.add_edges_from(to_edges(part))
        return G

    def to_edges(l):
        """
            treat `l` as a Graph and returns it's edges
            to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
        """
        it = iter(l)
        last = next(it)
        for current in it:
            yield last, current
            last = current

    G = to_graph(l)
    return [list(i) for i in connected_components(G)]


def groupcomputer(center_lists, distance_tred=0):
    new_celllist = center_lists.copy()
    if center_lists.shape[1] < 2:
        return 0, [], new_celllist
    # print(center_lists[:, 0])
    centerlist = center_lists[:, :2].tolist()
    redius = center_lists[:, 2]
    centerlist = np.array(centerlist)
    tree = KDTree(centerlist)
    centerindex = []
    center_aix = []
    topk = min(centerlist.shape[0], 50)
    for ind in range(centerlist.shape[0]):
        dist, index = tree.query([centerlist[ind]], k=topk)
        for num in range(1, topk):
            if dist[0][num] <= redius[ind] + redius[index[0][num]] + distance_tred:
                centerindex.append([index[0][0], index[0][num]])
    sortedlist = merge_lists(centerindex)
    len_list = [len(i) for i in sortedlist]
    for list in sortedlist:
        temp = [(int(centerlist[k][0]), int(centerlist[k][1])) for k in list]
        center_aix.append(temp)
        for k in list:
            new_celllist[k, 3] = len(temp)
    return sum(len_list), center_aix, new_celllist


def get_roimask(result, img):
    mask = img.copy()
    mask[mask > 0] = 1
    result["label_map"] = cv2.bitwise_and(result['label_map'], result['label_map'], mask=mask)
    return result


def mark_hand_center(frame_in, cont):
    max_d = 0
    pt = (0, 0)
    x, y, w, h = cv2.boundingRect(cont)
    for ind_y in range(int(y + 0.3 * h),
                       int(y + 0.8 * h)):  # around 0.25 to 0.6 region of height (Faster calculation with ok results)
        for ind_x in range(int(x + 0.3 * w),
                           int(x + 0.6 * w)):  # around 0.3 to 0.6 region of width (Faster calculation with ok results)
            dist = cv2.pointPolygonTest(cont, (ind_x, ind_y), True)
            if (dist > max_d):
                max_d = int(dist)
                pt = (ind_x, ind_y)
    return frame_in, pt, max_d


def get_cricle_ROI(im, result, thred=15):
    ret, thresh = cv2.threshold(result['label_map'], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh2 = cv2.resize(thresh, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
    cnts = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh2.shape, dtype="uint8")
    frame_in, pt, max_d = mark_hand_center(mask.copy(), c)
    mask = cv2.circle(mask, pt, max_d + thred, 255, -1)
    testmask = cv2.resize(mask, thresh.shape[::-1])
    mask_inv = cv2.bitwise_not(testmask)
    seg_object = cv2.bitwise_and(im, im, mask=testmask)
    return seg_object


def get_squre_ROI(im, result, epsilon=8, dilate_kenel=31, dilate_iter=0):
    ret, thresh = cv2.threshold(result['label_map'], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    approx = cv2.approxPolyDP(c, epsilon, True)
    testmask = cv2.fillPoly(mask, [approx], 255)
    if dilate_iter > 0:
        kernel = np.ones((dilate_kenel, dilate_kenel), np.uint8)
        testmask = cv2.dilate(testmask, kernel, iterations=dilate_iter)
    seg_object = cv2.bitwise_and(im, im, mask=testmask)
    return seg_object


def get_roi(im, result):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    seg_object = cv2.bitwise_and(im, im, mask=result['label_map'])
    return seg_object


def image_concat(divide_image):
    m, n, grid_h, grid_w = [divide_image.shape[0], divide_image.shape[1],  # 每行，每列的图像块数
                            divide_image.shape[2], divide_image.shape[3]]  # 每个图像块的尺寸
    restore_image = np.zeros([m * grid_h, n * grid_w, 3], np.uint8)
    for i in range(m):
        for j in range(n):
            restore_image[i * grid_h:(i + 1) * grid_h, j * grid_w:(j + 1) * grid_w] = divide_image[i, j, :]
    return restore_image


def counter_image(im, labelmap, color=(255, 255, 255), linetrikness=1):
    mask_values = Mask(labelmap).polygons()
    val = mask_values.points
    resultimg = cv2.drawContours(im, val, -1, color, linetrikness)
    return resultimg


def getcellnum(img, imgmask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(imgmask, cv2.MORPH_OPEN, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
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
    return cell_num


class Muti_Thread_get_result(threading.Thread):
    def __init__(self, func, args):
        super(Muti_Thread_get_result, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result  # 如果子线程不使用join方法，此处可能会报没有self.result的错误
        except Exception:
            return None


def compute_roundness(contours):
    # 计算圆度
    a = cv2.contourArea(contours) * 4 * math.pi
    b = math.pow(cv2.arcLength(contours, True), 2)
    if b == 0:
        return 0
    return a / b


def compute_centerpoint(conuter):
    try:
        mom = cv2.moments(conuter)
        pt = [int(mom['m10'] / mom['m00']), int(mom['m01'] / mom['m00'])]
    except:
        x, y, w, h = cv2.boundingRect(conuter)
        pt = [int((x + w) / 2), int((y + h) / 2)]
    return pt


def getcirclepoint(conuter):
    x, y, w, h = cv2.boundingRect(conuter)
    pt = [int((x + w) / 2), int((y + h) / 2)]
    return pt


def opencv_computer_type(img, im0s_gray, imgmask, markersend, label):
    # bf_channel=channel_struct(local_channel=0,type=1)
    # bf_channel.get_seg_info(cell_img=imgroi,cell_gray=gray,bg_gray=bg_gray)
    mask = np.zeros(imgmask.shape, dtype="uint8")
    mask[markersend == label] = 1
    maskinv = np.ones(imgmask.shape, dtype="uint8")
    maskinv[markersend == label] = 0
    x, y, w, h = cv2.boundingRect(mask)
    bg_gray = cv2.bitwise_and(im0s_gray, im0s_gray, mask=maskinv)[y:y + h, x:x + w]
    gray = cv2.bitwise_and(im0s_gray, im0s_gray, mask=mask)[y:y + h, x:x + w]
    imgroi = cv2.bitwise_and(img, img, mask=mask)[y:y + h, x:x + w]
    center_x, center_y = getcirclepoint(mask)
    cell = cell_struct(
        BF=channel_struct(local_channel=0, type=1).get_seg_info(cell_img=imgroi, cell_gray=gray, bg_gray=bg_gray,
                                                                center_x=center_x, center_y=center_y), FL1=None,
        FL2=None)
    # cv2.circle(img,(cell.get_channel_info(0).center_x,cell.get_channel_info(0).center_y),int(cell.get_channel_info(0).radius),(255,0,0),5)
    return cell


def opencv_computer_other(img, im0s_gray, imgmask, markersend, label):
    # im0s_gray, imgmask, markersend, label=paremlist[0],paremlist[1],paremlist[2],paremlist[3]
    mask = np.zeros(imgmask.shape, dtype="uint8")
    mask[markersend == label] = 1
    maskinv = np.ones(imgmask.shape, dtype="uint8")
    maskinv[markersend == label] = 0
    # # 检测到遮罩中的轮廓   并提取最大的轮廓-该轮廓将代表图像中给定对象的轮廓/边界。
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    maxc = max(cnts, key=cv2.contourArea)
    max_index = cnts.index(max(cnts, key=cv2.contourArea))
    pt = compute_centerpoint(maxc)
    # # 绘制围绕对象的包围圆边界。我们还可以计算对象的边界框，应用按位运算，并提取每个单独的对象。
    # ((x, y), r) = cv2.minEnclosingCircle(c)
    # # cv2.circle(img, (int(x), int(y)), int(r), (0, 255, 0), 2)
    # cv2.drawContours(img, cnts, max_index, (220,220,220), 2)
    # cv2.putText(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "#{}".format(label), (int(x) - 10, int(y)),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    x, y, w, h = cv2.boundingRect(maxc)
    gray = cv2.bitwise_and(im0s_gray, im0s_gray, mask=mask)[y:y + h, x:x + w]
    imgroi = cv2.bitwise_and(img, img, mask=mask)[y:y + h, x:x + w]
    liangdu = np.mean(cv2.cvtColor(imgroi, cv2.COLOR_BGR2HSV)[:, :, 2])
    bg_gray = cv2.bitwise_and(im0s_gray, im0s_gray, mask=maskinv)[y:y + h, x:x + w]
    edge_diff = np.max(bg_gray) - np.min(bg_gray)
    bg_diff = int(np.max(gray)) - int(np.max(bg_gray))
    # cv2.imwrite(r"H:\test\%s_one.jpg"%label,gray)
    # cv2.imwrite(r"H:\test\%s_two.jpg"%label,bg_gray)
    # gray = cv2.cvtColor(seg_object, cv2.COLOR_BGR2GRAY)
    sortgray = np.sort(np.unique(gray.flatten()))
    mingray = sortgray[1] if len(sortgray) > 1 else sortgray[0]
    sharpne = cv2.Laplacian(gray, cv2.CV_64F).var()
    if len(cnts):
        area = cv2.contourArea(maxc)  # 面积
        perimeter = float(cv2.arcLength(maxc, True))
        result = [pt[0], pt[1], (w + h) / 4, 1, 1, 0, area, perimeter, max(w, h), min(w, h), compute_roundness(maxc),
                  sharpne, edge_diff, np.mean(gray), np.sum(gray),
                  np.max(gray), mingray, np.max(gray) - mingray, bg_diff,
                  (area * 4) / (math.pi * math.pow(max(w, h), 2)), liangdu]
    else:
        result = [pt[0], pt[1], (w + h) / 4, 1, 1, 0, w * h, 2 * (w + h), max(w, h), min(w, h), w / h, sharpne,
                  edge_diff, np.mean(gray), np.sum(gray),
                  np.max(gray), mingray, np.max(gray) - mingray, bg_diff,
                  (w * h * 4) / (math.pi * math.pow(max(w, h), 2)), liangdu]
    return result


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def is_pic(img_name):
    """判断是否是图片
    参数：
        img_name (str): 图片路径
    返回：
        flag (bool): 判断值。
    """
    valid_suffix = ['JPEG', 'jpeg', 'JPG', 'jpg', 'BMP', 'bmp', 'PNG', 'png']
    suffix = img_name.split('.')[-1]
    flag = True
    if suffix not in valid_suffix:
        flag = False
    return flag

def save_resultimage(img,save_path):
    try:
        if not os.path.exists(os.path.dirname(save_path)):
           os.makedirs(os.path.dirname(save_path))
    except Exception as e:
        print(e)
    cv2.imencode('.jpg', img)[1].tofile(save_path)
def get_color_map_list(num_classes):
    """ Returns the color map for visualizing the segmentation mask,
        which can support arbitrary number of classes.
    Args:
        num_classes: Number of classes
    Returns:
        The color map
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    return color_map


def visualize(image,
              result,
              weight=0.6,
              save_path='./',
              color=None):
    label_map = result['label_map']
    color_map = get_color_map_list(256)
    if color is not None:
        for i in range(len(color) // 3):
            color_map[i] = color[i * 3:(i + 1) * 3]
    color_map = np.array(color_map).astype("uint8")

    # Use OpenCV LUT for color mapping
    c1 = cv2.LUT(label_map, color_map[:, 0])
    c2 = cv2.LUT(label_map, color_map[:, 1])
    c3 = cv2.LUT(label_map, color_map[:, 2])
    pseudo_img = np.dstack((c1, c2, c3))

    if isinstance(image, np.ndarray):
        im = image
        image_name = str(int(time.time() * 1000)) + '.jpg'
        if image.shape[2] != 3:
            print(
                "The image is not 3-channel array, so predicted label map is shown as a pseudo color image."
            )
            weight = 0.
    else:
        image_name = os.path.split(image)[-1]
        if not is_pic(image):
            print(
                "The image cannot be opened by opencv, so predicted label map is shown as a pseudo color image."
            )
            image_name = image_name.split('.')[0] + '.jpg'
            weight = 0.
        else:
            im = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)

    if abs(weight) < 1e-5:
        vis_result = pseudo_img
    else:
        vis_result = cv2.addWeighted(im, weight, pseudo_img.astype(im.dtype), 1 - weight, 0)
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        # out_path = os.path.join(save_dir, 'visualize_{}'.format(image_name))
        # cv2.imwrite(save_path, vis_result)
        cv2.imencode('.jpg', vis_result)[1].tofile(save_path)
        print('The visualized result is saved as {}'.format(save_path))
    else:
        return vis_result


def crop_rotate_box(cnt, img):
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    width = int(rect[1][0])
    height = int(rect[1][1])
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped = cv2.warpPerspective(img, M, (width, height))
    return warped


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(image.shape[2]):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def divide_method(img, m, n):  # 分割成m行n列
    h, w = img.shape[0], img.shape[1]
    grid_h = int(h * 1.0 / (m - 1) + 0.5)  # 每个网格的高
    grid_w = int(w * 1.0 / (n - 1) + 0.5)  # 每个网格的宽

    # 满足整除关系时的高、宽
    h = grid_h * (m - 1)
    w = grid_w * (n - 1)

    # 图像缩放
    img_re = cv2.resize(img, (w, h),
                        cv2.INTER_LINEAR)  # 也可以用img_re=skimage.transform.resize(img, (h,w)).astype(np.uint8)
    # plt.imshow(img_re)
    gx, gy = np.meshgrid(np.linspace(0, w, n), np.linspace(0, h, m))
    gx = gx.astype(np.int32)
    gy = gy.astype(np.int32)
    divide_image = np.zeros([m - 1, n - 1, grid_h, grid_w, 3],
                            np.uint8)  # 这是一个五维的张量，前面两维表示分块后图像的位置（第m行，第n列），后面三维表示每个分块后的图像信息

    for i in range(m - 1):
        for j in range(n - 1):
            divide_image[i, j, ...] = img_re[
                                      gy[i][j]:gy[i + 1][j + 1], gx[i][j]:gx[i + 1][j + 1], :]
    return divide_image


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # top left x
    y[:, 1] = h * x[:, 1] + padh  # top left y
    return y


def segment2box(segment, width=640, height=640):
    # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy


def segments2boxes(segments):
    # Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def resample_segments(segments, n=1000):
    # Up-sample an (n,2) segment
    for i, s in enumerate(segments):
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
    return segments


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def mask_image(im, labelmap, add_merge=True, aplha=0.3):
    color_map = get_color_map_list(256)
    color_map = np.array(color_map).astype("uint8")
    # # Use OpenCV LUT for color mapping
    try:
        c1 = cv2.LUT(labelmap, color_map[:, 0])
    except:
        labelmap = img_as_ubyte(labelmap)
    c1 = cv2.LUT(labelmap, color_map[:, 0])
    c2 = cv2.LUT(labelmap, color_map[:, 1])
    c3 = cv2.LUT(labelmap, color_map[:, 2])
    pseudo_img = np.dstack((c1, c2, c3))
    if add_merge:
        resultimg = im + pseudo_img.astype(im.dtype) * 0.4
    else:
        resultimg = cv2.addWeighted(im, aplha, pseudo_img.astype(im.dtype), 1 - aplha, 0)
    return resultimg


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def get_centerpoint(lis):
    _x_list = [vertex[0] for vertex in lis]
    _y_list = [vertex[1] for vertex in lis]
    _len = len(lis)
    _x = sum(_x_list) / _len
    _y = sum(_y_list) / _len
    return int(_x), int(_y)


def plot_cricle(center, radius, im, color=(128, 128, 128), line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    # cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    cv2.circle(im, center, int(radius), color, tl, cv2.LINE_AA)


def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def use_lnms_nms(detections, conf_thres, iou_thres, cutoff_distance=64):
    pred = detections[0]
    cell_info = np.empty([pred.shape[0], 6])
    cell_info[:, :4] = xywh2xyxy(pred[:, :4])  ##box
    cell_info[:, 4] = pred[:, 4]  ##sorce
    pred[:, 5:] *= pred[:, 4:5]  # conf
    cell_info[:, 5] = np.nanargmax(pred[:, 5:], axis=1)  # label
    try:
        keep = nms(cell_info[:, :4], cell_info[:, 4], iou_threshold=iou_thres, score_threshold=conf_thres,
                   cutoff_distance=cutoff_distance)
        det = cell_info[keep]
        return det
    except:
        return []
