import os
import time
from tkinter import filedialog
from imutils.paths import list_images

device="GPU"
if device == 'GPU':
	from GPU_Predictor import TrtPredictor,YoLov5TRT
	import ctypes
	kuangpredictor = TrtPredictor("seg_model/Seg_D3m_kuang_class__openvino_model_fix/paddle2onnx_model.engine",
								  "seg_model/Seg_D3m_kuang_class__openvino_model_fix/model.yml")
	grouppredictor = TrtPredictor(
		"seg_model/Seg_D3m_dankelong_huizhong0415_1024_openvino_model_fix/paddle2onnx_model.engine",
		"seg_model/Seg_D3m_dankelong_huizhong0415_1024_openvino_model_fix/model.yml")
	chopredictor = TrtPredictor(
		"seg_model/Seg_unet_CHO115huizong_add_1115ma__openvino_model_fix/paddle2onnx_model.engine",
		"seg_model/Seg_unet_CHO115huizong_add_1115ma__openvino_model_fix/model.yml")
	zuanranpredictor = TrtPredictor("seg_model/Seg2_D3m_zuanran_agg_2_openvino_model_fix/paddle2onnx_model.engine",
									"seg_model/Seg2_D3m_zuanran_agg_2_openvino_model_fix/model.yml")
	TB_predictor=YoLov5TRT("det_model/taipanlan_huizong_20220429_agg_y5s/best.engine")
	BF2_predictor=YoLov5TRT("det_model/2022051mingchang_agg/best.engine")
	FL2_predictor=YoLov5TRT("det_model/2022051yinguang_agg/best.engine")
	from detection_inference import detmodel_iference
	from seg_inference import segmodel_iference
else:
	from Predictor import Predictor
	from detection_inference import load_ovdet_models, detmodel_iference
	from seg_inference import segmodel_iference
	import ctypes

	TB_predictor = load_ovdet_models(weights="det_model/taipanlan_huizong_20220429_agg_y5s/best.xml")
	# BF_predictor=load_ovdet_models(device=device,weights="det_model/BF_openvino_1130_640/best-sim.xml")
	# BF2_predictor=load_ovdet_models(device=device,weights="det_model/weizhu_and_beads_mingchang_agg/best.xml")
	BF2_predictor = load_ovdet_models( weights="det_model/2022051mingchang_agg/best.xml")
	#FL_predictor = load_ovdet_models(device=device, weights="det_model/dangyingguang_openvino_1130_640/best-sim.xml")
	FL2_predictor = load_ovdet_models(weights="det_model/2022051yinguang_agg/best.xml")
	# FL2_predictor=load_ovdet_models(device=device,weights="det_model/yingguangbuchong_agg/best.xml")
	# r_M_predictor=load_ovdet_models(device=device,weights="det_model/ronghe_yolov5_openvino_640/best-sim.xml")
	kuangpredictor = Predictor("seg_model/Seg_D3m_kuang_class__openvino_model_fix/paddle2onnx_model.xml",
							   "seg_model/Seg_D3m_kuang_class__openvino_model_fix/model.yml")
	grouppredictor = Predictor("seg_model/Seg_D3m_dankelong_huizhong0415_1024_openvino_model_fix/paddle2onnx_model.xml",
							   "seg_model/Seg_D3m_dankelong_huizhong0415_1024_openvino_model_fix/model.yml")
	chopredictor = Predictor("seg_model/Seg_unet_CHO115huizong_add_1115ma__openvino_model_fix/paddle2onnx_model.xml",
							 "seg_model/Seg_unet_CHO115huizong_add_1115ma__openvino_model_fix/model.yml")
	zuanranpredictor = Predictor("seg_model/Seg2_D3m_zuanran_agg_2_openvino_model_fix/paddle2onnx_model.xml",
								 "seg_model/Seg2_D3m_zuanran_agg_2_openvino_model_fix/model.yml")

# DLL库调用接口封装类
class SingleCloneDllHandle():
	# dll，所有对象共有
	_dll = None

	# 初始化
	def __init__(self, dll_path, model_path=None):
		if self._dll == None:
			self._dll = ctypes.cdll.LoadLibrary(dll_path)
		if model_path and type(model_path) == str:
			self._dll.loadModel(model_path.encode())

	# 克隆团识别
	def doPredict(self, input_image_path, image_save_path, data_save_path, pre_data_path, thrd_wh=1.2, thrd_defect=1.5,
				  radius=0):
		# 调用dll中doPredict函数，进行识别运算
		ret = self._dll.doPredict(
			input_image_path.encode('GBK'),
			image_save_path.encode('GBK'),
			data_save_path.encode('GBK'),
			pre_data_path.encode('GBK'),
			ctypes.c_float(thrd_wh),
			ctypes.c_float(thrd_defect),
			ctypes.c_int(radius)
		)
		return ret

	# 释放内存
	def release(self):
		self._dll.cleanup()

if __name__ == '__main__':

	detentinon=True
	while detentinon:
		select = int(input("请输入检测模式：1.AI克隆团算法 2.AI汇合度算法 3.AI转染算法 4.AI台盼蓝算法 5.AI明场算法 6.AI荧光算法 7.AI融合算法 8.Z克隆团算法 \n"))
		#select=1
		if select==1: # 克隆团算法
			# originpath=r"H:\test\多克隆团识别"
			# savepath=r"H:\test"
			originpath = filedialog.askdirectory(title="请选择要处理的文件夹路径")
			savepath = filedialog.askdirectory(title="请选择结果保存路径")
			cut_distance = input("请输入单克隆团切割阈值，值越大分割块数越少(默认为120)")
			#cut_distance=120
			if cut_distance == "":
				cut_distance = 120
			else:
				cut_distance = abs(int(cut_distance))
			print("originpath",originpath, "savepath", savepath, "cut_distance", cut_distance)
			for im_path in list(list_images(originpath)):
				print("处理图片: %s"%im_path)
				segmodel_iference(kuangpredictor, grouppredictor, chopredictor, zuanranpredictor, "single_clone", im_path,
						  os.path.join(savepath,im_path.split(os.sep)[-1][:-4]+"_result.jpg"), os.path.join(savepath,im_path.split(os.sep)[-1][:-4]+"_data.csv"), "", cut_patch=3, fastmode=False,
						  mab_min_distance=cut_distance)
		if select==2: # 汇合度算法
			start_time=time.time()
			originpath = filedialog.askdirectory(title="请选择要处理的文件夹路径")
			savepath = filedialog.askdirectory(title="请选择结果保存路径")
			cut_patch = input("请输入切块个数:1. 1x1 2. 2x2 3. 3x3 4. 4x4 5.5x5 6.6x6 7.7x7(回车默认使用3x3)  ")
			if cut_patch == "":
				cut_patch = 3
			else:
				cut_patch = int(cut_patch)
			cell_stistis = int(input("检测模式：1.汇合度,2.汇合度加个数统计分析 "))
			if cell_stistis==1:
				fast_mode=True
			else:
				fast_mode=False
			print("处理 ","originpath", originpath, "savepath", savepath, "cut_patch", cut_patch,"fast_mode", fast_mode)
			for im_path in list(list_images(originpath)):
				print("处理图片: %s"%im_path)
				segmodel_iference(kuangpredictor, grouppredictor, chopredictor, zuanranpredictor, "confluence",
								  im_path, os.path.join(savepath,im_path.split(os.sep)[-1][:-4]+"_result.jpg"), os.path.join(savepath,im_path.split(os.sep)[-1][:-4]+"_data.csv"), "",
								  cut_patch=cut_patch, fastmode=fast_mode)
			print("cost time %.2f"%(time.time()-start_time))

		if select==3: # 转染算法
			askcontunu=True
			while askcontunu:
				print("需要使用单通道模式在选择另一个通道的图片的时,不用选择图片")
				BFpath = filedialog.askopenfilename(title="请选择BF图片")
				FL1path = filedialog.askopenfilename(title="请选择要FL1图片")
				FL2path = filedialog.askopenfilename(title="请选择要FL2图片")
				savepath=filedialog.askdirectory(title="请选择结果保存路径")
				print("savepath",savepath)
				cut_patch = input("请输入切块个数:1. 1x1 2. 2x2 3. 3x3 4. 4x4 5.5x5 6.6x6 7.7x7(回车默认使用3x3) ")
				if cut_patch == "":
					cut_patch = 3
				else:
					cut_patch = int(cut_patch)
				print("BF_path",BFpath,"FL_path",FL1path,"FL2_path",FL2path,"cut_patch",cut_patch)
				segmodel_iference(kuangpredictor, grouppredictor, chopredictor, zuanranpredictor, "transfection",
								  [BFpath, FL1path, FL2path],
								  [os.path.join(savepath,BFpath.split("/")[-1][:-4]+"_BF.jpg"), os.path.join(savepath,BFpath.split("/")[-1][:-4]+"_FL1.jpg"), os.path.join(savepath,BFpath.split("/")[-1][:-4]+"_FL2.jpg"), os.path.join(savepath,BFpath.split("/")[-1][:-4]+"_merge.jpg")],
								  os.path.join(savepath,BFpath.split("/")[-1][:-4]+"_data.csv"), os.path.join(savepath,BFpath.split("/")[-1][:-4]+"_fcs.csv"), cut_patch=cut_patch, fastmode=False)
				askcontunu=input("是否继续？(y/n)")
				if askcontunu=="n":
					askcontunu=False
		if select==4: #AI台盼蓝算法
			originpath = filedialog.askdirectory(title="请选择要处理的文件夹路径")
			savepath = filedialog.askdirectory(title="请选择结果保存路径")
			conf_thrd = input("请输入检测器置信度阈值(回车使用默认值0.1)：")
			if conf_thrd == "":
				conf_thrd = 0.1
			else:
				conf_thrd = float(conf_thrd)
			iou_thrd = input("请输入检测器iou阈值(回车使用默认值0.1)：")
			if iou_thrd == "":
				iou_thrd = 0.1
			else:
				iou_thrd = float(iou_thrd)
			print("处理 ","originpath",originpath,"savepath",savepath,"检测器置信度阈值：", conf_thrd, "检测器IOU阈值：", iou_thrd)
			for im_path in list(list_images(originpath)):
				print("处理图片: %s" % im_path)
				detmodel_iference(input_im_path=im_path,
								  image_save_path=os.path.join(savepath, im_path.split(os.sep)[-1][:-4] + "_result.jpg"),
								  data_save_path=os.path.join(savepath, im_path.split(os.sep)[-1][:-4] + "_data.csv"),
								  fcs_save_path="",
								  cut_size=640, conf_thres=conf_thrd,
								  iou_thres=iou_thrd, typeblue_model=TB_predictor, extype="trypanblue")

		if select==5: #AI明场算法
			originpath = filedialog.askdirectory(title="请选择要处理的文件夹路径")
			savepath = filedialog.askdirectory(title="请选择结果保存路径")
			conf_thrd = input("请输入检测器置信度阈值(回车使用默认值0.1)：")
			if conf_thrd == "":
				conf_thrd = 0.1
			else:
				conf_thrd = float(conf_thrd)
			iou_thrd = input("请输入检测器iou阈值(回车使用默认值0.1)：")
			if iou_thrd == "":
				iou_thrd = 0.1
			else:
				iou_thrd = float(iou_thrd)
			print("originpath",originpath,"savepath",savepath,"检测器置信度阈值：", conf_thrd, "检测器IOU阈值：", iou_thrd)
			for im_path in list(list_images(originpath)):
				print("处理图片: %s"%im_path)
				detmodel_iference(input_im_path=im_path, image_save_path=os.path.join(savepath,im_path.split(os.sep)[-1][:-4]+"_result.jpg"),
								  data_save_path=os.path.join(savepath,im_path.split(os.sep)[-1][:-4]+"_data.csv"), fcs_save_path="", conf_thres=conf_thrd,
								  cut_size=640, iou_thres=iou_thrd, BFdetection_model=BF2_predictor, extype="suspension_BF")
		if select==6: #AI荧光算法
			askcontunu=True
			while askcontunu:
				print("需要使用单通道模式时在选择另一个通道的图片的时,不用选择图片")
				FL1path = filedialog.askopenfilename(title="请选择要FL1图片")
				FL2path = filedialog.askopenfilename(title="请选择要FL2图片")
				savepath=filedialog.askdirectory(title="请选择结果保存路径")
				FL1_conf_thrd = input("请输入FL1检测器置信度阈值(回车使用默认值0.1)：")
				if FL1_conf_thrd == "":
					FL1_conf_thrd = 0.1
				else:
					FL1_conf_thrd = float(FL1_conf_thrd)
				FL1_iou_thrd = input("请输入FL1检测器iou阈值(回车使用默认值0.1)：")
				if FL1_iou_thrd == "":
					FL1_iou_thrd = 0.1
				else:
					FL1_iou_thrd = float(FL1_iou_thrd)
				FL2_conf_thrd = input("请输入FL2检测器置信度阈值(回车使用默认值0.1)：")
				if FL2_conf_thrd == "":
					FL2_conf_thrd = 0.1
				else:
					FL2_conf_thrd = float(FL2_conf_thrd)
				FL2_iou_thrd = input("请输入FL2检测器iou阈值(回车使用默认值0.1)：")
				if FL2_iou_thrd == "":
					FL2_iou_thrd = 0.1
				else:
					FL2_iou_thrd = float(FL2_iou_thrd)
				print("处理 ","FL1path",FL1path,"FL2path",FL2path,"FL1检测器置信度阈值：", FL1_conf_thrd, "FL1检测器IOU阈值：", FL1_iou_thrd
					  ,"FL2检测器置信度阈值：", FL2_conf_thrd, "FL2检测器IOU阈值：", FL2_iou_thrd)
				detmodel_iference(input_im_path=[FL1path, FL2path],
								  image_save_path=[os.path.join(savepath,FL1path.split("/")[-1][:-4]+"_FL1.jpg"), os.path.join(savepath,FL2path.split("/")[-1][:-4]+"_FL2.jpg"), os.path.join(savepath,"FL_merge.jpg")],
								  data_save_path=os.path.join(savepath,"AI_data.csv"), fcs_save_path=os.path.join(savepath,"FCS_data.csv"), FL1_conf_thrd=FL1_conf_thrd,FL1_iou_thrd=FL1_iou_thrd,FL2_conf_thrd=FL2_conf_thrd,FL2_iou_thrd=FL2_iou_thrd,
								  cut_size=640,FLdeteciton_model=FL2_predictor, extype="suspension_FL")
				askcontunu=input("是否继续？(y/n)")
				if askcontunu=="n":
					askcontunu=False
		if select == 7: #AI融合算法
			askcontunu = True
			while askcontunu:
				print("需要使用单通道模式时在选择另一个通道的图片的时,不用选择图片")
				BFpath = filedialog.askopenfilename(title="请选择BF图片")
				FL1path = filedialog.askopenfilename(title="请选择要FL1图片")
				FL2path = filedialog.askopenfilename(title="请选择要FL2图片")
				savepath = filedialog.askdirectory(title="请选择结果保存路径")
				BF_conf_thrd = input("请输入BF检测器置信度阈值(回车使用默认值0.1)：")
				if BF_conf_thrd == "":
					BF_conf_thrd = 0.1
				else:
					BF_conf_thrd = float(BF_conf_thrd)
				BF_iou_thrd = input("请输入BF检测器iou阈值(回车使用默认值0.1)：")
				if BF_iou_thrd == "":
					BF_iou_thrd = 0.1
				else:
					BF_iou_thrd = float(BF_iou_thrd)
				FL1_conf_thrd = input("请输入FL1检测器置信度阈值(回车使用默认值0.1)：")
				if FL1_conf_thrd == "":
					FL1_conf_thrd = 0.1
				else:
					FL1_conf_thrd = float(FL1_conf_thrd)
				FL1_iou_thrd = input("请输入FL1检测器iou阈值(回车使用默认值0.1)：")
				if FL1_iou_thrd == "":
					FL1_iou_thrd = 0.1
				else:
					FL1_iou_thrd = float(FL1_iou_thrd)
				FL2_conf_thrd = input("请输入FL2检测器置信度阈值(回车使用默认值0.1)：")
				if FL2_conf_thrd == "":
					FL2_conf_thrd = 0.1
				else:
					FL2_conf_thrd = float(FL2_conf_thrd)
				FL2_iou_thrd = input("请输入FL2检测器iou阈值(回车使用默认值0.1)：")
				if FL2_iou_thrd == "":
					FL2_iou_thrd = 0.1
				else:
					FL2_iou_thrd = float(FL2_iou_thrd)
				print("处理 ","BFpath",BFpath,"FL1path",FL1path,"FL2path",FL2path,"BF检测器置信度阈值：", BF_conf_thrd, "BF检测器IOU阈值：", BF_iou_thrd
					  ,"FL1检测器置信度阈值：", FL1_conf_thrd, "FL1检测器IOU阈值：", FL1_iou_thrd
					  ,"FL2检测器置信度阈值：", FL2_conf_thrd, "FL2检测器IOU阈值：", FL2_iou_thrd)
				detmodel_iference(input_im_path=[BFpath, FL1path, FL2path],
								  image_save_path=[os.path.join(savepath,BFpath.split("/")[-1][:-4]+"_BF.jpg"), os.path.join(savepath,BFpath.split("/")[-1][:-4]+"_FL1.jpg"), os.path.join(savepath,BFpath.split("/")[-1][:-4]+"_FL2.jpg"),
												   os.path.join(savepath,BFpath.split("/")[-1][:-4]+"_merge.jpg")], data_save_path=os.path.join(savepath,BFpath.split("/")[-1][:-4]+"_data.csv"),
								  fcs_save_path=os.path.join(savepath,BFpath.split("/")[-1][:-4]+"_fcs.csv"), cut_size=640, BF_conf_thrd=BF_conf_thrd,
								  BF_iou_thrd=BF_iou_thrd,FL1_conf_thrd=FL1_conf_thrd,FL1_iou_thrd=FL1_iou_thrd,FL2_conf_thrd=FL2_conf_thrd,FL2_iou_thrd=FL2_iou_thrd, BFdetection_model=BF2_predictor, FLdeteciton_model=FL2_predictor,
								  extype="AOPI_merge")

				askcontunu = input("是否继续？(y/n)")
				if askcontunu == "n":
					askcontunu = False
		if select==8:
			dll_path = "SingleCloneDll/SingleCloneAiDll.dll"
			model_path = "model/v1-acc-0418"
			single_clone_handle = SingleCloneDllHandle(dll_path, model_path)  # 创建dll handle
			originpath = filedialog.askdirectory(title="请选择要处理的文件夹路径")
			savepath = filedialog.askdirectory(title="请选择结果保存路径")
			defectRatio = input("请输入单凹陷阈值，为克隆团边缘凹陷区域面积与克隆团总面积的比值。大于该值的凹陷处进行分割(默认为1.5)")
			#cut_distance=120
			if defectRatio == "":
				defectRatio = 1.5
			else:
				defectRatio = abs(float(defectRatio))
			aspectRatio = input("请输入克隆团长宽比阈值，长宽比小于该值的克隆团不进行分割处理。取值：大于1(默认为1.2)")
			if aspectRatio == "":
				aspectRatio = 1.2
			else:
				cut_distance = abs(float(aspectRatio))
			print("originpath",originpath, "savepath", savepath)
			for im_path in list(list_images(originpath)):
				print("处理图片: %s"%im_path)
				try:
					ret=single_clone_handle.doPredict(
						im_path,
						os.path.join(savepath,im_path.split(os.sep)[-1][:-4]+"_result.jpg"),
						os.path.join(savepath,im_path.split(os.sep)[-1][:-4]+"_data.csv"),
						"",
						thrd_defect=defectRatio,
						thrd_wh=aspectRatio,
						radius=0
					)
					print("ret",im_path,os.path.join(savepath,im_path.split(os.sep)[-1][:-4]+"_result.jpg"),os.path.join(savepath,im_path.split(os.sep)[-1][:-4]+"_data.csv"),ret)
				except Exception as e:
					print(e)
			single_clone_handle.release()
			askcontunu = input("是否继续？(y/n)")
			if askcontunu == "n":
				askcontunu = False
		replay = input("是否继续其他测试？(y/n)")
		if replay == "n":
			replay = False
	print("连接已关闭,程序退出")


