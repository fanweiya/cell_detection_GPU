import ctypes
import json
import traceback
import pika
import os
import time
from datetime import datetime
from six import StringIO

# import psutil
# import subprocess
# kill掉进程
# def check_process_exit(process_name):
#     pl = psutil.pids()
#     for pid in pl:
#         if psutil.Process(pid).name() == process_name:
#             print("%s pid is %s," % (psutil.Process(pid).name(), pid))
#             # kill进程
#             find_kill = 'taskkill -f -pid %s' % (pid)
#             p=subprocess.run(find_kill, stdout=subprocess.PIPE)
#             output = p.stdout.decode('gbk')
#             print("%s" % output)
#             print("Success: the process with PID %s has stopped" % pid)
#             return True
#     print("%s is not exist!"%process_name)
#     return False
def getnowtime():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def writequeue(data):
    send_credentials = pika.PlainCredentials('guest', 'guest')
    send_connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost',
                                                                        port=5672,
                                                                        virtual_host='/',
                                                                        heartbeat=0,
                                                                        credentials=send_credentials))
    send_channel = send_connection.channel()
    send_channel.queue_declare(queue='done-list', durable=True)
    message = json.dumps(data)
    send_channel.basic_publish(exchange='',
                               routing_key='done-list',
                               body=message.encode())
    # # 关闭与rabbitmq server的连接
    send_connection.close()


def write_log(data):
    with open("Run_LOG.txt", "a+") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " " + str(data) + "\n")


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
    def doPredict(self, input_image_path, image_save_path, data_save_path, pre_data_path, thrd_wh=1.2, thrd_defect=1.5,radius=0):
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
    # check_process_exit("rmq_main.exe")
    if not os.path.exists('Run_LOG.txt'):
        print("create Log file")
        file = open('Run_LOG.txt', 'w')
        write_log("start running")
        file.close()
    elif os.path.getsize('Run_LOG.txt') > 100 * 1000000:  # 大于100M删除
        print("Log file size out 100M,remove it")
        os.remove("Run_LOG.txt")
        file = open('Run_LOG.txt', 'w')
        write_log("start running")
        file.close()
    else:
        pass
    device = "GPU"
    if device == 'GPU':
        from GPU_Predictor import TrtPredictor, YoLov5TRT
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
        TB_predictor = YoLov5TRT("det_model/taipanlan_huizong_20220429_agg_y5s/best.engine")
        BF2_predictor = YoLov5TRT("det_model/2022051mingchang_agg/best.engine")
        FL2_predictor = YoLov5TRT("det_model/2022051yinguang_agg/best.engine")
    else:
        from Predictor import Predictor
        from detection_inference import load_ovdet_models, detmodel_iference
        from seg_inference import segmodel_iference
        import ctypes

        TB_predictor = load_ovdet_models(weights="det_model/taipanlan_huizong_20220429_agg_y5s/best.xml")
        # BF_predictor=load_ovdet_models(device=device,weights="det_model/BF_openvino_1130_640/best-sim.xml")
        # BF2_predictor=load_ovdet_models(device=device,weights="det_model/weizhu_and_beads_mingchang_agg/best.xml")
        BF2_predictor = load_ovdet_models(weights="det_model/2022051mingchang_agg/best.xml")
        # FL_predictor = load_ovdet_models(device=device, weights="det_model/dangyingguang_openvino_1130_640/best-sim.xml")
        FL2_predictor = load_ovdet_models(weights="det_model/2022051yinguang_agg/best.xml")
        # FL2_predictor=load_ovdet_models(device=device,weights="det_model/yingguangbuchong_agg/best.xml")
        # r_M_predictor=load_ovdet_models(device=device,weights="det_model/ronghe_yolov5_openvino_640/best-sim.xml")
        kuangpredictor = Predictor("seg_model/Seg_D3m_kuang_class__openvino_model_fix/paddle2onnx_model.xml",
                                   "seg_model/Seg_D3m_kuang_class__openvino_model_fix/model.yml")
        grouppredictor = Predictor(
            "seg_model/Seg_D3m_dankelong_huizhong0415_1024_openvino_model_fix/paddle2onnx_model.xml",
            "seg_model/Seg_D3m_dankelong_huizhong0415_1024_openvino_model_fix/model.yml")
        chopredictor = Predictor(
            "seg_model/Seg_unet_CHO115huizong_add_1115ma__openvino_model_fix/paddle2onnx_model.xml",
            "seg_model/Seg_unet_CHO115huizong_add_1115ma__openvino_model_fix/model.yml")
        zuanranpredictor = Predictor("seg_model/Seg2_D3m_zuanran_agg_2_openvino_model_fix/paddle2onnx_model.xml",
                                     "seg_model/Seg2_D3m_zuanran_agg_2_openvino_model_fix/model.yml")
    dll_path = "SingleCloneDll/SingleCloneAiDll.dll"
    model_path = "model/v1-acc-0418"
    single_clone_handle = SingleCloneDllHandle(dll_path, model_path)  # 创建dll handle
    # warm_loadnms()
    def processing_image(ch, method, properties, body):
        # 手动发送确认消息
        global task_id
        createTime = getnowtime()
        task_id = ' '
        sucess_msg = "sucess"
        status = "sucess"
        try:
            messge = json.loads(body.decode())
            write_log("[start processing]: " + str(messge))
            task_id = messge["tid"]
            exp_type = messge["exp_type"]
            if exp_type == "version":
                sucess_msg = "V1.2.7"
            elif exp_type == "single_cloneCV":
                print(exp_type, task_id)
                input_image_path = messge["input_image_path"]
                image_save_path = messge["image_save_path"]
                data_save_path = messge["data_save_path"]
                try:
                    fcs_save_path = messge["fcs_save_path"]
                except:
                    fcs_save_path = ""
                try:
                    defectRatio = float(messge["defectRatio"])
                except:
                    defectRatio = 1.5
                try:
                    aspectRatio = float(messge["aspectRatio"])
                except:
                    aspectRatio = 1.2
                try:
                    ret = single_clone_handle.doPredict(
                        input_image_path,
                        image_save_path,
                        data_save_path,
                        "",
                        thrd_wh=aspectRatio,
                        thrd_defect=defectRatio,
                        radius=0,
                    )
                    print("ret:", ret,"input_image_path:",input_image_path,"image_save_path:",image_save_path,"data_save_path:",data_save_path)
                except Exception as e:
                    print(e)
                #single_clone_handle.release()
            elif exp_type == "single_clone":
                print(exp_type, task_id)
                input_image_path = messge["input_image_path"]
                image_save_path = messge["image_save_path"]
                data_save_path = messge["data_save_path"]
                try:
                    fcs_save_path = messge["fcs_save_path"]
                except:
                    fcs_save_path = ""
                try:
                    cut_rhd = int(messge["cut_thrd"])
                except:
                    cut_rhd = 120
                segmodel_iference(kuangpredictor, grouppredictor, chopredictor, zuanranpredictor, exp_type,
                                  input_image_path, image_save_path, data_save_path, fcs_save_path, cut_patch=3,
                                  fastmode=False, mab_min_distance=cut_rhd)
            elif exp_type == "confluence":
                print(exp_type, task_id)
                input_image_path = messge["input_image_path"]
                image_save_path = messge["image_save_path"]
                fastmode_state = messge["fast_mode"]
                data_save_path = messge["data_save_path"]
                try:
                    fcs_save_path = messge["fcs_save_path"]
                except:
                    fcs_save_path = ""
                if fastmode_state == "True":
                    fastmode_state = True
                if fastmode_state == "False":
                    fastmode_state = False
                try:
                    cut_patch_number = int(messge["cut_patch"])
                except:
                    cut_patch_number = 3
                segmodel_iference(kuangpredictor, grouppredictor, chopredictor, zuanranpredictor, exp_type,
                                  input_image_path, image_save_path, data_save_path, fcs_save_path,
                                  cut_patch=cut_patch_number, fastmode=fastmode_state)
            elif exp_type == "transfection":
                print(exp_type, task_id)
                bf_image_path = messge["BF_input_path"]
                fl1_image_path = messge["FL1_input_path"]
                fl2_image_path = messge["FL2_input_path"]
                bf_save_image_path = messge["BF_save_path"]
                fl1_save_image_path = messge["FL1_save_path"]
                fl2_save_image_path = messge["FL2_save_path"]
                merge_save_path = messge["merge_save_path"]
                data_save_path = messge["data_save_path"]
                try:
                    fcs_save_path = messge["fcs_save_path"]
                except:
                    fcs_save_path = ""
                try:
                    cut_patch_number = int(messge["cut_patch"])
                except:
                    cut_patch_number = 3
                segmodel_iference(kuangpredictor, grouppredictor, chopredictor, zuanranpredictor, exp_type,
                                  [bf_image_path, fl1_image_path, fl2_image_path],
                                  [bf_save_image_path, fl1_save_image_path, fl2_save_image_path, merge_save_path],
                                  data_save_path, fcs_save_path, cut_patch=cut_patch_number, fastmode=False)
            elif exp_type == "trypanblue":
                print(exp_type, task_id)
                image_save_path = messge["image_save_path"]
                input_image_path = messge["input_image_path"]
                data_save_path = messge["data_save_path"]
                try:
                    fcs_save_path = messge["fcs_save_path"]
                except:
                    fcs_save_path = ""
                conf_thrd = messge["conf_thrd"]
                iou_thrd = messge["iou_thrd"]
                detmodel_iference(input_im_path=input_image_path, image_save_path=image_save_path,
                                  data_save_path=data_save_path, fcs_save_path=fcs_save_path,
                                  cut_size=640, conf_thres=conf_thrd,
                                  iou_thres=iou_thrd, typeblue_model=TB_predictor, extype=exp_type)
            elif exp_type == "suspension_BF":
                print(exp_type, task_id)
                image_save_path = messge["image_save_path"]
                input_image_path = messge["input_image_path"]
                data_save_path = messge["data_save_path"]
                try:
                    fcs_save_path = messge["fcs_save_path"]
                except:
                    fcs_save_path = ""
                conf_thrd = messge["conf_thrd"]
                iou_thrd = messge["iou_thrd"]
                detmodel_iference(input_im_path=input_image_path, image_save_path=image_save_path,
                                  data_save_path=data_save_path, fcs_save_path=fcs_save_path, conf_thres=conf_thrd,
                                  cut_size=640, iou_thres=iou_thrd, BFdetection_model=BF2_predictor, extype=exp_type)
            elif exp_type == "suspension_FL":
                print(exp_type, task_id)
                FL1_image_path = messge["FL1_image_path"]
                FL2_image_path = messge["FL2_image_path"]
                FL1_save_path = messge["FL1_save_path"]
                FL2_save_path = messge["FL2_save_path"]
                mrege_save_path = messge["merge_save_path"]
                data_save_path = messge["data_save_path"]
                try:
                    fcs_save_path = messge["fcs_save_path"]
                except:
                    fcs_save_path = ""
                try:
                    FL1_conf_thrd = messge["FL1_conf_thrd"]
                    FL1_iou_thrd = messge["FL1_iou_thrd"]
                    FL2_conf_thrd = messge["FL2_conf_thrd"]
                    FL2_iou_thrd = messge["FL2_iou_thrd"]
                except Exception as e:
                    print(e)
                    FL1_conf_thrd = messge["conf_thrd"]
                    FL1_iou_thrd = messge["iou_thrd"]
                    FL2_conf_thrd = messge["conf_thrd"]
                    FL2_iou_thrd = messge["iou_thrd"]
                    print(FL1_conf_thrd, FL1_iou_thrd, FL2_conf_thrd, FL2_iou_thrd)
                detmodel_iference(input_im_path=[FL1_image_path, FL2_image_path],
                                  image_save_path=[FL1_save_path, FL2_save_path, mrege_save_path],
                                  data_save_path=data_save_path, fcs_save_path=fcs_save_path,
                                  cut_size=640, FL1_conf_thrd=FL1_conf_thrd, FL1_iou_thrd=FL1_iou_thrd,
                                  FL2_conf_thrd=FL2_conf_thrd, FL2_iou_thrd=FL2_iou_thrd,
                                  FLdeteciton_model=FL2_predictor, extype=exp_type)
            elif exp_type == "AOPI_merge":
                print(exp_type, task_id)
                bf_image_path = messge["BF_input_path"]
                fl1_image_path = messge["FL1_input_path"]
                fl2_image_path = messge["FL2_input_path"]
                bf_save_image_path = messge["BF_save_path"]
                fl1_save_image_path = messge["FL1_save_path"]
                fl2_save_image_path = messge["FL2_save_path"]
                merge_save_path = messge["merge_save_path"]
                data_save_path = messge["data_save_path"]
                try:
                    fcs_save_path = messge["fcs_save_path"]
                except:
                    fcs_save_path = ""
                try:
                    BF_conf_thrd = messge["BF_conf_thrd"]
                    BF_iou_thrd = messge["BF_iou_thrd"]
                    FL1_conf_thrd = messge["FL1_conf_thrd"]
                    FL1_iou_thrd = messge["FL1_iou_thrd"]
                    FL2_conf_thrd = messge["FL2_conf_thrd"]
                    FL2_iou_thrd = messge["FL2_iou_thrd"]
                except Exception as e:
                    print(e)
                    BF_conf_thrd = messge["conf_thrd"]
                    BF_iou_thrd = messge["iou_thrd"]
                    FL1_conf_thrd = messge["conf_thrd"]
                    FL1_iou_thrd = messge["iou_thrd"]
                    FL2_conf_thrd = messge["conf_thrd"]
                    FL2_iou_thrd = messge["iou_thrd"]
                    print(BF_conf_thrd, BF_iou_thrd, FL1_conf_thrd, FL1_iou_thrd, FL2_conf_thrd, FL2_iou_thrd)
                detmodel_iference(input_im_path=[bf_image_path, fl1_image_path, fl2_image_path],
                                  image_save_path=[bf_save_image_path, fl1_save_image_path, fl2_save_image_path,
                                                   merge_save_path], data_save_path=data_save_path,
                                  fcs_save_path=fcs_save_path, cut_size=640, BF_conf_thrd=BF_conf_thrd,
                                  BF_iou_thrd=BF_iou_thrd, FL1_conf_thrd=FL1_conf_thrd, FL1_iou_thrd=FL1_iou_thrd,
                                  FL2_conf_thrd=FL2_conf_thrd, FL2_iou_thrd=FL2_iou_thrd,
                                  BFdetection_model=BF2_predictor, FLdeteciton_model=FL2_predictor, extype=exp_type)
            else:
                write_log("%s is not exsit!" % exp_type)
                sucess_msg = "%s is not exsit!" % exp_type
                status = "fail"
            data = {"ProcessID": os.getppid(), "tid": task_id, "createTime": createTime, "status": status,
                    "msg": sucess_msg, "updateTime": getnowtime()}
            write_log("[processing completed]:" + str(messge))
            writequeue(data)
        except Exception as e:
            print(e)
            # traceback.print_exc(file=open("Error_LOG.txt", 'a+'))
            errr_content = StringIO()
            traceback.print_exc(file=errr_content)
            write_log("[processing fial]: " + "tid " + task_id + " " + str(e) + " " + errr_content.getvalue())
            data = {"ProcessID": os.getppid(), "tid": task_id, "createTime": createTime, "status": "fail",
                    "msg": str(e) + " " + errr_content.getvalue(), "updateTime": getnowtime()}
            writequeue(data)
        ch.basic_ack(delivery_tag=method.delivery_tag)


    while True:
        try:
            print("start connect RMQ")
            write_log("start connect RMQ")
            credentials = pika.PlainCredentials('guest', 'guest')
            connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost',
                                                                           port=5672,
                                                                           virtual_host='/',
                                                                           heartbeat=0,
                                                                           credentials=credentials))
            channel = connection.channel()
            channel.queue_declare(queue='processing-list', durable=True)
            # 告诉rabbitmq，用callback来接收消息
            # 默认情况下是要对消息进行确认的，以防止消息丢失。
            # 此处将auto_ack明确指明为True，不对消息进行确认。
            channel.basic_consume('processing-list', on_message_callback=processing_image)
            # 开始接收信息，并进入阻塞状态，队列里有信息才会调用callback进行处理
            channel.start_consuming()
        # Don't recover if connection was closed by broker
        except Exception as e:
            print(e)
            # traceback.print_exc(file=open("Error_LOG.txt", 'a+'))
            errr_content = StringIO()
            traceback.print_exc(file=errr_content)
            write_log("[processing fial]: " + str(e) + " " + errr_content.getvalue())

# qpt.exe -f ./RMQ_softwareV20211124 -p ./RMQ_softwareV20211124/rmq_main.py -s ./RMQ_softwareV20211124_qpt -h True -r ./RMQ_softwareV20211124/requirements_with_opt.txt
