# 生产者代码
import pika
import json
import uuid
credentials = pika.PlainCredentials('guest', 'guest')  # mq用户名和密码，没有则需要自己创建
# 虚拟队列需要指定参数 virtual_host，如果是默认的可以不填。
connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost',
                                                               port=5672,
                                                               virtual_host='/',
                                                               heartbeat=0,
                                                               credentials=credentials))

# 建立rabbit协议的通道
channel = connection.channel()
# 声明消息队列，消息将在这个队列传递，如不存在，则创建。durable指定队列是否持久化
channel.queue_declare(queue='processing-list', durable=True)

# message不能直接发送给queue，需经exchange到达queue，此处使用以空字符串标识的默认的exchange
# 向队列插入数值 routing_key是队列名
# for i in range(1):
#     message = json.dumps({"tid":str(uuid.uuid3(uuid.NAMESPACE_DNS,str(i))),"input_path": "G:\detect\F_1.jpg", "Prob": 0.7,
#               "save_path": "G:\detect\F_1_result.jpg", "data_save_path": "G:\detect\F_1_result.csv"})
#     channel.basic_publish(exchange='',
#                           routing_key='waiting-list',
#                           body=message)
# for i in range(2):
#     message = json.dumps({"tid":str(uuid.uuid3(uuid.NAMESPACE_DNS,str(i))),"input_path": "G:\detect\F_1.jpg", "Prob": 0.5,
#               "save_path": "G:\detect\F_1_result.jpg", "data_save_path": "G:\detect\F_1_result.csv"})
#     channel.basic_publish(exchange='',
#                           routing_key='waiting-list',
#                           body=message)

for i in range(1):

    #MAB
    message = json.dumps({"tid":str(uuid.uuid3(uuid.NAMESPACE_DNS,str(i))),"exp_type":"single_cloneCV","input_image_path":r"H:\单图图库\单克隆\1\C9_01_cut.jpg",
                         "image_save_path":r"H:\test\C9_01_count_BF_AI.jpg","data_save_path":r"H:\test\C9_01_count_BF_AI.csv","cut_thrd":100})
    channel.basic_publish(exchange='',
                          routing_key='processing-list',
                          body=message)
    # #
    # message = json.dumps({"tid":str(uuid.uuid3(uuid.NAMESPACE_DNS,str(i))),"exp_type":"single_clone","input_image_path":r"H:\单图图库\单克隆\1\C11_01_cut.jpg",
    #                      "image_save_path":r"H:\test\C11_01_count_BF_AI.jpg","data_save_path":r"H:\test\C11_01_count_BF_AI.csv","fcs_save_path":r"H:\test\A3_01_01_01_Trans.csv","cut_thrd":100})
    # channel.basic_publish(exchange='',
    #                       routing_key='processing-list',
    #                       body=message)
    # CHO
    # message = json.dumps({"tid":str(uuid.uuid3(uuid.NAMESPACE_DNS,str(i))),"exp_type":"confluence","input_image_path":r"H:\单图图库\汇合度\1\H11_01_cut.jpg",
    #                      "image_save_path":r"H:\test\H11_01_count_BF_AI.jpg","data_save_path":r"H:\test\H11_01_count_BF_AI.csv","cut_patch":3,"fcs_save_path":r"H:\test\A3_01_01_01_Trans.csv","fast_mode":True})
    # channel.basic_publish(exchange='',
    #                       routing_key='processing-list',
    #                       body=message)
    # message = json.dumps({"tid":str(uuid.uuid3(uuid.NAMESPACE_DNS,str(i))),"exp_type":"confluence","input_image_path":r"H:\单图图库\汇合度\1\H12_01_cut.jpg",
    #                      "image_save_path":r"H:\test\H12_01_count_BF_AI.jpg","data_save_path":r"H:\test\H12_01_count_BF_AI.csv","cut_patch":3,"fcs_save_path":r"H:\test\A3_01_01_01_Trans.csv","fast_mode":False})
    # channel.basic_publish(exchange='',
    #                       routing_key='processing-list',
    #                       body=message)
    # # trypanblue
    # message = json.dumps({"tid":str(uuid.uuid3(uuid.NAMESPACE_DNS,str(i))),"input_image_path": r"H:\单图图库\台盼蓝\A1_01_01_01.jpg", "exp_type": "trypanblue",
    #           "image_save_path":r"H:\test\A1_01_01_01_AI.jpg","data_save_path":r"H:\test\A1_01_01_01_AI.csv","conf_thrd":0.4,"iou_thrd":0.1,"fcs_save_path":r"H:\test\A1_01_01_01_AI.csv"})
    # channel.basic_publish(exchange='',
    #                       routing_key='processing-list',
    #                       body=message)
    # # #suspension_BF
    # message = json.dumps({"tid":str(uuid.uuid3(uuid.NAMESPACE_DNS,str(i))),"input_image_path": r"D:\AI质检\1123-beadsA-焦距模糊严重\1\A8_01_01_01.jpg", "exp_type": "suspension_BF",
    #           "image_save_path":r"H:\test\A1_01_01_01_bf.jpg","data_save_path":r"H:\test\A1_01_01_01_bf.csv","conf_thrd":0.1,"iou_thrd":0.1})
    # channel.basic_publish(exchange='',
    #                       routing_key='processing-list',
    #                       body=message)
    # # # suspension_FL
    # message = json.dumps({"tid":str(uuid.uuid3(uuid.NAMESPACE_DNS,str(i))),"FL1_image_path": r"H:\单图图库\AOPI\A1_01_01_01FL1.jpg", "FL2_image_path": r"H:\单图图库\AOPI\A1_01_01_01FL2.jpg", "exp_type": "suspension_FL",
    #           "FL1_save_path":r"H:\test\A1_01_01_01_fl1.jpg","FL2_save_path":r"H:\test\A1_01_01_01_fl2.jpg","merge_save_path":r"H:\test\A1_01_01_01_merge.jpg","data_save_path":r"H:\test\A1_01_01_01_megre.csv","conf_thrd":0.1,"iou_thrd":0.1})
    # channel.basic_publish(exchange='',
    #                       routing_key='processing-list',
    #                       body=message)
    # message = json.dumps({"tid":str(uuid.uuid3(uuid.NAMESPACE_DNS,str(i))),"FL1_image_path": "", "FL2_image_path": r"D:\AI质检\1123-beadsA-焦距模糊严重\4\A8_01_01_01.jpg", "exp_type": "suspension_FL",
    #           "FL1_save_path":r"H:\test\A1_01_01_01_fl1.jpg","FL2_save_path":r"H:\test\A1_01_01_01_fl2.jpg","merge_save_path":r"H:\test\A1_01_01_01_merge.jpg","data_save_path":r"H:\test\A1_01_01_01_fl2.csv","fcs_save_path":r"H:\test\A3_01_01_01_Trans.csv","conf_thrd":0.1,"iou_thrd":0.1})
    # channel.basic_publish(exchange='',
    #                       routing_key='processing-list',
    #                       body=message)
    # message = json.dumps({"tid":str(uuid.uuid3(uuid.NAMESPACE_DNS,str(i))),"FL1_image_path": r"D:\AI质检\1123-beadsA-焦距模糊严重\2\A8_01_01_01.jpg", "FL2_image_path": "", "exp_type": "suspension_FL",
    #           "FL1_save_path":r"H:\test\A1_01_01_01_fl1.jpg","FL2_save_path":r"H:\test\A1_01_01_01_fl2.jpg","merge_save_path":r"H:\test\A1_01_01_01_merge.jpg","data_save_path":r"H:\test\A1_01_01_01_fl2.csv","fcs_save_path":r"H:\test\A3_01_01_01_Trans.csv","conf_thrd":0.1,"iou_thrd":0.1})
    # channel.basic_publish(exchange='',
    #                       routing_key='processing-list',
    #                       body=message)
    # #AOPI_merge
    message = json.dumps({"tid":str(uuid.uuid3(uuid.NAMESPACE_DNS,str(i))),"BF_input_path": r"H:\单图图库\AOPI\A1_01_01_01BF.jpg",
                          "FL1_input_path": r"H:\单图图库\AOPI\A1_01_01_01FL1.jpg","FL2_input_path": r"H:\单图图库\AOPI\A1_01_01_01FL2.jpg","exp_type": "AOPI_merge",
              "BF_save_path":r"H:\test\A1_01_01_01BF.jpg","FL1_save_path":r"H:\test\A1_01_01_01FL1.jpg","FL2_save_path":r"H:\test\A1_01_01_01FL2.jpg","merge_save_path":r"H:\test\A1_01_01_01_merge.jpg","data_save_path":r"H:\test\A1_01_01_01_merge.csv","fcs_save_path":r"H:\test\A3_01_01_01_fcs.csv","conf_thrd":0.1,"iou_thrd":0.01})
    channel.basic_publish(exchange='',
                          routing_key='processing-list',
                          body=message)
    # message = json.dumps({"tid":str(uuid.uuid3(uuid.NAMESPACE_DNS,str(i))),"BF_input_path": r"H:\单图图库\AOPI\A1_01_01_01BF.jpg",
    #                       "FL1_input_path": "","FL2_input_path": r"H:\单图图库\AOPI\A1_01_01_01FL2.jpg","exp_type": "AOPI_merge",
    #           "BF_save_path":r"H:\test\A1_01_01_01BF.jpg","FL1_save_path":r"H:\test\A1_01_01_01FL1.jpg","FL2_save_path":r"H:\test\A1_01_01_01FL2.jpg","merge_save_path":r"H:\test\A1_01_01_01_merge.jpg","data_save_path":r"H:\test\A1_01_01_01_merge.csv","fcs_save_path":r"H:\test\A3_01_01_01_Trans.csv","conf_thrd":0.1,"iou_thrd":0.01})
    # channel.basic_publish(exchange='',
    #                       routing_key='processing-list',
    #                       body=message)
    # message = json.dumps({"tid":str(uuid.uuid3(uuid.NAMESPACE_DNS,str(i))),"BF_input_path": r"H:\单图图库\AOPI\A1_01_01_01BF.jpg",
    #                       "FL1_input_path": r"H:\单图图库\AOPI\A1_01_01_01FL1.jpg","FL2_input_path": "","exp_type": "AOPI_merge",
    #           "BF_save_path":r"H:\test\A1_01_01_01BF.jpg","FL1_save_path":r"H:\test\A1_01_01_01FL1.jpg","FL2_save_path":r"H:\test\A1_01_01_01FL2.jpg","merge_save_path":r"H:\test\A1_01_01_01_merge.jpg","data_save_path":r"H:\test\A1_01_01_01_merge.csv","fcs_save_path":r"H:\test\A3_01_01_01_Trans.csv","conf_thrd":0.1,"iou_thrd":0.1})
    # channel.basic_publish(exchange='',
    #                       routing_key='processing-list',
    #                       body=message)
    # message = json.dumps({"tid":str(uuid.uuid3(uuid.NAMESPACE_DNS,str(i))),"BF_input_path": r"H:\test\图1\1\A1_01_01_01.jpg",
    #                       "FL1_input_path": r"H:\test\图1\2\A1_01_01_01.jpg","FL2_input_path": r"H:\test\图1\4\A1_01_01_01.jpg","exp_type": "AOPI_merge_2",
    #           "BF_save_path":r"H:\test\A1_01_01_01BF_1.jpg","FL1_save_path":r"H:\test\A1_01_01_01FL1_1.jpg","FL2_save_path":r"H:\test\A1_01_01_01FL2_1.jpg","merge_save_path":r"H:\test\A1_01_01_01_merge_1.jpg","data_save_path":r"H:\test\A1_01_01_01_merge_1.csv","conf_thrd":0.1,"iou_thrd":0.1})
    # channel.basic_publish(exchange='',
    #                       routing_key='processing-list',
    #                       body=message)
    # message = json.dumps({"tid":str(uuid.uuid3(uuid.NAMESPACE_DNS,str(i))),"BF_input_path": r"H:\test\图2\1\A1_01_01_01.jpg",
    #                       "FL1_input_path": r"H:\test\图2\2\A1_01_01_01.jpg","FL2_input_path": r"H:\test\图2\4\A1_01_01_01.jpg","exp_type": "AOPI_merge_2",
    #           "BF_save_path":r"H:\test\A1_01_01_01BF_2.jpg","FL1_save_path":r"H:\test\A1_01_01_01FL1_2.jpg","FL2_save_path":r"H:\test\A1_01_01_01FL2_2.jpg","merge_save_path":r"H:\test\A1_01_01_01_merge_2.jpg","data_save_path":r"H:\test\A1_01_01_01_merge_2.csv","conf_thrd":0.1,"iou_thrd":0.1})
    # channel.basic_publish(exchange='',
    #                       routing_key='processing-list',
    #                       body=message)
    # message = json.dumps({"tid":str(uuid.uuid3(uuid.NAMESPACE_DNS,str(i))),"BF_input_path": r"H:\test\图3\1\A1_01_01_01.jpg",
    #                       "FL1_input_path": r"H:\test\图3\2\A1_01_01_01.jpg","FL2_input_path": r"H:\test\图3\4\A1_01_01_01.jpg","exp_type": "AOPI_merge_2",
    #           "BF_save_path":r"H:\test\A1_01_01_01BF_3.jpg","FL1_save_path":r"H:\test\A1_01_01_01FL1_3.jpg","FL2_save_path":r"H:\test\A1_01_01_01FL2_3.jpg","merge_save_path":r"H:\test\A1_01_01_01_merge_3.jpg","data_save_path":r"H:\test\A1_01_01_01_merge_3.csv","conf_thrd":0.1,"iou_thrd":0.1})
    # channel.basic_publish(exchange='',
    #                       routing_key='processing-list',
    #                       body=message)
    #transfection
    # message = json.dumps({"tid":str(uuid.uuid3(uuid.NAMESPACE_DNS,str(i))),"BF_input_path": r"H:\单图图库\细胞转染\100\1\A3_01_01_01.jpg",
    #                       "FL1_input_path": r"H:\单图图库\细胞转染\100\2\A3_01_01_01.jpg","FL2_input_path": r"H:\单图图库\细胞转染\100\4\A3_01_01_01.jpg","exp_type": "transfection",
    #           "BF_save_path":r"H:\test\A3_01_01_01_BF.jpg","FL1_save_path":r"H:\test\A3_01_01_01_FL1.jpg","FL2_save_path":r"H:\test\A3_01_01_01_FL2.jpg","merge_save_path":r"H:\test\A3_01_01_01_merge.jpg","data_save_path":r"H:\test\A3_01_01_01_Trans.csv","fcs_save_path":r"H:\test\A3_01_01_01_Trans.csv","cut_patch":4})
    # channel.basic_publish(exchange='',routing_key='processing-list',body=message)
    #
    # message = json.dumps({"tid":str(uuid.uuid3(uuid.NAMESPACE_DNS,str(i))),"BF_input_path": r"H:\单图图库\细胞转染\100\1\A3_01_01_01.jpg",
    #                       "FL1_input_path": r"H:\单图图库\细胞转染\100\2\A3_01_01_01.jpg","FL2_input_path": r"H:\单图图库\细胞转染\100\4\A3_01_01_01.jpg","exp_type": "transfection",
    #           "BF_save_path":r"H:\test\A3_01_01_01_BF.jpg","FL1_save_path":r"H:\test\A3_01_01_01_FL1.jpg","FL2_save_path":r"H:\test\A3_01_01_01_FL2.jpg","merge_save_path":r"H:\test\A3_01_01_01_merge.jpg","data_save_path":r"H:\test\A3_01_01_01_Trans.csv","fcs_save_path":r"H:\test\A3_01_01_01_Trans.csv","cut_patch":4})
    # channel.basic_publish(exchange='',routing_key='processing-list',body=message)