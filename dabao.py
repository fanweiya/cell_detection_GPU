from GPU_Predictor import TrtPredictor,YoLov5TRT

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
print(kuangpredictor, grouppredictor, chopredictor, zuanranpredictor)