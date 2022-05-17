import os.path as osp
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import pycuda.autoinit
import yaml
import cv2
class TrtPredictor(object):
    def __init__(self, model_file_path, model_yaml):
        self.logger = trt.Logger(trt.Logger.ERROR)
        if not osp.exists(model_file_path):
            print("model xml file is not exists in {}".format(model_file_path))
        with open(model_file_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
            }
            print("binding:", binding)
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
        self.context = self.engine.create_execution_context()
        cuda.memcpy_htod(self.inputs[0]['allocation'],
                         np.ascontiguousarray(np.zeros(self.inputs[0]['shape'], dtype=self.inputs[0]['dtype'])))
        print(model_file_path, self.inputs[0]['shape'], self.inputs[0]['dtype'])
        print(model_file_path, self.outputs[0]['shape'], self.outputs[0]['dtype'])
        self.context.execute_v2(self.allocations)
        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0
        if not osp.exists(model_yaml):
            print("model yaml file is not exists in {}".format(model_yaml))
        with open(model_yaml) as f:
            self.info = yaml.load(f.read(), Loader=yaml.Loader)
        self.model_type = self.info['_Attributes']['model_type']
        self.model_name = self.info['Model']
        self.num_classes = self.info['_Attributes']['num_classes']
        self.labels = self.info['_Attributes']['labels']
        transforms_mode = self.info.get('TransformsMode', 'RGB')
        if transforms_mode == 'RGB':
            to_rgb = True
        else:
            to_rgb = False
        self.transforms = self.build_transforms(self.info['Transforms'],
                                                to_rgb)


    def build_transforms(self, transforms_info, to_rgb=True):
        if self.model_type == "classifier":
            import transforms.cls_transforms as transforms
        elif self.model_type == "detector":
            import transforms.det_transforms as transforms
        elif self.model_type == "segmenter":
            import transforms.seg_transforms as transforms
        op_list = list()
        for op_info in transforms_info:
            op_name = list(op_info.keys())[0]
            op_attr = op_info[op_name]
            if not hasattr(transforms, op_name):
                raise Exception(
                    "There's no operator named '{}' in transforms of {}".
                    format(op_name, self.model_type))
            op_list.append(getattr(transforms, op_name)(**op_attr))
        eval_transforms = transforms.Compose(op_list)
        if hasattr(eval_transforms, 'to_rgb'):
            eval_transforms.to_rgb = to_rgb
        self.arrange_transforms(eval_transforms)
        return eval_transforms

    def arrange_transforms(self, eval_transforms):
        if self.model_type == 'classifier':
            import transforms.cls_transforms as transforms
            arrange_transform = transforms.ArrangeClassifier
        elif self.model_type == 'segmenter':
            import transforms.seg_transforms as transforms
            arrange_transform = transforms.ArrangeSegmenter
        elif self.model_type == 'detector':
            import transforms.det_transforms as transforms
            arrange_name = 'Arrange{}'.format(self.model_name)
            arrange_transform = getattr(transforms, arrange_name)
        else:
            raise Exception("Unrecognized model type: {}".format(
                self.model_type))
        if type(eval_transforms.transforms[-1]).__name__.startswith('Arrange'):
            eval_transforms.transforms[-1] = arrange_transform(mode='test')
        else:
            eval_transforms.transforms.append(arrange_transform(mode='test'))

    def raw_predict(self, preprocessed_input):
        self.count_num += 1
        feed_dict = {}
        if self.model_name == "YOLOv3":
            inputs = self.net.input_info
            for name in inputs:
                if (len(inputs[name].shape) == 2):
                    feed_dict[name] = preprocessed_input['im_size']
                elif (len(inputs[name].shape) == 4):
                    feed_dict[name] = preprocessed_input['image']
                else:
                    pass
        else:
            input_blob = next(iter(self.net.input_info))
            feed_dict[input_blob] = preprocessed_input['image']
        #Start sync inference
        print("Starting inference in synchronous mode")
        res = self.predictor.infer(inputs=feed_dict)

        #Processing output blob
        print("Processing output blob")
        return res

    def preprocess(self, image):
        res = dict()
        if self.model_type == "classifier":
            im = self.transforms(image)
            im = np.expand_dims(im, axis=0).copy()
            res['image'] = im
        elif self.model_type == "detector":
            if self.model_name == "YOLOv3":
                im, im_shape = self.transforms(image)
                im = np.expand_dims(im, axis=0).copy()
                im_shape = np.expand_dims(im_shape, axis=0).copy()
                res['image'] = im
                res['im_size'] = im_shape
        elif self.model_type == "segmenter":
            im, im_info = self.transforms(image)
            im = np.expand_dims(im, axis=0).copy()
            res['image'] = im
            res['im_info'] = im_info
        return res

    def classifier_postprocess(self, preds, topk=1):
        """ 对分类模型的预测结果做后处理
        """
        true_topk = min(self.num_classes, topk)
        output_name = next(iter(self.net.outputs))
        pred_label = np.argsort(-preds[output_name][0])[:true_topk]
        result = [{
            'category_id': l,
            'category': self.labels[l],
            'score': preds[output_name][0][l],
        } for l in pred_label]
        return result
    def segmenter_postprocess_trt(self,preprocessed_inputs):
        """ 对语义分割结果做后处理
        """
        cuda.memcpy_htod(self.inputs[0]['allocation'], np.ascontiguousarray(preprocessed_inputs['image']))
        self.context.execute_v2(self.allocations)
        output = np.zeros(self.outputs[0]['shape'], dtype=self.outputs[0]['dtype'])
        cuda.memcpy_dtoh(output, self.outputs[0]['allocation'])
        label_map = np.squeeze(output).astype('uint8')
        im_info = preprocessed_inputs['im_info']
        for info in im_info[::-1]:
            if info[0] == 'resize':
                w, h = info[1][1], info[1][0]
                label_map = cv2.resize(label_map, (w, h), cv2.INTER_NEAREST)
                #score_map = cv2.resize(score_map, (w, h), cv2.INTER_LINEAR)
            elif info[0] == 'padding':
                w, h = info[1][1], info[1][0]
                label_map = label_map[0:h, 0:w]
                #score_map = score_map[0:h, 0:w, :]
        return {'label_map': label_map}

    def segmenter_postprocess(self, preds, preprocessed_inputs):
        """ 对语义分割结果做后处理
        """
        it = iter(self.net.outputs)
        try:
            next(it)
            label_name = next(it)
            label_map = np.squeeze(preds[label_name]).astype('uint8')
            # score_name = next(it)
            # score_map = np.squeeze(preds[score_name])
            # score_map = np.transpose(score_map, (1, 2, 0))
        except:
            it= iter(list(preds.keys())[::-1])
            label_name = next(it)
            #print("label_name:", label_name)
            label_map = np.squeeze(preds[label_name]).astype('uint8')
            #score_name = next(it)
            #print("score_name:", score_name)
            #score_map = np.squeeze(preds[score_name])
            #score_map = np.transpose(score_map, (1, 2, 0))

        im_info = preprocessed_inputs['im_info']
        for info in im_info[::-1]:
            if info[0] == 'resize':
                w, h = info[1][1], info[1][0]
                label_map = cv2.resize(label_map, (w, h), cv2.INTER_NEAREST)
                #score_map = cv2.resize(score_map, (w, h), cv2.INTER_LINEAR)
            elif info[0] == 'padding':
                w, h = info[1][1], info[1][0]
                label_map = label_map[0:h, 0:w]
                #score_map = score_map[0:h, 0:w, :]

        return {'label_map': label_map}

    def detector_postprocess(self, preds, preprocessed_inputs):
        """对图像检测结果做后处理
        """
        outputs = self.net.outputs
        for name in outputs:
            if (len(outputs[name].shape) == 2):
                output = preds[name]
        result = []
        for out in output:
            if (out[0] >= 0):
                result.append(out.tolist())
            else:
                pass
        return result

    def predict(self, image, topk=1, threshold=0.5):
        preprocessed_input = self.preprocess(image)
        #print("model_pred",model_pred)
        if self.model_type == "classifier":
            model_pred = self.raw_predict(preprocessed_input)
            results = self.classifier_postprocess(model_pred, topk)
        elif self.model_type == "detector":
            model_pred = self.raw_predict(preprocessed_input)
            results = self.detector_postprocess(model_pred, preprocessed_input)
        elif self.model_type == "segmenter":
            results=self.segmenter_postprocess_trt(preprocessed_input)
        return results
    def overlap_tile_predict(self,
                             img_file,
                             tile_size=[512, 512],
                             pad_size=[64, 64]):
        """有重叠的大图切小图预测。
        Args:
            img_file(str|np.ndarray): 预测图像路径，或者是解码后的排列格式为（H, W, C）且类型为float32且为BGR格式的数组。
            tile_size(list|tuple): 滑动窗口的大小，该区域内用于拼接预测结果，格式为（W，H）。默认值为[512, 512]。
            pad_size(list|tuple): 滑动窗口向四周扩展的大小，扩展区域内不用于拼接预测结果，格式为（W，H）。默认值为[64，64]。
            batch_size(int)：对窗口进行批量预测时的批量大小。默认值为32
            transforms(paddlex.cv.transforms): 数据预处理操作。


        Returns:
            dict: 包含关键字'label_map'和'score_map', 'label_map'存储预测结果灰度图，
                像素值表示对应的类别，'score_map'存储各类别的概率，shape=(h, w, num_classes)
        """

        if isinstance(img_file, str):
            image= cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), cv2.IMREAD_COLOR)
        elif isinstance(img_file, np.ndarray):
            image = img_file.copy()
        else:
            raise Exception("im_file must be str/ndarray")

        height, width, channel = image.shape
        image_tile_list = list()

        # Padding along the left and right sides
        if pad_size[0] > 0:
            left_pad = cv2.flip(image[0:height, 0:pad_size[0], :], 1)
            right_pad = cv2.flip(image[0:height, -pad_size[0]:width, :], 1)
            padding_image = cv2.hconcat([left_pad, image])
            padding_image = cv2.hconcat([padding_image, right_pad])
        else:
            import copy
            padding_image = copy.deepcopy(image)

        # Padding along the upper and lower sides
        padding_height, padding_width, _ = padding_image.shape
        if pad_size[1] > 0:
            upper_pad = cv2.flip(
                padding_image[0:pad_size[1], 0:padding_width, :], 0)
            lower_pad = cv2.flip(
                padding_image[-pad_size[1]:padding_height, 0:padding_width, :],
                0)
            padding_image = cv2.vconcat([upper_pad, padding_image])
            padding_image = cv2.vconcat([padding_image, lower_pad])

        # crop the padding image into tile pieces
        padding_height, padding_width, _ = padding_image.shape

        for h_id in range(0, height // tile_size[1] + 1):
            for w_id in range(0, width // tile_size[0] + 1):
                left = w_id * tile_size[0]
                upper = h_id * tile_size[1]
                right = min(left + tile_size[0] + pad_size[0] * 2,
                            padding_width)
                lower = min(upper + tile_size[1] + pad_size[1] * 2,
                            padding_height)
                image_tile = padding_image[upper:lower, left:right, :]
                image_tile_list.append(image_tile)

        # predict
        label_map = np.zeros((height, width), dtype=np.uint8)
        # score_map = np.zeros(
        #     (height, width, self.num_classes), dtype=np.float32)
        num_tiles = len(image_tile_list)
        print("num_tiles:", num_tiles)
        for i in range(0, num_tiles):
            res = self.predict(image_tile_list[i])
            h_id = i // (width // tile_size[0] + 1)
            w_id = i % (width // tile_size[0] + 1)
            left = w_id * tile_size[0]
            upper = h_id * tile_size[1]
            right = min((w_id + 1) * tile_size[0], width)
            lower = min((h_id + 1) * tile_size[1], height)
            tile_label_map = res["label_map"]
            #tile_score_map = res["score_map"]
            tile_upper = pad_size[1]
            tile_lower = tile_label_map.shape[0] - pad_size[1]
            tile_left = pad_size[0]
            tile_right = tile_label_map.shape[1] - pad_size[0]
            label_map[upper:lower, left:right] = \
                tile_label_map[tile_upper:tile_lower, tile_left:tile_right]
            # score_map[upper:lower, left:right, :] = \
            #     tile_score_map[tile_upper:tile_lower, tile_left:tile_right, :]
        result = {"label_map": label_map}
        return result
class YoLov5TRT(object):
    def __init__(self, model_file_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        if not osp.exists(model_file_path):
            print("model xml file is not exists in {}".format(model_file_path))
        with open(model_file_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
            }
            print("binding:", binding)
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
        self.context = self.engine.create_execution_context()
        cuda.memcpy_htod(self.inputs[0]['allocation'],
                         np.ascontiguousarray(np.zeros(self.inputs[0]['shape'], dtype=self.inputs[0]['dtype'])))
        print(model_file_path, self.inputs[0]['shape'], self.inputs[0]['dtype'])
        print(model_file_path, self.outputs[0]['shape'], self.outputs[0]['dtype'])
        self.context.execute_v2(self.allocations)
        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def infer(self, input_image):
        cuda.memcpy_htod(self.inputs[0]['allocation'], np.ascontiguousarray(input_image))
        self.context.execute_v2(self.allocations)
        output = np.zeros(self.outputs[0]['shape'], dtype=self.outputs[0]['dtype'])
        cuda.memcpy_dtoh(output, self.outputs[0]['allocation'])
        return output
