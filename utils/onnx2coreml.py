import os
import onnx
from onnx import onnx_pb
from onnx_coreml import convert
import glob


# https://github.com/onnx/onnx-coreml
# http://machinethink.net/blog/mobilenet-ssdlite-coreml/
# https://github.com/hollance/YOLO-CoreML-MPSNNGraph

def main():
    os.system('rm -rf saved_models && mkdir saved_models')
    files = glob.glob('saved_models/*.onnx') + glob.glob('../yolov3/weights/*.onnx')

    for f in files:
        # 1. ONNX to CoreML
        name = 'saved_models/' + f.split('/')[-1].replace('.onnx', '')

        # # Load the ONNX model
        model = onnx.load(f)

        # Check that the IR is well formed
        print(onnx.checker.check_model(model))

        # Print a human readable representation of the graph
        print(onnx.helper.printable_graph(model.graph))

        model_file = open(f, 'rb')
        model_proto = onnx_pb.ModelProto()
        model_proto.ParseFromString(model_file.read())
        yolov3_model = convert(model_proto, image_input_names=['0'], preprocessing_args={'image_scale': 1. / 255})

        # 2. Reduce model to FP16, change outputs to DOUBLE and save
        import coremltools

        spec = yolov3_model.get_spec()
        for i in range(2):
            spec.description.output[i].type.multiArrayType.dataType = \
                coremltools.proto.FeatureTypes_pb2.ArrayFeatureType.ArrayDataType.Value('DOUBLE')

        spec = coremltools.utils.convert_neural_network_spec_weights_to_fp16(spec)
        yolov3_model = coremltools.models.MLModel(spec)

        num_classes = 80
        num_anchors = 507
        spec.description.output[0].type.multiArrayType.shape.append(num_anchors)
        spec.description.output[0].type.multiArrayType.shape.append(num_classes)
        # spec.description.output[0].type.multiArrayType.shape.append(1)

        spec.description.output[1].type.multiArrayType.shape.append(num_anchors)
        spec.description.output[1].type.multiArrayType.shape.append(4)
        # spec.description.output[1].type.multiArrayType.shape.append(1)

        # rename
        # input_mlmodel = input_tensor.replace(":", "__").replace("/", "__")
        # class_output_mlmodel = class_output_tensor.replace(":", "__").replace("/", "__")
        # bbox_output_mlmodel = bbox_output_tensor.replace(":", "__").replace("/", "__")
        #
        # for i in range(len(spec.neuralNetwork.layers)):
        #     if spec.neuralNetwork.layers[i].input[0] == input_mlmodel:
        #         spec.neuralNetwork.layers[i].input[0] = 'image'
        #     if spec.neuralNetwork.layers[i].output[0] == class_output_mlmodel:
        #         spec.neuralNetwork.layers[i].output[0] = 'scores'
        #     if spec.neuralNetwork.layers[i].output[0] == bbox_output_mlmodel:
        #         spec.neuralNetwork.layers[i].output[0] = 'boxes'

        spec.neuralNetwork.preprocessing[0].featureName = '0'

        yolov3_model.save(name + '.mlmodel')
        # yolov3_model.visualize_spec()
        print(spec.description)

        # 2.5. Try to Predict:
        from PIL import Image
        img = Image.open('../yolov3/data/samples/zidane_416.jpg')
        out = yolov3_model.predict({'0': img}, useCPUOnly=True)
        print(out['148'].shape, out['150'].shape)

        # 3. Create NMS protobuf
        import numpy as np

        nms_spec = coremltools.proto.Model_pb2.Model()
        nms_spec.specificationVersion = 3

        for i in range(2):
            decoder_output = yolov3_model._spec.description.output[i].SerializeToString()

            nms_spec.description.input.add()
            nms_spec.description.input[i].ParseFromString(decoder_output)

            nms_spec.description.output.add()
            nms_spec.description.output[i].ParseFromString(decoder_output)

        nms_spec.description.output[0].name = 'confidence'
        nms_spec.description.output[1].name = 'coordinates'

        output_sizes = [num_classes, 4]
        for i in range(2):
            ma_type = nms_spec.description.output[i].type.multiArrayType
            ma_type.shapeRange.sizeRanges.add()
            ma_type.shapeRange.sizeRanges[0].lowerBound = 0
            ma_type.shapeRange.sizeRanges[0].upperBound = -1
            ma_type.shapeRange.sizeRanges.add()
            ma_type.shapeRange.sizeRanges[1].lowerBound = output_sizes[i]
            ma_type.shapeRange.sizeRanges[1].upperBound = output_sizes[i]
            del ma_type.shape[:]

        nms = nms_spec.nonMaximumSuppression
        nms.confidenceInputFeatureName = '148'  # 1x507x80
        nms.coordinatesInputFeatureName = '150'  # 1x507x4
        nms.confidenceOutputFeatureName = 'confidence'
        nms.coordinatesOutputFeatureName = 'coordinates'
        nms.iouThresholdInputFeatureName = 'iouThreshold'
        nms.confidenceThresholdInputFeatureName = 'confidenceThreshold'

        nms.iouThreshold = 0.6
        nms.confidenceThreshold = 0.3
        nms.pickTop.perClass = True

        labels = np.loadtxt('../yolov3/data/coco.names', dtype=str, delimiter='\n')
        nms.stringClassLabels.vector.extend(labels)

        nms_model = coremltools.models.MLModel(nms_spec)
        nms_model.save(name + '_nms.mlmodel')

        # out_nms = nms_model.predict({
        #     '143': out['143'].squeeze().reshape((80, 507)),
        #     '144': out['144'].squeeze().reshape((4, 507))
        # })
        # print(out_nms['confidence'].shape, out_nms['coordinates'].shape)

        # # # 3.5 Add Softmax model
        # from coremltools.models import datatypes
        # from coremltools.models import neural_network
        #
        # input_features = [
        #     ("141", datatypes.Array(num_anchors, num_classes, 1)),
        #     ("143", datatypes.Array(num_anchors, 4, 1))
        # ]
        #
        # output_features = [
        #     ("141", datatypes.Array(num_anchors, num_classes, 1)),
        #     ("143", datatypes.Array(num_anchors, 4, 1))
        # ]
        #
        # builder = neural_network.NeuralNetworkBuilder(input_features, output_features)
        # builder.add_softmax(name="softmax_pcls",
        #                     dim=(0, 3, 2, 1),
        #                     input_name="scores",
        #                     output_name="permute_scores_output")
        # softmax_model = coremltools.models.MLModel(builder.spec)
        # softmax_model.save("softmax.mlmodel")

        # 4. Pipeline models togethor
        from coremltools.models import datatypes
        # from coremltools.models import neural_network
        from coremltools.models.pipeline import Pipeline

        input_features = [('0', datatypes.Array(3, 416, 416)),
                          ('iouThreshold', datatypes.Double()),
                          ('confidenceThreshold', datatypes.Double())]

        output_features = ['confidence', 'coordinates']

        pipeline = Pipeline(input_features, output_features)

        # Add 3rd dimension of size 1 (apparently not needed, produces error on compile)
        yolov3_output = yolov3_model._spec.description.output
        yolov3_output[0].type.multiArrayType.shape[:] = [num_anchors, num_classes, 1]
        yolov3_output[1].type.multiArrayType.shape[:] = [num_anchors, 4, 1]

        nms_input = nms_model._spec.description.input
        for i in range(2):
            nms_input[i].type.multiArrayType.shape[:] = yolov3_output[i].type.multiArrayType.shape[:]

        # And now we can add the three models, in order:
        pipeline.add_model(yolov3_model)

        pipeline.add_model(nms_model)

        # Correct datatypes
        pipeline.spec.description.input[0].ParseFromString(yolov3_model._spec.description.input[0].SerializeToString())
        pipeline.spec.description.output[0].ParseFromString(nms_model._spec.description.output[0].SerializeToString())
        pipeline.spec.description.output[1].ParseFromString(nms_model._spec.description.output[1].SerializeToString())

        # Update metadata
        pipeline.spec.description.metadata.versionString = 'yolov3-tiny.pt imported from PyTorch'
        pipeline.spec.description.metadata.shortDescription = 'https://github.com/ultralytics/yolov3'
        pipeline.spec.description.metadata.author = 'glenn.jocher@ultralytics.com'
        pipeline.spec.description.metadata.license = 'https://github.com/ultralytics/yolov3'

        user_defined_metadata = {'classes': ','.join(labels),
                                 'iou_threshold': str(nms.iouThreshold),
                                 'confidence_threshold': str(nms.confidenceThreshold)}
        pipeline.spec.description.metadata.userDefined.update(user_defined_metadata)

        # Save the model
        pipeline.spec.specificationVersion = 3
        final_model = coremltools.models.MLModel(pipeline.spec)
        final_model.save((name + '_pipelined.mlmodel'))


if __name__ == '__main__':
    main()
