import os
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

        model_file = open(f, 'rb')
        model_proto = onnx_pb.ModelProto()
        model_proto.ParseFromString(model_file.read())
        coreml_model = convert(model_proto, image_input_names=['0'])
        # coreml_model.save(model_out)

        # 2. Reduce model to FP16, change outputs to DOUBLE and save
        import coremltools

        spec = coreml_model.get_spec()
        for i in range(2):
            spec.description.output[i].type.multiArrayType.dataType = \
                coremltools.proto.FeatureTypes_pb2.ArrayFeatureType.ArrayDataType.Value('DOUBLE')

        spec = coremltools.utils.convert_neural_network_spec_weights_to_fp16(spec)
        coreml_model = coremltools.models.MLModel(spec)

        num_classes = 80
        num_anchors = 507
        spec.description.output[0].type.multiArrayType.shape.append(num_classes)
        spec.description.output[0].type.multiArrayType.shape.append(num_anchors)

        spec.description.output[1].type.multiArrayType.shape.append(4)
        spec.description.output[1].type.multiArrayType.shape.append(num_anchors)
        coreml_model.save(name + '.mlmodel')
        print(spec.description)

        # 3. Create NMS protobuf
        import numpy as np

        nms_spec = coremltools.proto.Model_pb2.Model()
        nms_spec.specificationVersion = 3

        for i in range(2):
            decoder_output = coreml_model._spec.description.output[i].SerializeToString()

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
        nms.confidenceInputFeatureName = '133'  # 1x507x80
        nms.coordinatesInputFeatureName = '134'  # 1x507x4
        nms.confidenceOutputFeatureName = 'confidence'
        nms.coordinatesOutputFeatureName = 'coordinates'
        nms.iouThresholdInputFeatureName = 'iouThreshold'
        nms.confidenceThresholdInputFeatureName = 'confidenceThreshold'

        nms.iouThreshold = 0.6
        nms.confidenceThreshold = 0.4
        nms.pickTop.perClass = True

        labels = np.loadtxt('../yolov3/data/coco.names', dtype=str, delimiter='\n')
        nms.stringClassLabels.vector.extend(labels)

        nms_model = coremltools.models.MLModel(nms_spec)
        nms_model.save(name + '_nms.mlmodel')

        # 4. Pipeline models togethor
        from coremltools.models import datatypes
        # from coremltools.models import neural_network
        from coremltools.models.pipeline import Pipeline

        input_features = [('image', datatypes.Array(3, 416, 416)),
                          ('iouThreshold', datatypes.Double()),
                          ('confidenceThreshold', datatypes.Double())]

        output_features = ['confidence', 'coordinates']

        pipeline = Pipeline(input_features, output_features)

        # Add 3rd dimension of size 1 (apparently not needed, produces error on compile)
        # ssd_output = coreml_model._spec.description.output
        # ssd_output[0].type.multiArrayType.shape[:] = [num_classes, num_anchors, 1]
        # ssd_output[1].type.multiArrayType.shape[:] = [4, num_anchors, 1]

        # And now we can add the three models, in order:
        pipeline.add_model(coreml_model)
        pipeline.add_model(nms_model)

        # Correct datatypes
        pipeline.spec.description.input[0].ParseFromString(coreml_model._spec.description.input[0].SerializeToString())
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
