using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.Examples.ImageProcessing.YOLO
{
    public class YOLOv3
    {
        Config cfg;
        Tensor trainable;
        Tensor input_data;
        Dictionary<int, string> classes;
        int num_class;
        NDArray strides;
        NDArray anchors;
        int anchor_per_scale;
        float iou_loss_thresh;
        string upsample_method;
        Tensor conv_lbbox;
        Tensor conv_mbbox;
        Tensor conv_sbbox;

        public YOLOv3(Config cfg_, Tensor input_data_, Tensor trainable_)
        {
            cfg = cfg_;
            input_data = input_data_;
            trainable = trainable_;
            classes = Utils.read_class_names(cfg.YOLO.CLASSES);
            num_class = len(classes);
            strides = np.array(cfg.YOLO.STRIDES);
            anchors = Utils.get_anchors(cfg.YOLO.ANCHORS);
            anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE;
            iou_loss_thresh = cfg.YOLO.IOU_LOSS_THRESH;
            upsample_method = cfg.YOLO.UPSAMPLE_METHOD;

            (conv_lbbox, conv_mbbox, conv_sbbox) = __build_nework(input_data);
        }

        private (Tensor, Tensor, Tensor) __build_nework(Tensor input_data)
        {
            Tensor route_1, route_2;
            (route_1, route_2, input_data) = backbone.darknet53(input_data, trainable);

            return (conv_lbbox, conv_mbbox, conv_sbbox);
        }
    }
}
