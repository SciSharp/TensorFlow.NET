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

            tf_with(tf.variable_scope("pred_sbbox"), scope =>
            {
                // pred_sbbox = decode(conv_sbbox, anchors[0], strides[0]);
            });

            tf_with(tf.variable_scope("pred_mbbox"), scope =>
            {
                // pred_sbbox = decode(conv_sbbox, anchors[0], strides[0]);
            });

            tf_with(tf.variable_scope("pred_lbbox"), scope =>
            {
                // pred_sbbox = decode(conv_sbbox, anchors[0], strides[0]);
            });
        }

        private (Tensor, Tensor, Tensor) __build_nework(Tensor input_data)
        {
            Tensor route_1, route_2;
            (route_1, route_2, input_data) = backbone.darknet53(input_data, trainable);
            input_data = common.convolutional(input_data, new[] { 1, 1, 1024, 512 }, trainable, "conv52");
            input_data = common.convolutional(input_data, new[] { 3, 3, 512, 1024 }, trainable, "conv53");
            input_data = common.convolutional(input_data, new[] { 1, 1, 1024, 512 }, trainable, "conv54");
            input_data = common.convolutional(input_data, new[] { 3, 3, 512, 1024 }, trainable, "conv55");
            input_data = common.convolutional(input_data, new[] { 1, 1, 1024, 512 }, trainable, "conv56");

            var conv_lobj_branch = common.convolutional(input_data, new[] { 3, 3, 512, 1024 }, trainable, name: "conv_lobj_branch");
            var conv_lbbox = common.convolutional(conv_lobj_branch, new[] { 1, 1, 1024, 3 * (num_class + 5) },
                                          trainable: trainable, name: "conv_lbbox", activate: false, bn: false);

            input_data = common.convolutional(input_data, new[] { 1, 1, 512, 256 }, trainable, "conv57");
            input_data = common.upsample(input_data, name: "upsample0", method: upsample_method);

            return (conv_lbbox, conv_mbbox, conv_sbbox);
        }
    }
}
