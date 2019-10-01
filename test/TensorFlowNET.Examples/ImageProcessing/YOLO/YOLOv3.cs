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
        Tensor pred_sbbox;
        Tensor pred_mbbox;
        Tensor pred_lbbox;

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
                pred_sbbox = decode(conv_sbbox, anchors[0], strides[0]);
            });

            tf_with(tf.variable_scope("pred_mbbox"), scope =>
            {
                pred_mbbox = decode(conv_mbbox, anchors[1], strides[1]);
            });

            tf_with(tf.variable_scope("pred_lbbox"), scope =>
            {
                pred_lbbox = decode(conv_lbbox, anchors[2], strides[2]);
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

            tf_with(tf.variable_scope("route_1"), delegate
            {
                input_data = tf.concat(new[] { input_data, route_2 }, axis: -1);
            });

            input_data = common.convolutional(input_data, new[] { 1, 1, 768, 256 }, trainable, "conv58");
            input_data = common.convolutional(input_data, new[] { 3, 3, 256, 512 }, trainable, "conv59");
            input_data = common.convolutional(input_data, new[] { 1, 1, 512, 256 }, trainable, "conv60");
            input_data = common.convolutional(input_data, new[] { 3, 3, 256, 512 }, trainable, "conv61");
            input_data = common.convolutional(input_data, new[] { 1, 1, 512, 256 }, trainable, "conv62");

            var conv_mobj_branch = common.convolutional(input_data, new[] { 3, 3, 256, 512 }, trainable, name: "conv_mobj_branch");
            conv_mbbox = common.convolutional(conv_mobj_branch, new[] { 1, 1, 512, 3 * (num_class + 5) },
                                          trainable: trainable, name: "conv_mbbox", activate: false, bn: false);

            input_data = common.convolutional(input_data, new[] { 1, 1, 256, 128 }, trainable, "conv63");
            input_data = common.upsample(input_data, name: "upsample1", method: upsample_method);

            tf_with(tf.variable_scope("route_2"), delegate
            {
                input_data = tf.concat(new[] { input_data, route_1 }, axis: -1);
            });

            input_data = common.convolutional(input_data, new[] { 1, 1, 384, 128 }, trainable, "conv64");
            input_data = common.convolutional(input_data, new[] { 3, 3, 128, 256 }, trainable, "conv65");
            input_data = common.convolutional(input_data, new[] { 1, 1, 256, 128 }, trainable, "conv66");
            input_data = common.convolutional(input_data, new[] { 3, 3, 128, 256 }, trainable, "conv67");
            input_data = common.convolutional(input_data, new[] { 1, 1, 256, 128 }, trainable, "conv68");

            var conv_sobj_branch = common.convolutional(input_data, new[] { 3, 3, 128, 256 }, trainable, name: "conv_sobj_branch");
            conv_sbbox = common.convolutional(conv_sobj_branch, new[] { 1, 1, 256, 3 * (num_class + 5) },
                                          trainable: trainable, name: "conv_sbbox", activate: false, bn: false);
            
            return (conv_lbbox, conv_mbbox, conv_sbbox);
        }

        private Tensor decode(Tensor conv_output, NDArray anchors, int stride)
        {
            var conv_shape = tf.shape(conv_output);
            var batch_size = conv_shape[0];
            var output_size = conv_shape[1];
            anchor_per_scale = len(anchors);

            conv_output = tf.reshape(conv_output, new object[] { batch_size, output_size, output_size, anchor_per_scale, 5 + num_class });

            var conv_raw_dxdy = conv_output[":", ":", ":", ":", "0:2"];
            var conv_raw_dwdh = conv_output[":", ":", ":", ":", "2:4"];
            var conv_raw_conf = conv_output[":", ":", ":", ":", "4:5"];
            var conv_raw_prob = conv_output[":", ":", ":", ":", "5:"];

            var y = tf.tile(tf.range(output_size, dtype: tf.int32)[":", tf.newaxis], new object[] { 1, output_size });
            var x = tf.tile(tf.range(output_size, dtype: tf.int32)[tf.newaxis, ":"], new object[] { output_size, 1 });

            var xy_grid = tf.concat(new[] { x[":", ":", tf.newaxis], y[":", ":", tf.newaxis] }, axis: -1);
            xy_grid = tf.tile(xy_grid[tf.newaxis, ":", ":", tf.newaxis, ":"], new object[] { batch_size, 1, 1, anchor_per_scale, 1 });
            xy_grid = tf.cast(xy_grid, tf.float32);

            var pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride;
            var pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride;
            var pred_xywh = tf.concat(new[] { pred_xy, pred_wh }, axis: -1);

            var pred_conf = tf.sigmoid(conv_raw_conf);
            var pred_prob = tf.sigmoid(conv_raw_prob);

            return tf.concat(new[] { pred_xywh, pred_conf, pred_prob }, axis: -1);
        }

        public (Tensor, Tensor, Tensor) compute_loss(Tensor label_sbbox, Tensor label_mbbox, Tensor label_lbbox,
            Tensor true_sbbox, Tensor true_mbbox, Tensor true_lbbox)
        {
            Tensor giou_loss = null, conf_loss = null, prob_loss = null;
            (Tensor, Tensor, Tensor) loss_sbbox = (null, null, null);
            (Tensor, Tensor, Tensor) loss_mbbox = (null, null, null);
            (Tensor, Tensor, Tensor) loss_lbbox = (null, null, null);

            tf_with(tf.name_scope("smaller_box_loss"), delegate
            {
                loss_sbbox = loss_layer(conv_sbbox, pred_sbbox, label_sbbox, true_sbbox,
                                         anchors: anchors[0], stride: strides[0]);
            });

            tf_with(tf.name_scope("medium_box_loss"), delegate
            {
                loss_mbbox = loss_layer(conv_mbbox, pred_mbbox, label_mbbox, true_mbbox,
                                         anchors: anchors[1], stride: strides[1]);
            });

            tf_with(tf.name_scope("bigger_box_loss"), delegate
            {
                loss_lbbox = loss_layer(conv_lbbox, pred_lbbox, label_lbbox, true_lbbox,
                                         anchors: anchors[2], stride: strides[2]);
            });

            tf_with(tf.name_scope("giou_loss"), delegate
            {
                giou_loss = loss_sbbox.Item1 + loss_mbbox.Item1 + loss_lbbox.Item1;
            });

            tf_with(tf.name_scope("conf_loss"), delegate
            {
                conf_loss = loss_sbbox.Item2 + loss_mbbox.Item2 + loss_lbbox.Item2;
            });

            tf_with(tf.name_scope("prob_loss"), delegate
            {
                prob_loss = loss_sbbox.Item3 + loss_mbbox.Item3 + loss_lbbox.Item3;
            });

            return (giou_loss, conf_loss, prob_loss);
        }

        public (Tensor, Tensor, Tensor) loss_layer(Tensor conv, Tensor pred, Tensor label, Tensor bboxes, NDArray anchors, int stride)
        {
            var conv_shape = tf.shape(conv);
            var batch_size = conv_shape[0];
            var output_size = conv_shape[1];
            var input_size = stride * output_size;
            conv = tf.reshape(conv, new object[] {batch_size, output_size, output_size,
                                 anchor_per_scale, 5 + num_class });
            var conv_raw_conf = conv[":", ":", ":", ":", "4:5"];
            var conv_raw_prob = conv[":", ":", ":", ":", "5:"];

            var pred_xywh = pred[":", ":", ":", ":", "0:4"];
            var pred_conf = pred[":", ":", ":", ":", "4:5"];

            var label_xywh = label[":", ":", ":", ":", "0:4"];
            var respond_bbox = label[":", ":", ":", ":", "4:5"];
            var label_prob = label[":", ":", ":", ":", "5:"];

            var giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis: -1);
            input_size = tf.cast(input_size, tf.float32);

            var bbox_loss_scale = 2.0 - 1.0 * label_xywh[":", ":", ":", ":", "2:3"] * label_xywh[":", ":", ":", ":", "3:4"] / (tf.sqrt(input_size));
            var giou_loss = respond_bbox * bbox_loss_scale * (1 - giou);

            var iou = bbox_iou(pred_xywh[":", ":", ":", ":", tf.newaxis, ":"], bboxes[":", tf.newaxis, tf.newaxis, tf.newaxis, ":", ":"]);
            var max_iou = tf.expand_dims(tf.reduce_max(iou, axis: new[] { -1 }), axis: -1);

            var respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < iou_loss_thresh, tf.float32);

            var conf_focal = focal(respond_bbox, pred_conf);

            var conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels: respond_bbox, logits: conv_raw_conf) +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels: respond_bbox, logits: conv_raw_conf));

            var prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels: label_prob, logits: conv_raw_prob);

            giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis: new[] { 1, 2, 3, 4 }));
            conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis: new[] { 1, 2, 3, 4 }));
            prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis: new[] { 1, 2, 3, 4 }));

            return (giou_loss, conf_loss, prob_loss);
        }

        public Tensor focal(Tensor target, Tensor actual, int alpha = 1, int gamma = 2)
        {
            var focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma);
            return focal_loss;
        }

        public Tensor bbox_giou(Tensor boxes1, Tensor boxes2)
        {
            boxes1 = tf.concat(new[] { boxes1["...", ":2"] - boxes1["...", "2:"] * 0.5,
                            boxes1["...", ":2"] + boxes1["...", "2:"] * 0.5}, axis: -1);
            boxes2 = tf.concat(new[] { boxes2["...", ":2"] - boxes2["...", "2:"] * 0.5,
                            boxes2["...", ":2"] + boxes2["...", "2:"] * 0.5}, axis: -1);

            boxes1 = tf.concat(new[] { tf.minimum(boxes1["...", ":2"], boxes1["...", "2:"]),
                            tf.maximum(boxes1["...", ":2"], boxes1["...", "2:"])}, axis: -1);
            boxes2 = tf.concat(new[] { tf.minimum(boxes2["...", ":2"], boxes2["...", "2:"]),
                            tf.maximum(boxes2["...", ":2"], boxes2["...", "2:"])}, axis: -1);

            var boxes1_area = (boxes1["...", "2"] - boxes1["...", "0"]) * (boxes1["...", "3"] - boxes1["...", "1"]);
            var boxes2_area = (boxes2["...", "2"] - boxes2["...", "0"]) * (boxes2["...", "3"] - boxes2["...", "1"]);

            var left_up = tf.maximum(boxes1["...", ":2"], boxes2["...", ":2"]);
            var right_down = tf.minimum(boxes1["...", "2:"], boxes2["...", "2:"]);

            var inter_section = tf.maximum(right_down - left_up, 0.0f);
            var inter_area = inter_section["...", "0"] * inter_section["...", "1"];
            var union_area = boxes1_area + boxes2_area - inter_area;
            var iou = inter_area / union_area;

            var enclose_left_up = tf.minimum(boxes1["...", ":2"], boxes2["...", ":2"]);
            var enclose_right_down = tf.maximum(boxes1["...", "2:"], boxes2["...", "2:"]);
            var enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0);
            var enclose_area = enclose["...", "0"] * enclose["...", "1"];
            var giou = iou - 1.0 * (enclose_area - union_area) / enclose_area;

            return giou;
        }

        public Tensor bbox_iou(Tensor boxes1, Tensor boxes2)
        {
            var boxes1_area = boxes1["...", "2"] * boxes1["...", "3"];
            var boxes2_area = boxes2["...", "2"] * boxes2["...", "3"];

            boxes1 = tf.concat(new[] { boxes1["...", ":2"] - boxes1["...", "2:"] * 0.5,
                            boxes1["...", ":2"] + boxes1["...", "2:"] * 0.5}, axis: -1);
            boxes2 = tf.concat(new[] { boxes2["...", ":2"] - boxes2["...", "2:"] * 0.5,
                            boxes2["...", ":2"] + boxes2["...", "2:"] * 0.5}, axis: -1);

            var left_up = tf.maximum(boxes1["...", ":2"], boxes2["...", ":2"]);
            var right_down = tf.minimum(boxes1["...", "2:"], boxes2["...", "2:"]);

            var inter_section = tf.maximum(right_down - left_up, 0.0);
            var inter_area = inter_section["...", "0"] * inter_section["...", "1"];
            var union_area = boxes1_area + boxes2_area - inter_area;
            var iou = 1.0 * inter_area / union_area;

            return iou;
        }
    }
}
