using NumSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using static Tensorflow.Binding;

namespace TensorFlowNET.Examples.ImageProcessing.YOLO
{
    public class Dataset
    {
        string annot_path;
        int[] input_sizes;
        int batch_size;
        bool data_aug;
        int[] train_input_sizes;
        NDArray strides;
        NDArray anchors;
        Dictionary<int, string> classes;
        int num_classes;
        int anchor_per_scale;
        int max_bbox_per_scale;
        string[] annotations;
        int num_samples;
        int batch_count;

        public int Length = 0;

        public Dataset(string dataset_type, Config cfg)
        {
            annot_path = dataset_type == "train" ? cfg.TRAIN.ANNOT_PATH : cfg.TEST.ANNOT_PATH;
            input_sizes = dataset_type == "train" ? cfg.TRAIN.INPUT_SIZE : cfg.TEST.INPUT_SIZE;
            batch_size = dataset_type == "train" ? cfg.TRAIN.BATCH_SIZE : cfg.TEST.BATCH_SIZE;
            data_aug = dataset_type == "train" ? cfg.TRAIN.DATA_AUG : cfg.TEST.DATA_AUG;
            train_input_sizes = cfg.TRAIN.INPUT_SIZE;
            strides = np.array(cfg.YOLO.STRIDES);

            classes = Utils.read_class_names(cfg.YOLO.CLASSES);
            num_classes = classes.Count;
            anchors = np.array(Utils.get_anchors(cfg.YOLO.ANCHORS));
            anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE;
            max_bbox_per_scale = 150;

            annotations = load_annotations();
            num_samples = len(annotations);
            batch_count = 0;
        }

        string[] load_annotations()
        {
            return File.ReadAllLines(annot_path);
        }
    }
}
