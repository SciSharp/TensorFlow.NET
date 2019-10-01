using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace TensorFlowNET.Examples.ImageProcessing.YOLO
{
    public class Config
    {
        public YoloConfig YOLO;
        public TrainConfig TRAIN;
        public TestConfig TEST;

        public Config(string root)
        {
            YOLO = new YoloConfig(root);
            TRAIN = new TrainConfig(root);
            TEST = new TestConfig(root);
        }

        public class YoloConfig
        {
            string _root;

            public string CLASSES;
            public string ANCHORS;
            public float MOVING_AVE_DECAY = 0.9995f;
            public int[] STRIDES = new int[] { 8, 16, 32 };
            public int ANCHOR_PER_SCALE = 3;
            public float IOU_LOSS_THRESH = 0.5f;
            public string UPSAMPLE_METHOD = "resize";
            public string ORIGINAL_WEIGHT;
            public string DEMO_WEIGHT;

            public YoloConfig(string root)
            {
                _root = root;
                CLASSES = Path.Combine(_root, "data", "classes", "coco.names");
                ANCHORS = Path.Combine(_root, "data", "anchors", "basline_anchors.txt");
                ORIGINAL_WEIGHT = Path.Combine(_root, "checkpoint", "yolov3_coco.ckpt");
                DEMO_WEIGHT = Path.Combine(_root, "checkpoint", "yolov3_coco_demo.ckpt");
            }
        }

        public class TrainConfig
        {
            string _root;

            public int BATCH_SIZE = 6;
            public int[] INPUT_SIZE = new int[] { 320, 352, 384, 416, 448, 480, 512, 544, 576, 608 };
            public bool DATA_AUG = true;
            public float LEARN_RATE_INIT = 1e-4f;
            public float LEARN_RATE_END = 1e-6f;
            public int WARMUP_EPOCHS = 2;
            public int FISRT_STAGE_EPOCHS = 20;
            public int SECOND_STAGE_EPOCHS = 30;
            public string INITIAL_WEIGHT;
            public string ANNOT_PATH;

            public TrainConfig(string root)
            {
                _root = root;
                INITIAL_WEIGHT = Path.Combine(_root, "data", "checkpoint", "yolov3_coco_demo.ckpt");
                ANNOT_PATH = Path.Combine(_root, "data", "dataset", "voc_train.txt");
            }
        }

        public class TestConfig
        {
            string _root;

            public int BATCH_SIZE = 2;
            public int[] INPUT_SIZE = new int[] { 544 };
            public bool DATA_AUG = false;
            public bool WRITE_IMAGE = true;
            public string WRITE_IMAGE_PATH;
            public string WEIGHT_FILE;
            public bool WRITE_IMAGE_SHOW_LABEL = true;
            public bool SHOW_LABEL = true;
            public int SECOND_STAGE_EPOCHS = 30;
            public float SCORE_THRESHOLD = 0.3f;
            public float IOU_THRESHOLD = 0.45f;
            public string ANNOT_PATH;

            public TestConfig(string root)
            {
                _root = root;
                ANNOT_PATH = Path.Combine(_root, "data", "dataset", "voc_test.txt");
                WRITE_IMAGE_PATH = Path.Combine(_root, "data", "detection");
                WEIGHT_FILE = Path.Combine(_root, "checkpoint", "yolov3_test_loss=9.2099.ckpt-5");
            }
        }
    }
}
