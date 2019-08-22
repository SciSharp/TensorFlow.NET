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
        public TrainConfig TEST;

        public Config(string root)
        {
            YOLO = new YoloConfig(root);
            TRAIN = new TrainConfig(root);
        }

        public class YoloConfig
        {
            string _root;

            public string CLASSES;
            public float MOVING_AVE_DECAY = 0.9995f;
            public int[] STRIDES = new int[] { 8, 16, 32 };

            public YoloConfig(string root)
            {
                _root = root;
                CLASSES = Path.Combine(_root, "data", "classes", "coco.names");
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
    }
}
