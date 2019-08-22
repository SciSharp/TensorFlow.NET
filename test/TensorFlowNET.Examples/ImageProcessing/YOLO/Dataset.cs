using System;
using System.Collections.Generic;
using System.Text;

namespace TensorFlowNET.Examples.ImageProcessing.YOLO
{
    public class Dataset
    {
        string annot_path;
        public int Length = 0;

        public Dataset(string dataset_type, Config cfg)
        {
            annot_path = dataset_type == "train" ? cfg.TRAIN.ANNOT_PATH : cfg.TEST.ANNOT_PATH;
        }
    }
}
