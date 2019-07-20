using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;

namespace Tensorflow.Hub
{
    public class ModelLoadSetting
    {
        public string TrainDir { get; set; }
        public bool OneHot { get; set; }
        public TF_DataType DtType { get; set; } = TF_DataType.TF_FLOAT;
        public bool ReShape { get; set; }
        public int ValidationSize { get; set; } = 5000;
        public int? TrainSize  { get; set; }
        public int? TestSize { get; set; }
        public string SourceUrl { get; set; }
    }
}
