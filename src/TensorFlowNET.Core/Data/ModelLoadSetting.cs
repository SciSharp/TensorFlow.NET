using System;

namespace Tensorflow
{
    public class ModelLoadSetting
    {
        public string TrainDir { get; set; }
        public bool OneHot { get; set; }
        public TF_DataType DataType { get; set; } = TF_DataType.TF_FLOAT;
        public bool ReShape { get; set; }
        public int ValidationSize { get; set; } = 5000;
        public int? TrainSize { get; set; }
        public int? TestSize { get; set; }
        public string SourceUrl { get; set; }
        public bool ShowProgressInConsole { get; set; }
    }
}
