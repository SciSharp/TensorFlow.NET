using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Estimator
{
    public class HyperParams
    {
        public string data_dir { get; set; }
        public string result_dir { get; set; }
        public string model_dir { get; set; }
        public string eval_dir { get; set; }

        public int dim { get; set; } = 300;
        public float dropout { get; set; } = 0.5f;
        public int num_oov_buckets { get; set; } = 1;
        public int epochs { get; set; } = 25;
        public int batch_size { get; set; } = 20;
        public int buffer { get; set; } = 15000;
        public int lstm_size { get; set; } = 100;

        public string words { get; set; }
        public string chars { get; set; }
        public string tags { get; set; }
        public string glove { get; set; }
    }
}
