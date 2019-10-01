using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Data;
using Tensorflow.Estimators;

namespace Tensorflow.Models.ObjectDetection
{
    public class TrainAndEvalDict
    {
        public Estimator estimator { get; set; }
        public Func<DatasetV1Adapter> train_input_fn { get; set; }
        public Action[] eval_input_fns { get; set; }
        public string[] eval_input_names { get; set; }
        public Action eval_on_train_input_fn { get; set; }
        public Action predict_input_fn { get; set; }
        public int train_steps { get; set; }
    }
}
