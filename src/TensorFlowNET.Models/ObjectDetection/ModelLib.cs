using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;
using Tensorflow.Estimators;

namespace Tensorflow.Models.ObjectDetection
{
    public class ModelLib
    {
        public void create_estimator_and_inputs(RunConfig run_config)
        {
            var estimator = tf.estimator.Estimator(config: run_config);
        }

        public void create_train_and_eval_specs()
        {

        }
    }
}
