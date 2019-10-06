using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Estimators
{
    public class Training
    {
        public static void train_and_evaluate(Estimator estimator, TrainSpec train_spec, EvalSpec eval_spec)
        {
            var executor = new _TrainingExecutor(estimator: estimator, train_spec: train_spec, eval_spec: eval_spec);
            var config = estimator.config;

            executor.run();
        }
    }
}
