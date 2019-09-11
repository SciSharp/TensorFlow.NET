using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Estimators
{
    public class _TrainingExecutor
    {
        Estimator _estimator;
        EvalSpec _eval_spec;
        TrainSpec _train_spec;

        public _TrainingExecutor(Estimator estimator, TrainSpec train_spec, EvalSpec eval_spec)
        {
            _estimator = estimator;
        }

        public void run()
        {
            run_local();
        }

        private void run_local()
        {
            var evaluator = new _Evaluator(_estimator, _eval_spec, _train_spec.max_steps);
            /*_estimator.train(input_fn: _train_spec.input_fn,
                 max_steps: _train_spec.max_steps,
                 hooks: train_hooks,
                 saving_listeners: saving_listeners);*/
        }
    }
}
