using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Estimators
{
    /// <summary>
    /// The executor to run `Estimator` training and evaluation.
    /// </summary>
    internal class _TrainingExecutor
    {
        Estimator _estimator;
        EvalSpec _eval_spec;
        TrainSpec _train_spec;

        public _TrainingExecutor(Estimator estimator, TrainSpec train_spec, EvalSpec eval_spec)
        {
            _estimator = estimator;
            _train_spec = train_spec;
            _eval_spec = eval_spec;
        }

        public void run()
        {
            var config = _estimator.config;
            Console.WriteLine("Running training and evaluation locally (non-distributed).");
            run_local();
        }

        /// <summary>
        /// Runs training and evaluation locally (non-distributed).
        /// </summary>
        private void run_local()
        {
            var train_hooks = new Action[0];
            Console.WriteLine("Start train and evaluate loop. The evaluate will happen " +
                "after every checkpoint. Checkpoint frequency is determined " +
                $"based on RunConfig arguments: save_checkpoints_steps {_estimator.config.save_checkpoints_steps} or " +
                $"save_checkpoints_secs {_estimator.config.save_checkpoints_secs}.");
            var evaluator = new _Evaluator(_estimator, _eval_spec, _train_spec.max_steps);
            var saving_listeners = new _NewCheckpointListenerForEvaluate[0];
            _estimator.train(input_fn: _train_spec.input_fn,
                 max_steps: _train_spec.max_steps,
                 hooks: train_hooks,
                 saving_listeners: saving_listeners);
        }
    }
}
