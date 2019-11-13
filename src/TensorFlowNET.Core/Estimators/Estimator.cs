using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using Tensorflow.Data;
using Tensorflow.Train;
using static Tensorflow.Binding;

namespace Tensorflow.Estimators
{
    /// <summary>
    /// Estimator class to train and evaluate TensorFlow models.
    /// </summary>
    public class Estimator<Thyp> : IObjectLife
    {
        RunConfig _config;
        public RunConfig config => _config;

        ConfigProto _session_config;
        public ConfigProto session_config => _session_config;

        Func<IEstimatorInputs, EstimatorSpec> _model_fn;

        Thyp _hyperParams;

        public Estimator(Func<IEstimatorInputs, EstimatorSpec> model_fn,
            string model_dir,
            RunConfig config,
            Thyp hyperParams)
        {
            _config = config;
            _config.model_dir = config.model_dir ?? model_dir;
            _session_config = config.session_config;
            _model_fn = model_fn;
            _hyperParams = hyperParams;
        }

        public Estimator<Thyp> train(Func<DatasetV1Adapter> input_fn, int max_steps = 1, Action[] hooks = null,
            _NewCheckpointListenerForEvaluate<Thyp>[] saving_listeners = null)
        {
            if(max_steps > 0)
            {
                var start_step = _load_global_step_from_checkpoint_dir(_config.model_dir);
                if (max_steps <= start_step)
                {
                    Console.WriteLine("Skipping training since max_steps has already saved.");
                    return this;
                }
            }

            var loss = _train_model(input_fn);
            print($"Loss for final step: {loss}.");
            return this;
        }

        private int _load_global_step_from_checkpoint_dir(string checkpoint_dir)
        {
            // var cp = tf.train.latest_checkpoint(checkpoint_dir);
            // should use NewCheckpointReader (not implemented)
            var cp = tf.train.get_checkpoint_state(checkpoint_dir);

            return cp.AllModelCheckpointPaths.Count - 1;
        }

        private Tensor _train_model(Func<DatasetV1Adapter> input_fn)
        {
            return _train_model_default(input_fn);
        }

        private Tensor _train_model_default(Func<DatasetV1Adapter> input_fn)
        {
            using (var g = tf.Graph().as_default())
            {
                var global_step_tensor = _create_and_assert_global_step(g);

                // Skip creating a read variable if _create_and_assert_global_step
                // returns None (e.g. tf.contrib.estimator.SavedModelEstimator).
                if (global_step_tensor != null)
                    TrainingUtil._get_or_create_global_step_read(g);

                var (features, labels) = _get_features_and_labels_from_input_fn(input_fn, "train");
            }

            throw new NotImplementedException("");
        }

        private (Dictionary<string, Tensor>, Dictionary<string, Tensor>) _get_features_and_labels_from_input_fn(Func<DatasetV1Adapter> input_fn, string mode)
        {
            var result = _call_input_fn(input_fn, mode);
            return EstimatorUtil.parse_input_fn_result(result);
        }

        /// <summary>
        /// Calls the input function.
        /// </summary>
        /// <param name="input_fn"></param>
        /// <param name="mode"></param>
        private DatasetV1Adapter _call_input_fn(Func<DatasetV1Adapter> input_fn, string mode)
        {
            return input_fn();
        }

        private Tensor _create_and_assert_global_step(Graph graph)
        {
            var step = _create_global_step(graph);
            Debug.Assert(step == tf.train.get_global_step(graph));
            Debug.Assert(step.dtype.is_integer());
            return step;
        }

        private RefVariable _create_global_step(Graph graph)
        {
            return tf.train.create_global_step(graph);
        }

        public string eval_dir(string name = null)
        {
            return Path.Combine(config.model_dir, string.IsNullOrEmpty(name) ? "eval" : $"eval_" + name);
        }

        public void __init__()
        {
            throw new NotImplementedException();
        }

        public void __enter__()
        {
            throw new NotImplementedException();
        }

        public void __del__()
        {
            throw new NotImplementedException();
        }

        public void __exit__()
        {
            throw new NotImplementedException();
        }

        public void Dispose()
        {
            
        }
    }
}
