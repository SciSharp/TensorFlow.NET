using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.Estimators
{
    /// <summary>
    /// Estimator class to train and evaluate TensorFlow models.
    /// </summary>
    public class Estimator : IObjectLife
    {
        RunConfig _config;
        public RunConfig config => _config;

        ConfigProto _session_config;
        public ConfigProto session_config => _session_config;

        string _model_dir;

        public Estimator(RunConfig config)
        {
            _config = config;
            _model_dir = _config.model_dir;
            _session_config = _config.session_config;
        }

        public Estimator train(Action input_fn, int max_steps = 1, 
            _NewCheckpointListenerForEvaluate[] saving_listeners = null)
        {
            _train_model();
            throw new NotImplementedException("");
        }

        private void _train_model()
        {
            _train_model_default();
        }

        private void _train_model_default()
        {
            using (var g = tf.Graph().as_default())
            {

            }
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
            throw new NotImplementedException();
        }
    }
}
