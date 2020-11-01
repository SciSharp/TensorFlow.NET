using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Engine
{
    public class MetricsContainer : Container
    {
        string[] _user_metrics;
        string[] _metrics;

        public MetricsContainer(string[] metrics, string[] output_names = null)
            : base(output_names)
        {
            _user_metrics = metrics;
            _metrics = metrics;
            _built = false;
        }

        public void update_state(Tensor y_true, Tensor y_pred, Tensor sample_weight = null)
        {
            if (!_built)
                Build();

            _built = true;
        }

        void Build()
        {

        }
    }
}
