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
    }
}
