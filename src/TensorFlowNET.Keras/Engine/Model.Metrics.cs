using System.Collections.Generic;
using Tensorflow.Keras.Metrics;

namespace Tensorflow.Keras.Engine
{
    public partial class Model
    {
        public IEnumerable<Metric> metrics
        {
            get
            {
                var _metrics = new List<Metric>();

                if (_is_compiled)
                {
                    if (compiled_loss != null)
                        _metrics.add(compiled_loss.metrics);
                    if (compiled_metrics != null)
                        _metrics.add(compiled_metrics.metrics);
                }

                /*foreach (var layer in _flatten_layers())
                    _metrics.extend(layer.metrics);*/

                return _metrics;
            }
        }

        void reset_metrics()
        {
            foreach (var metric in metrics)
                metric.reset_states();
        }
    }
}
