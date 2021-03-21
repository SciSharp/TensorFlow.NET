using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Keras.Metrics;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.Engine
{
    public class MetricsContainer : Container
    {
        string[] _user_metrics;
        string[] _metric_names;
        Metric[] _metrics;
        List<Metric> _metrics_in_order;

        public MetricsContainer(string[] metrics, string[] output_names = null)
            : base(output_names)
        {
            _user_metrics = metrics;
            _metric_names = metrics;
            _built = false;
        }

        public void update_state(Tensor y_true, Tensor y_pred, Tensor sample_weight = null)
        {
            if (!_built)
                Build(y_true, y_pred);

            foreach (var metric_obj in _metrics_in_order)
                metric_obj.update_state(y_true, y_pred);
        }

        void Build(Tensor y_true, Tensor y_pred)
        {
            _metrics = _get_metric_objects(_metric_names, y_true, y_pred);
            _set_metric_names();
            _create_ordered_metrics();
            _built = true;
        }

        void _set_metric_names()
        {

        }

        void _create_ordered_metrics()
        {
            _metrics_in_order = new List<Metric>();
            foreach (var m in _metrics)
                _metrics_in_order.append(m);
        }

        Metric[] _get_metric_objects(string[] metrics, Tensor y_t, Tensor y_p)
        {
            return metrics.Select(x => _get_metric_object(x, y_t, y_p)).ToArray();
        }

        Metric _get_metric_object(string metric, Tensor y_t, Tensor y_p)
        {
            Func<Tensor, Tensor, Tensor> metric_obj = null;
            if (metric == "accuracy" || metric == "acc")
            {
                var y_t_rank = y_t.rank;
                var y_p_rank = y_p.rank;
                var y_t_last_dim = y_t.shape[y_t.shape.Length - 1];
                var y_p_last_dim = y_p.shape[y_p.shape.Length - 1];

                bool is_binary = y_p_last_dim == 1;
                bool is_sparse_categorical = (y_t_rank < y_p_rank || y_t_last_dim == 1) && y_p_last_dim > 1;

                if (is_binary)
                    metric_obj = keras.metrics.binary_accuracy;
                else if (is_sparse_categorical)
                    metric_obj = keras.metrics.sparse_categorical_accuracy;
                else
                    metric_obj = keras.metrics.categorical_accuracy;

                return new MeanMetricWrapper(metric_obj, metric);
            }

            throw new NotImplementedException("");
        }

        public IEnumerable<Metric> metrics
        {
            get
            {
                if (!_built)
                    return new List<Metric>();

                return _metrics_in_order;
            }
        }
    }
}
