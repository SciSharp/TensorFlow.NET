using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Metrics;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.Engine
{
    public class MetricsContainer : Container
    {
        IMetricFunc[] _user_metrics = new IMetricFunc[0];
        string[] _metric_names = new string[0];
        Metric[] _metrics = new Metric[0];
        List<IMetricFunc> _metrics_in_order = new List<IMetricFunc>();

        public MetricsContainer(IMetricFunc[] metrics, string[] output_names = null)
            : base(output_names)
        {
            _user_metrics = metrics;
            _built = false;
        }

        public MetricsContainer(string[] metrics, string[] output_names = null)
            : base(output_names)
        {
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
            foreach (var m in _metrics)
                _metrics_in_order.append(m);

            foreach(var m in _user_metrics)
                _metrics_in_order.append(m);
        }

        Metric[] _get_metric_objects(string[] metrics, Tensor y_t, Tensor y_p)
        {
            return metrics.Select(x => _get_metric_object(x, y_t, y_p)).ToArray();
        }

        public Metric _get_metric_object(string metric, Tensor y_t, Tensor y_p)
        {
            Func<Tensor, Tensor, Tensor> metric_obj = null;
            if (metric == "accuracy" || metric == "acc")
            {
                var y_t_rank = y_t.rank;
                var y_p_rank = y_p.rank;
                var y_t_last_dim = y_t.shape[y_t.shape.ndim - 1];
                var y_p_last_dim = y_p.shape[y_p.shape.ndim - 1];

                bool is_binary = y_p_last_dim == 1;
                bool is_sparse_categorical = (y_t_rank < y_p_rank || y_t_last_dim == 1) && y_p_last_dim > 1;

                if (is_binary)
                    metric_obj = keras.metrics.binary_accuracy;
                else if (is_sparse_categorical)
                    metric_obj = keras.metrics.sparse_categorical_accuracy;
                else
                    metric_obj = keras.metrics.categorical_accuracy;

                metric = "accuracy";
            }
            else if(metric == "mean_absolute_error" || metric == "mae")
            {
                metric_obj = keras.metrics.mean_absolute_error;
                metric = "mean_absolute_error";
            }
            else if (metric == "mean_absolute_percentage_error" || metric == "mape")
            {
                metric_obj = keras.metrics.mean_absolute_percentage_error;
                metric = "mean_absolute_percentage_error";
            }
            else
                throw new NotImplementedException("");

            return new MeanMetricWrapper(metric_obj, metric);
        }

        public IEnumerable<IMetricFunc> metrics
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
