using System.Collections.Generic;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Metrics;

namespace Tensorflow.Keras.Engine
{
    public class LossesContainer : Container
    {
        ILossFunc _user_losses;
        ILossFunc _losses;
        Mean _loss_metric;
        bool _built;
        Tensor[] _per_output_metrics;

        public LossesContainer(ILossFunc losses, string[] output_names = null)
            : base(output_names)
        {
            _user_losses = losses;
            _losses = losses;
            _loss_metric = new Mean(name: "loss");
            _built = false;
        }

        /// <summary>
        /// Computes the overall loss.
        /// </summary>
        /// <param name="y_true"></param>
        /// <param name="y_pred"></param>
        public Tensor Call(Tensor y_true, Tensor y_pred, Tensor sample_weight = null)
        {
            if (!_built)
                Build(y_pred);
            var loss_value = _losses.Call(y_true, y_pred, sample_weight:sample_weight);
            var loss_metric_value = loss_value;
            var batch_dim = array_ops.shape(y_true)[0];

            var loss_values = new List<Tensor>();
            var loss_metric_values = new List<Tensor>();

            /*if (_losses.Reduction == ReductionV2.SUM_OVER_BATCH_SIZE
                || _losses.Reduction == ReductionV2.AUTO)
                loss_value = losses_utils.scale_loss_for_distribution(loss_value);*/
            loss_values.append(loss_value);
            loss_metric_values.append(loss_metric_value);

            if (loss_values.Count > 0)
            {
                var total_loss_metric_value = math_ops.add_n(loss_metric_values.ToArray());
                _loss_metric.update_state(total_loss_metric_value, batch_dim);
                // loss_values = losses_utils.cast_losses_to_common_dtype(loss_values);
                var total_loss = math_ops.add_n(loss_values.ToArray());
                return total_loss;
            }
            else
            {
                // Ok for a model to have no compiled loss.
                return array_ops.zeros(Shape.Null);
            }
        }

        public void Build(Tensor y_pred)
        {
            _create_metrics();
            _built = true;
        }

        void _create_metrics()
        {
            // _per_output_metrics = _output_names.Select(x => null);
        }

        public IEnumerable<Metric> metrics
        {
            get
            {
                if (!_built)
                    return new List<Metric>();

                return new[] { _loss_metric };
            }
        }
    }
}
