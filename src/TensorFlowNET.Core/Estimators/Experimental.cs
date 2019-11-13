using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tensorflow.Estimators
{
    public class Experimental
    {
        /// <summary>
        /// Creates hook to stop if metric does not increase within given max steps.
        /// </summary>
        /// <typeparam name="Thyp">type of hyper parameters</typeparam>
        /// <param name="estimator"></param>
        /// <param name="metric_name"></param>
        /// <param name="max_steps_without_increase"></param>
        /// <param name="eval_dir"></param>
        /// <param name="min_steps"></param>
        /// <param name="run_every_secs"></param>
        /// <param name="run_every_steps"></param>
        /// <returns></returns>
        public object stop_if_no_increase_hook<Thyp>(Estimator<Thyp> estimator,
            string metric_name,
            int max_steps_without_increase,
            string eval_dir = null,
            int min_steps = 0,
            int run_every_secs = 60,
            int run_every_steps = 0)
        => _stop_if_no_metric_improvement_hook(estimator: estimator,
            metric_name: metric_name,
            max_steps_without_increase: max_steps_without_increase,
            eval_dir: eval_dir,
            min_steps: min_steps,
            run_every_secs: run_every_secs,
            run_every_steps: run_every_steps);

        private object _stop_if_no_metric_improvement_hook<Thyp>(Estimator<Thyp> estimator,
            string metric_name,
            int max_steps_without_increase,
            string eval_dir = null,
            int min_steps = 0,
            int run_every_secs = 60,
            int run_every_steps = 0)
        {
            eval_dir = eval_dir ?? estimator.eval_dir();
            // var is_lhs_better = higher_is_better ? operator.gt: operator.lt;
            Func<bool> stop_if_no_metric_improvement_fn = () =>
            {
                return false;
            };

            return make_early_stopping_hook();
        }

        public object make_early_stopping_hook()
        {
            throw new NotImplementedException("");
        }
    }
}
