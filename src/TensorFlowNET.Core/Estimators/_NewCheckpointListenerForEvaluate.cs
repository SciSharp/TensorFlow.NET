using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Estimators
{
    public class _NewCheckpointListenerForEvaluate<Thyp>
    {
        _Evaluator<Thyp> _evaluator;

        public _NewCheckpointListenerForEvaluate(_Evaluator<Thyp> evaluator, int eval_throttle_secs)
        {
            _evaluator = evaluator;
        }
    }
}
