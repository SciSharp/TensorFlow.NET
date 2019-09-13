using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Estimators
{
    public class _NewCheckpointListenerForEvaluate
    {
        _Evaluator _evaluator;

        public _NewCheckpointListenerForEvaluate(_Evaluator evaluator, int eval_throttle_secs)
        {
            _evaluator = evaluator;
        }
    }
}
