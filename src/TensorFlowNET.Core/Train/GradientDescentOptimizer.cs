using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class GradientDescentOptimizer : Optimizer
    {
        public GradientDescentOptimizer(float learning_rate, bool use_locking = false, string name = "GradientDescent") 
            : base(learning_rate, use_locking, name)
        {
            LearningRate = learning_rate;
            LearningRateTensor = null;
        }

        public override void _prepare()
        {
            LearningRate = _call_if_callable(LearningRate);
            LearningRateTensor = ops.convert_to_tensor(LearningRate, name: "learning_rate");
        }
    }
}
