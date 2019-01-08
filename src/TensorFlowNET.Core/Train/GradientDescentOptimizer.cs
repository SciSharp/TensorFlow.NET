using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class GradientDescentOptimizer : Optimizer
    {
        public GradientDescentOptimizer(double learning_rate, bool use_locking = false, string name = "GradientDescent") 
            : base(learning_rate, use_locking, name)
        {
            LearningRate = learning_rate;
            LearningRateTensor = null;
        }
    }
}
