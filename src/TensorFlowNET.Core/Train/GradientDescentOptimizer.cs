using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Train
{
    /// <summary>
    /// Optimizer that implements the gradient descent algorithm.
    /// </summary>
    public class GradientDescentOptimizer : Optimizer
    {
        /// <summary>
        /// Construct a new gradient descent optimizer.
        /// </summary>
        /// <param name="learning_rate">A Tensor or a floating point value.  The learning
        /// rate to use.</param>
        /// <param name="use_locking">If true use locks for update operations.</param>
        /// <param name="name">Optional name prefix for the operations created when applying
        /// gradients.Defaults to "GradientDescent".</param>
        /// <remarks>
        /// When eager execution is enabled, `learning_rate` can be a callable that
        /// takes no arguments and returns the actual value to use.This can be useful
        /// for changing these values across different invocations of optimizer
        /// functions.
        /// </remarks>
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
