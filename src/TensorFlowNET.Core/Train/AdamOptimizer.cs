using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Train
{
    /// <summary>
    /// Optimizer that implements the Adam algorithm.
    /// http://arxiv.org/abs/1412.6980
    /// </summary>
    public class AdamOptimizer : Optimizer
    {
        private float _beta1;
        private float _beta2;
        private float _epsilon;

        public AdamOptimizer(float learning_rate, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f, bool use_locking = false, string name = "Adam")
            : base(learning_rate, use_locking, name)
        {
            _beta1 = beta1;
            _beta2 = beta2;
            _epsilon = epsilon;
        }
    }
}
