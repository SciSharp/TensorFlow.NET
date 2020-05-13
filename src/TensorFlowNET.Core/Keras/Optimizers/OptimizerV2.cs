using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Train;

namespace Tensorflow.Keras.Optimizers
{
    /// <summary>
    /// Updated base class for optimizers.
    /// </summary>
    public class OptimizerV2 : Trackable, IOptimizer
    {
        public OptimizerV2() : base()
        {

        }

        public void apply_gradients((Tensor, Tensor) gradients, 
            (ResourceVariable, ResourceVariable) vars)
        {

        }
    }
}
