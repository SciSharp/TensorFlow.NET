using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Eager;
using static Tensorflow.tensorflow;

namespace Tensorflow.Gradients
{
    public partial class Tape
    {
        public Tensor[] CallBackwardFunction(BackwardFunction backward_function,
            List<long> unneeded_gradients,
            List<Tensor> output_gradients)
        {
            var grads = new Tensor[output_gradients.Count];
            var result = backward_function(output_gradients.ToArray(), 
                unneeded_gradients.ToArray());

            return result;
        }
    }
}
