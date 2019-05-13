using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using NumSharp;

namespace Makina.Layers
{
    public interface ILayer
    {
        TensorShape __shape__();
        ILayer __build__(TensorShape input_shape, int seed = 1, float stddev = -1f);
        Tensor __call__(Tensor x);
        TensorShape output_shape(TensorShape input_shape);
    }
}
