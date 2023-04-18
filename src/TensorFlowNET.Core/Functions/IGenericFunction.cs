using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Functions
{
    public interface IGenericFunction
    {
        Tensors Apply(Tensors args);
        ConcreteFunction get_concrete_function(params Tensor[] args);
    }
}
