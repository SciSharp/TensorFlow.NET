using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Functions
{
    public interface IGenericFunction
    {
        object[] Apply(params object[] args);
        ConcreteFunction get_concrete_function(params object[] args);
    }
}
