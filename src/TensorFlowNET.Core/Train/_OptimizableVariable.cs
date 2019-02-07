using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public interface _OptimizableVariable
    {
        Tensor target();
        Operation update_op(Optimizer optimizer, Tensor g);
    }
}
