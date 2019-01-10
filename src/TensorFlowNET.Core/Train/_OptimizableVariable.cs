using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public interface _OptimizableVariable
    {
        Tensor target();
        void update_op(Graph g);
    }
}
