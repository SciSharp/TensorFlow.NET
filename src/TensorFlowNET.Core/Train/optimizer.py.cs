using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class optimizer
    {
        public static _OptimizableVariable _get_processor(RefVariable v)
        {
            return new _RefVariableProcessor(v);
        }
    }

    public class _RefVariableProcessor : _OptimizableVariable
    {
        private RefVariable _v;

        public _RefVariableProcessor(RefVariable v)
        {
            _v = v;
        }

        public Tensor target()
        {
            return _v._ref();
        }

        public void update_op(Graph g)
        {
            
        }
    }
}
