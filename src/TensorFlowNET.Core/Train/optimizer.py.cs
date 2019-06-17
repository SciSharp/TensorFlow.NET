using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Framework;

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

        public Operation update_op(Optimizer optimizer, Tensor g)
        {
            var update_op = optimizer._apply_dense(g, _v);

            return update_op;
        }

        public Operation update_op(Optimizer optimizer, IndexedSlices g)
        {
            var update_op = optimizer._apply_dense(g, _v);

            return update_op;
        }
    }
}
