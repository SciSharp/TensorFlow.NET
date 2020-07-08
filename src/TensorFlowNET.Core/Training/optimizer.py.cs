/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using Tensorflow.Framework;

namespace Tensorflow
{
    public class optimizer
    {
        public static _OptimizableVariable _get_processor(RefVariable v)
        {
            return new _RefVariableProcessor(v);
        }

        public static _OptimizableVariable _get_processor(ResourceVariable v)
        {
            return new _DenseResourceVariableProcessor(v);
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
            Operation update_op = null;

            if (g.Tag == null)
            {
                update_op = optimizer._apply_dense(g, _v);
            }
            else if (g.Tag is IndexedSlices)
            {
                return optimizer._apply_sparse_duplicate_indices(g, _v);
            }

            return update_op;
        }
    }

    public class _DenseResourceVariableProcessor : _OptimizableVariable
    {
        private ResourceVariable _v;

        public _DenseResourceVariableProcessor(ResourceVariable v)
        {
            _v = v;
        }

        public Tensor target()
        {
            return _v.Handle;
        }

        public Operation update_op(Optimizer optimizer, Tensor g)
        {
            Operation update_op = null;

            if (g.Tag == null)
            {
                update_op = optimizer._apply_dense(g, _v);
            }
            else if (g.Tag is IndexedSlices)
            {
                return optimizer._apply_sparse_duplicate_indices(g, _v);
            }

            return update_op;
        }
    }
}
