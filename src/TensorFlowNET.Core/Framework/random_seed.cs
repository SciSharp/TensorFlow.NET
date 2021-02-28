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

using System.Collections.Generic;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class random_seed
    {
        private static int DEFAULT_GRAPH_SEED = 87654321;
        private static Dictionary<string, int> _graph_to_seed_dict = new Dictionary<string, int>();

        public static (int?, int?) get_seed(int? op_seed = null)
        {
            int? global_seed;

            if (tf.executing_eagerly())
                global_seed = tf.Context.global_seed();
            else
                global_seed = ops.get_default_graph().seed;

            if (global_seed.HasValue)
            {
                if (!op_seed.HasValue)
                    if (tf.executing_eagerly())
                        op_seed = tf.Context.internal_operation_seed();
                    else
                    {
                        if (!_graph_to_seed_dict.TryGetValue(ops.get_default_graph().graph_key, out int seed))
                            seed = 0;
                        _graph_to_seed_dict[ops.get_default_graph().graph_key] = seed + 1;
                        op_seed = seed;
                    }

                return (global_seed, op_seed);
            }

            if (op_seed.HasValue)
                return (DEFAULT_GRAPH_SEED, op_seed);
            else
                return (null, null);
        }

        public static (Tensor, Tensor) get_seed_tensor(int? op_seed = null)
        {
            var (seed, seed2) = get_seed(op_seed);
            Tensor _seed, _seed2;
            if (seed is null)
                _seed = constant_op.constant(0, dtype: TF_DataType.TF_INT64, name: "seed");
            else
                _seed = constant_op.constant(seed.Value, dtype: TF_DataType.TF_INT64, name: "seed");

            if (seed2 is null)
                _seed2 = constant_op.constant(0, dtype: TF_DataType.TF_INT64, name: "seed2");
            else
            {
                _seed2 = tf_with(ops.name_scope("seed2"), scope =>
                {
                    _seed2 = constant_op.constant(seed2.Value, dtype: TF_DataType.TF_INT64);
                    return array_ops.where_v2(
                      math_ops.logical_and(
                          math_ops.equal(_seed, 0l), math_ops.equal(_seed2, 0l)),
                      constant_op.constant(2^31 - 1, dtype: dtypes.int64),
                      _seed2,
                      name: scope);
                });
            }
                
            return (_seed, _seed2);
        }
    }
}
