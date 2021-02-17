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

using static Tensorflow.Binding;

namespace Tensorflow
{
    public class random_seed
    {
        private static int DEFAULT_GRAPH_SEED = 87654321;

        public static (int?, int?) get_seed(int? op_seed = null)
        {
            int? global_seed;

            if (tf.executing_eagerly())
                global_seed = tf.Context.global_seed();
            else
                global_seed = ops.get_default_graph().seed;

            if (global_seed.HasValue)
                return (global_seed, op_seed);

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
                _seed2 = constant_op.constant(seed2.Value, dtype: TF_DataType.TF_INT64, name: "seed2");

            return (_seed, _seed2);
        }
    }
}
