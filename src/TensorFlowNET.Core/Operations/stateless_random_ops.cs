/*****************************************************************************
   Copyright 2023 Haiping Chen. All Rights Reserved.

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

using static Tensorflow.ApiDef.Types;
using System.Reflection;
using static Tensorflow.Binding;
using System;

namespace Tensorflow;

public class stateless_random_ops
{
    public static Tensor stateless_random_normal(Shape shape,
        float mean = 0.0f,
        float stddev = 1.0f,
        TF_DataType dtype = TF_DataType.TF_FLOAT,
        int[]? seed = null,
        string name = null)
    {
        return tf_with(ops.name_scope(name, "stateless_random_normal", new { shape, seed, mean, stddev }), scope =>
        {
            name = scope;
            var shape_tensor = _ShapeTensor(shape);
            var mean_tensor = ops.convert_to_tensor(mean, dtype: dtype, name: "mean");
            var stddev_tensor = ops.convert_to_tensor(stddev, dtype: dtype, name: "stddev");

            if (seed == null)
            {
                seed = new[] { new Random().Next(), 0 };
            }
            var (key, counter) = _get_key_counter(seed, 3);
            var rnd = gen_random_ops.stateless_random_normal_v2(shape: shape_tensor, key: key, counter: counter, dtype: dtype, alg: 3);
            var value = math_ops.add(rnd * stddev, mean_tensor, name: name);
            // tensor_util.maybe_set_static_shape(value, shape)
            return value;
        });
    }

    private static Tensor _ShapeTensor(int[] shape)
    {
        return ops.convert_to_tensor(shape, name: "shape");
    }

    private static (Tensor, Tensor) _get_key_counter(int[] seed, int alg)
    {
        var results = gen_random_ops.stateless_random_get_key_counter(seed);
        return (results[0], results[1]);
    }
}
