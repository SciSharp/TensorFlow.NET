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

using System;
using System.Linq;
using static Tensorflow.Binding;

namespace Tensorflow.Operations.Initializers;
using System.Collections.Generic;

public class Orthogonal : IInitializer
{
    float _gain = 0f;
    int? _seed;

    public Orthogonal(float gain = 1.0f, int? seed = null)
    {
        _gain = gain;
        _seed = seed;
    }

    private readonly Dictionary<string, object> _config;

    public string ClassName => "Orthogonal";
    public IDictionary<string, object> Config => throw new NotImplementedException();
    public Tensor Apply(InitializerArgs args)
    {
        return _generate_init_val(args.Shape, args.DType == TF_DataType.DtInvalid ? TF_DataType.TF_FLOAT : args.DType);
    }

    private Tensor _generate_init_val(Shape shape, TF_DataType dtype)
    {
        var num_rows = 1L;
        foreach (var dim in shape.dims.Take(shape.ndim - 1))
            num_rows *= dim;
        var num_cols = shape.dims.Last();
        var flat_shape = (Math.Max(num_cols, num_rows), Math.Min(num_cols, num_rows));

        var a = tf.random.stateless_normal(flat_shape, dtype: dtype);
        // Compute the qr factorization
        var (q, r) = tf.linalg.qr(a, full_matrices: false);
        // Make Q uniform
        var d = tf.linalg.tensor_diag_part(r.Single);
        q *= tf.sign(d);

        if (num_rows < num_cols)
        {
            q = array_ops.matrix_transpose(q);
        }

        return _gain * tf.reshape(q, shape);
    }
}
