﻿/*****************************************************************************
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
    public class gen_random_ops
    {
        /// <summary>
        /// Outputs random values from a normal distribution.
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="dtype"></param>
        /// <param name="seed"></param>
        /// <param name="seed2"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor random_standard_normal(Tensor shape, TF_DataType dtype = TF_DataType.DtInvalid, int? seed = null, int? seed2 = null, string name = null)
            => tf.Context.ExecuteOp("RandomStandardNormal", name, new ExecuteOpArgs(shape)
                .SetAttributes(new { dtype, seed = seed ?? 0, seed2 = seed2 ?? 0 }));

        /// <summary>
        /// Outputs random integers from a uniform distribution.
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="minval"></param>
        /// <param name="maxval"></param>
        /// <param name="seed"></param>
        /// <param name="seed2"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor random_uniform_int(Tensor shape, Tensor minval, Tensor maxval, int? seed = 0, int? seed2 = 0, string name = null)
         => tf.Context.ExecuteOp("RandomUniformInt", name, new ExecuteOpArgs(shape, minval, maxval)
                .SetAttributes(new { seed = seed ?? 0, seed2 = seed2 ?? 0 }));

        /// <summary>
        /// Outputs random values from a uniform distribution.
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="dtype"></param>
        /// <param name="seed"></param>
        /// <param name="seed2"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor random_uniform(Tensor shape, TF_DataType dtype, int? seed = 0, int? seed2 = 0, string name = null)
            => tf.Context.ExecuteOp("RandomUniform", name, new ExecuteOpArgs(shape)
                .SetAttributes(new { dtype, seed = seed ?? 0, seed2 = seed2 ?? 0 }));

        /// <summary>
        /// 
        /// </summary>
        /// <param name="value"></param>
        /// <param name="seed"></param>
        /// <param name="seed2"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor random_shuffle(Tensor value, int? seed = 0, int? seed2 = 0,
            string name = null)
               => tf.Context.ExecuteOp("RandomShuffle", name, new ExecuteOpArgs(value)
                   .SetAttributes(new { seed = seed ?? 0, seed2 = seed2 ?? 0 }));

        /// <summary>
        /// Outputs random values from a truncated normal distribution.
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="dtype"></param>
        /// <param name="seed"></param>
        /// <param name="seed2"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor truncated_normal(Tensor shape, TF_DataType dtype, int? seed = 0,
            int? seed2 = 0, string name = null)
                => tf.Context.ExecuteOp("TruncatedNormal", name, new ExecuteOpArgs(shape)
                    .SetAttributes(new { dtype, seed = seed ?? 0, seed2 = seed2 ?? 0 }));

        public static Tensor multinomial(Tensor logits, int num_samples, int? seed = 0,
            int? seed2 = 0, TF_DataType output_dtype = TF_DataType.TF_INT64, string name = null)
        {
            if (!seed.HasValue)
                seed = 0;
            if (!seed2.HasValue)
                seed2 = 0;
            if (output_dtype == TF_DataType.DtInvalid)
                output_dtype = TF_DataType.TF_INT64;

            var _op = tf.OpDefLib._apply_op_helper("Multinomial",
                name: name,
                args: new { logits, num_samples, seed, seed2, output_dtype });

            return _op.output;
        }
    }
}
