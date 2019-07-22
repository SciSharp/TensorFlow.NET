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

namespace Tensorflow
{
    public class gen_random_ops
    {
        public static OpDefLibrary _op_def_lib = new OpDefLibrary();

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
        {
            if (!seed.HasValue)
                seed = 0;
            if (!seed2.HasValue)
                seed2 = 0;

            var _op = _op_def_lib._apply_op_helper("RandomStandardNormal", 
                name: name,
                args: new { shape, dtype, seed, seed2 });

            return _op.outputs[0];
        }

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
        {
            if (!seed.HasValue)
                seed = 0;
            if (!seed2.HasValue)
                seed2 = 0;

            var _op = _op_def_lib._apply_op_helper("RandomUniformInt",
                name: name,
                args: new { shape, minval, maxval, seed, seed2 });

            return _op.outputs[0];
        }

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
        {
            if (!seed.HasValue)
                seed = 0;
            if (!seed2.HasValue)
                seed2 = 0;

            var _op = _op_def_lib._apply_op_helper("RandomUniform",
                name: name,
                args: new { shape, dtype, seed, seed2});

            return _op.outputs[0];
        }

        /// <summary>
        /// Outputs random values from a truncated normal distribution.
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="dtype"></param>
        /// <param name="seed"></param>
        /// <param name="seed2"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor truncated_normal(Tensor shape, TF_DataType dtype, int? seed = 0, int? seed2 = 0, string name = null)
        {
            if (!seed.HasValue)
                seed = 0;
            if (!seed2.HasValue)
                seed2 = 0;

            var _op = _op_def_lib._apply_op_helper("TruncatedNormal",
                name: name,
                args: new { shape, dtype, seed, seed2 });

            return _op.outputs[0];
        }
    }
}
