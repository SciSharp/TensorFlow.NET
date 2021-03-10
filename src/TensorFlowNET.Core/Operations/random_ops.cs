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
    public class random_ops
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="mean"></param>
        /// <param name="stddev"></param>
        /// <param name="dtype"></param>
        /// <param name="seed"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor random_normal(TensorShape shape,
            float mean = 0.0f,
            float stddev = 1.0f,
            TF_DataType dtype = TF_DataType.TF_FLOAT,
            int? seed = null,
            string name = null)
        {
            return tf_with(ops.name_scope(name, "random_normal", new { shape, mean, stddev }), scope =>
            {
                name = scope;
                var shape_tensor = _ShapeTensor(shape);
                var mean_tensor = ops.convert_to_tensor(mean, dtype: dtype, name: "mean");
                var stddev_tensor = ops.convert_to_tensor(stddev, dtype: dtype, name: "stddev");
                var (seed1, seed2) = random_seed.get_seed(seed);
                var rnd = gen_random_ops.random_standard_normal(shape_tensor, dtype: dtype, seed: seed1, seed2: seed2);
                var mul = rnd * stddev_tensor;
                var value = math_ops.add(mul, mean_tensor, name: name);
                // tensor_util.maybe_set_static_shape(value, shape)
                return value;
            });
        }

        /// <summary>
        /// Outputs random values from a uniform distribution.
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="minval"></param>
        /// <param name="maxval"></param>
        /// <param name="dtype">The type of the output</param>
        /// <param name="seed">Used to create a random seed for the distribution.</param>
        /// <param name="name">A name for the operation</param>
        /// <returns>A tensor of the specified shape filled with random uniform values.</returns>
        public static Tensor random_uniform(int[] shape,
            float minval = 0,
            float maxval = 1,
            TF_DataType dtype = TF_DataType.TF_FLOAT,
            int? seed = null,
            string name = null)
        {
            return tf_with(ops.name_scope(name, "random_uniform", new { shape, minval, maxval }), scope =>
            {
                name = scope;
                var (seed1, seed2) = random_seed.get_seed(seed);
                var tensorShape = tensor_util.shape_tensor(shape);
                var minTensor = ops.convert_to_tensor(minval, dtype: dtype, name: "min");
                var maxTensor = ops.convert_to_tensor(maxval, dtype: dtype, name: "max");
                var rnd = gen_random_ops.random_uniform(tensorShape, dtype, seed: seed1, seed2: seed2);
                return math_ops.add(rnd * (maxTensor - minTensor), minTensor, name: name);
            });
        }

        /// <summary>
        /// Outputs random values from a uniform distribution.
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="minval"></param>
        /// <param name="maxval"></param>
        /// <param name="dtype">The type of the output</param>
        /// <param name="seed">Used to create a random seed for the distribution.</param>
        /// <param name="name">A name for the operation</param>
        /// <returns>A tensor of the specified shape filled with random uniform values.</returns>
        public static Tensor random_uniform_int(int[] shape,
            int minval = 0,
            int maxval = 1,
            TF_DataType dtype = TF_DataType.TF_FLOAT,
            int? seed = null,
            string name = null)
        {
            return tf_with(ops.name_scope(name, "random_uniform_int", new { shape, minval, maxval }), scope =>
            {
                name = scope;
                var (seed1, seed2) = random_seed.get_seed(seed);
                var tensorShape = tensor_util.shape_tensor(shape);
                var minTensor = ops.convert_to_tensor(minval, dtype: dtype, name: "min");
                var maxTensor = ops.convert_to_tensor(maxval, dtype: dtype, name: "max");
                return gen_random_ops.random_uniform_int(tensorShape, minTensor, maxTensor, seed: seed1, seed2: seed2);
            });
        }

        public static Tensor random_uniform(Tensor shape,
            int minval = 0,
            Tensor maxval = null,
            TF_DataType dtype = TF_DataType.TF_FLOAT,
            int? seed = null,
            string name = null)
        {
            return tf_with(ops.name_scope(name, "random_uniform", new { shape, minval, maxval }), scope =>
            {
                name = scope;
                var minTensor = ops.convert_to_tensor(minval, dtype: dtype, name: "min");
                var maxTensor = ops.convert_to_tensor(maxval == null ? 1 : (int)maxval, dtype: dtype, name: "max");
                var (seed1, seed2) = random_seed.get_seed(seed);
                if (dtype.is_integer())
                {
                    return gen_random_ops.random_uniform_int(shape, minTensor, maxTensor, seed: seed1, seed2: seed2, name: name);
                }
                else
                {
                    var rnd = gen_random_ops.random_uniform(shape, dtype);
                    return math_ops.add(rnd * (maxTensor - minTensor), minTensor, name: name);
                }
            });
        }

        /// <summary>
        /// Randomly shuffles a tensor along its first dimension.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="seed"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor random_shuffle(Tensor value, int? seed = null, string name = null)
        {
            var (seed1, seed2) = random_seed.get_seed(seed);
            return gen_random_ops.random_shuffle(value, seed: seed1, seed2: seed2, name: name);
        }

        public static Tensor truncated_normal(int[] shape,
            float mean = 0.0f,
            float stddev = 1.0f,
            TF_DataType dtype = TF_DataType.TF_FLOAT,
            int? seed = null,
            string name = null)
        {
            return tf_with(ops.name_scope(name, "truncated_normal", new { shape, mean, stddev }), scope =>
            {
                name = scope;
                var shape_tensor = _ShapeTensor(shape);
                var mean_tensor = ops.convert_to_tensor(mean, dtype: dtype, name: "mean");
                var stddev_tensor = ops.convert_to_tensor(stddev, dtype: dtype, name: "stddev");
                var (seed1, seed2) = random_seed.get_seed(seed);
                var rnd = gen_random_ops.truncated_normal(shape_tensor, dtype, seed: seed1, seed2: seed2);
                var mul = rnd * stddev_tensor;
                var value = math_ops.add(mul, mean_tensor, name: name);
                return value;
            });
        }

        private static Tensor _ShapeTensor(int[] shape)
        {
            return ops.convert_to_tensor(shape, name: "shape");
        }

        public static Tensor multinomial(Tensor logits, int num_samples, int? seed = null,
            string name = null, TF_DataType output_dtype = TF_DataType.DtInvalid)
        {
            return tf_with(ops.name_scope(name, "multinomial", new { logits }), delegate
            {
                return multinomial_categorical_impl(logits, num_samples, output_dtype, seed);
            });
        }

        /// <summary>
        /// Implementation for random.categorical (v1) and random.categorical (v2).
        /// </summary>
        /// <param name="logits"></param>
        /// <param name="num_samples"></param>
        /// <param name="dtype"></param>
        /// <param name="seed"></param>
        /// <returns></returns>
        private static Tensor multinomial_categorical_impl(Tensor logits, int num_samples, TF_DataType dtype = TF_DataType.DtInvalid,
            int? seed = null)
        {
            logits = ops.convert_to_tensor(logits, name: "logits");
            var (seed1, seed2) = random_seed.get_seed(seed);
            return gen_random_ops.multinomial(logits,
                num_samples,
                seed: seed1,
                seed2: seed2,
                output_dtype: dtype);
        }
    }
}

