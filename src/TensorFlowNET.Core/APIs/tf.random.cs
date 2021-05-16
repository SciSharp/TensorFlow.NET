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

namespace Tensorflow
{
    public partial class tensorflow
    {
        public Random random => new Random();

        public class Random
        {
            /// <summary>
            /// Outputs random values from a normal distribution.
            /// </summary>
            /// <param name="shape"></param>
            /// <param name="mean"></param>
            /// <param name="stddev"></param>
            /// <param name="dtype"></param>
            /// <param name="seed"></param>
            /// <param name="name"></param>
            /// <returns></returns>
            public Tensor normal(TensorShape shape,
                float mean = 0.0f,
                float stddev = 1.0f,
                TF_DataType dtype = TF_DataType.TF_FLOAT,
                int? seed = null,
                string name = null) => random_ops.random_normal(shape, mean, stddev, dtype, seed, name);

            /// <summary>
            /// Outputs random values from a truncated normal distribution.
            /// </summary>
            /// <param name="shape"></param>
            /// <param name="mean"></param>
            /// <param name="stddev"></param>
            /// <param name="dtype"></param>
            /// <param name="seed"></param>
            /// <param name="name"></param>
            /// <returns></returns>
            public Tensor truncated_normal(TensorShape shape,
                float mean = 0.0f,
                float stddev = 1.0f,
                TF_DataType dtype = TF_DataType.TF_FLOAT,
                int? seed = null,
                string name = null) => random_ops.truncated_normal(shape, mean, stddev, dtype, seed, name);

            public Tensor categorical(
                Tensor logits,
                int num_samples,
                int? seed = null,
                string name = null,
                TF_DataType output_dtype = TF_DataType.DtInvalid) => random_ops.multinomial(logits, num_samples, seed: seed, name: name, output_dtype: output_dtype);

            public Tensor uniform(TensorShape shape,
                float minval = 0,
                float maxval = 1,
                TF_DataType dtype = TF_DataType.TF_FLOAT,
                int? seed = null,
                string name = null)
            {
                if (dtype.is_integer())
                    return random_ops.random_uniform_int(shape, (int)minval, (int)maxval, dtype, seed, name);
                else
                    return random_ops.random_uniform(shape, minval, maxval, dtype, seed, name);
            }
        }

        public Tensor random_uniform(TensorShape shape,
            float minval = 0,
            float maxval = 1,
            TF_DataType dtype = TF_DataType.TF_FLOAT,
            int? seed = null,
            string name = null)
            => random.uniform(shape, minval: minval, maxval: maxval, dtype: dtype, seed: seed, name: name);

        public Tensor truncated_normal(TensorShape shape,
            float mean = 0.0f,
            float stddev = 1.0f,
            TF_DataType dtype = TF_DataType.TF_FLOAT,
            int? seed = null,
            string name = null)
            => random_ops.truncated_normal(shape, mean, stddev, dtype, seed, name);

        /// <summary>
        /// Randomly shuffles a tensor along its first dimension.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="seed"></param>
        /// <param name="name"></param>
        /// <returns>
        /// A tensor of same shape and type as value, shuffled along its 
        /// first dimension.
        /// </returns>
        public Tensor random_shuffle(Tensor value, int? seed = null, string name = null)
            => random_ops.random_shuffle(value, seed: seed, name: name);

        public void set_random_seed(int seed)
        {
            if (executing_eagerly())
                Context.set_global_seed(seed);
            else
                ops.get_default_graph().seed = seed;
        }

        public Tensor multinomial(Tensor logits, int num_samples, int? seed = null,
            string name = null, TF_DataType output_dtype = TF_DataType.DtInvalid)
            => random_ops.multinomial(logits, num_samples, seed: seed,
                name: name, output_dtype: output_dtype);
    }
}
