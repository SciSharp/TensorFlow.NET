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

using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Layers
{
    public class Embedding : Layer
    {
        private int input_dim;
        private int output_dim;
        private bool mask_zero;
        public IVariableV1 embeddings;
        public IInitializer embeddings_initializer;
        int input_length;

        public Embedding(int input_dim, int output_dim,
            IInitializer embeddings_initializer = null,
            bool mask_zero = false,
            TF_DataType dtype = TF_DataType.TF_FLOAT,
            int[] input_shape = null,
            int input_length = -1) :
            base(new LayerArgs
            {
                DType = dtype,
                InputShape = input_shape ?? new[] { input_length }
            })
        {
            this.input_dim = input_dim;
            this.output_dim = output_dim;
            this.embeddings_initializer = embeddings_initializer == null ? tf.uniform_initializer : embeddings_initializer;
            this.mask_zero = mask_zero;
            SupportsMasking = mask_zero;
            this.input_length = input_length;
        }

        protected override void build(TensorShape input_shape)
        {
            embeddings = add_weight(shape: new int[] { input_dim, output_dim },
                initializer: embeddings_initializer,
                name: "embeddings");
            built = true;
        }

        protected override Tensor[] call(Tensor[] inputs, bool is_training = false, Tensor state = null)
        {
            var dtype = inputs[0].dtype;
            if (dtype != tf.int32 && dtype != tf.int64)
                inputs[0] = math_ops.cast(inputs[0], tf.int32);

            var @out = embedding_ops.embedding_lookup(embeddings, inputs[0]);
            return new[] { @out, @out };
        }
    }
}
