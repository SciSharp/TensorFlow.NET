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

namespace Tensorflow.Keras.Layers
{
    public class Embedding : Layer
    {
        private int input_dim;
        private int output_dim;
        private bool mask_zero;
        public RefVariable embeddings;
        public IInitializer embeddings_initializer;

        public Embedding(int input_dim, int output_dim,
            IInitializer embeddings_initializer = null,
            bool mask_zero = false,
            TF_DataType dtype = TF_DataType.TF_FLOAT,
            int[] input_shape = null) : base(dtype: dtype, input_shape: input_shape)
        {
            this.input_dim = input_dim;
            this.output_dim = output_dim;
            this.embeddings_initializer = embeddings_initializer == null ? tf.uniform_initializer : embeddings_initializer;
            this.mask_zero = mask_zero;
            supports_masking = mask_zero;
        }

        protected override void build(TensorShape input_shape)
        {
            embeddings = add_weight(shape: new int[] { input_dim, output_dim },
                initializer: embeddings_initializer,
                name: "embeddings");
            built = true;
        }
    }
}
