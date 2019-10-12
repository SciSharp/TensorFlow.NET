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

using System.Linq;
using Tensorflow.Keras.Layers;

namespace Tensorflow
{
    public static partial class keras
    {
        public static class layers
        {
            public static Embedding Embedding(int input_dim, int output_dim,
                IInitializer embeddings_initializer = null,
                bool mask_zero = false) => new Embedding(input_dim, output_dim,
                    embeddings_initializer,
                    mask_zero);

            public static Tensor[] Input(int[] batch_shape = null,
                TF_DataType dtype = TF_DataType.DtInvalid,
                string name = null,
                bool sparse = false,
                Tensor tensor = null)
            {
                var batch_size = batch_shape[0];
                var shape = batch_shape.Skip(1).ToArray();

                InputLayer input_layer = null;
                if (batch_shape != null)
                    input_layer = new InputLayer(
                        batch_input_shape: batch_shape,
                        name: name,
                        dtype: dtype,
                        sparse: sparse,
                        input_tensor: tensor);
                else
                    input_layer = new InputLayer(
                        input_shape: shape,
                        batch_size: batch_size,
                        name: name,
                        dtype: dtype,
                        sparse: sparse,
                        input_tensor: tensor);

                var outputs = input_layer.inbound_nodes[0].output_tensors;

                return outputs;
            }
        }
    }
}
