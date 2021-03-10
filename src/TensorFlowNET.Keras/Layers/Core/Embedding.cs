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
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Layers
{
    /// <summary>
    /// Turns positive integers (indexes) into dense vectors of fixed size.
    /// https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
    /// </summary>
    public class Embedding : Layer
    {
        EmbeddingArgs args;
        int input_dim => args.InputDim;
        int output_dim => args.OutputDim;
        bool mask_zero => args.MaskZero;
        IVariableV1 embeddings;
        IInitializer embeddings_initializer;

        public Embedding(EmbeddingArgs args)
            : base(new LayerArgs // copy args
            {
                DType = args.DType,
                Name = args.Name
            })
        {
            this.args = args;
            if (args.InputShape == null)
                args.InputShape = args.InputLength;

            if (args.BatchInputShape == null)
                args.BatchInputShape = new int[] { args.BatchSize }.Concat(args.InputShape.dims).ToArray();

            embeddings_initializer = args.EmbeddingsInitializer ?? tf.random_uniform_initializer;
            SupportsMasking = mask_zero;
        }

        protected override void build(Tensors inputs)
        {
            tf.Context.eager_mode();
            embeddings = add_weight(shape: (input_dim, output_dim),
                initializer: embeddings_initializer,
                name: "embeddings");
            tf.Context.graph_mode();
            built = true;
        }

        protected override Tensors Call(Tensors inputs, Tensor state = null, bool? training = null)
        {
            var dtype = inputs.dtype;
            if (dtype != tf.int32 && dtype != tf.int64)
                inputs = math_ops.cast(inputs, tf.int32);

            var outputs = embedding_ops.embedding_lookup(embeddings, inputs);
            return outputs;
        }
    }
}
