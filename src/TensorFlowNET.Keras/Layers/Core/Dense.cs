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

using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Layers
{
    /// <summary>
    /// Just your regular densely-connected NN layer.
    /// </summary>
    public class Dense : Layer
    {
        DenseArgs args;
        IVariableV1 kernel;
        IVariableV1 bias;
        Activation activation => args.Activation;

        public Dense(DenseArgs args) :
            base(args)
        {
            this.args = args;
            this.SupportsMasking = true;
            this.inputSpec = new InputSpec(min_ndim: 2);
        }

        protected override void build(Tensors inputs)
        {
            TensorShape input_shape = inputs.shape;
            var last_dim = input_shape.dims.Last();
            var axes = new Dictionary<int, int>();
            axes[-1] = last_dim;
            inputSpec = new InputSpec(min_ndim: 2, axes: axes);
            kernel = add_weight(
                "kernel",
                shape: new TensorShape(last_dim, args.Units),
                initializer: args.KernelInitializer,
                dtype: DType,
                trainable: true);
            if (args.UseBias)
                bias = add_weight(
                  "bias",
                  shape: new TensorShape(args.Units),
                  initializer: args.BiasInitializer,
                  dtype: DType,
                  trainable: true);

            built = true;
        }

        protected override Tensors Call(Tensors inputs, Tensor state = null, bool? training = null)
        {
            Tensor outputs = null;
            var rank = inputs.rank;
            if (rank > 2)
            {
                throw new NotImplementedException("call rank > 2");
            }
            else
            {
                outputs = gen_math_ops.mat_mul(inputs, kernel.AsTensor());
            }

            if (args.UseBias)
                outputs = tf.nn.bias_add(outputs, bias);
            if (args.Activation != null)
                outputs = activation(outputs);

            return outputs;
        }

        public static Dense from_config(LayerArgs args)
        {
            return new Dense(args as DenseArgs);
        }
    }
}
