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
using System.Diagnostics;
using System.Linq;
using Tensorflow.Common.Types;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Saving;
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

        public override void build(KerasShapesWrapper input_shape)
        {
            _buildInputShape = input_shape;
            Debug.Assert(input_shape.Shapes.Length <= 1);
            var single_shape = input_shape.ToSingleShape();
            var last_dim = single_shape.dims.Last();
            var axes = new Dictionary<int, int>();
            axes[-1] = (int)last_dim;
            inputSpec = new InputSpec(min_ndim: 2, axes: axes);
            kernel = add_weight(
                "kernel",
                shape: new Shape(last_dim, args.Units),
                initializer: args.KernelInitializer,
                dtype: DType,
                trainable: true);
            if (args.UseBias)
                bias = add_weight(
                  "bias",
                  shape: new Shape(args.Units),
                  initializer: args.BiasInitializer,
                  dtype: DType,
                  trainable: true);

            built = true;
        }

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            Tensor outputs = null;
            var rank = inputs.rank;
            if (rank > 2)
            {
                outputs = tf.linalg.tensordot(inputs, kernel.AsTensor(), new[,] { { rank - 1 }, { 0 } });
            }
            else
            {
                outputs = math_ops.matmul(inputs, kernel.AsTensor());
            }

            if (args.UseBias)
                outputs = tf.nn.bias_add(outputs, bias);
            if (args.Activation != null)
                outputs = activation.Apply(outputs);

            return outputs;
        }
    }
}
