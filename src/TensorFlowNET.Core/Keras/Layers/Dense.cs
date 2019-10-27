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
using Tensorflow.Keras.Engine;
using Tensorflow.Operations.Activation;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Layers
{
    public class Dense : Tensorflow.Layers.Layer
    {
        protected int units;
        protected IActivation activation;
        protected bool use_bias;
        protected IInitializer kernel_initializer;
        protected IInitializer bias_initializer;
        protected RefVariable kernel;
        protected RefVariable bias;

        public Dense(int units,
            IActivation activation,
            bool use_bias = true,
            bool trainable = false,
            IInitializer kernel_initializer = null,
            IInitializer bias_initializer = null) : base(trainable: trainable)
        {
            this.units = units;
            this.activation = activation;
            this.use_bias = use_bias;
            this.kernel_initializer = kernel_initializer;
            this.bias_initializer = bias_initializer;
            this.supports_masking = true;
            this.input_spec = new InputSpec(min_ndim: 2);
        }

        protected override void build(TensorShape input_shape)
        {
            var last_dim = input_shape.dims.Last();
            var axes = new Dictionary<int, int>();
            axes[-1] = last_dim;
            input_spec = new InputSpec(min_ndim: 2, axes: axes);
            kernel = (RefVariable)add_weight(
                "kernel",
                shape: new int[] { last_dim, units },
                initializer: kernel_initializer,
                dtype: _dtype,
                trainable: true);
            if (use_bias)
                bias = (RefVariable)add_weight(
                  "bias",
                  shape: new int[] { units },
                  initializer: bias_initializer,
                  dtype: _dtype,
                  trainable: true);

            built = true;
        }

        protected override (Tensor, Tensor) call(Tensor inputs, Tensor training = null)
        {
            Tensor outputs = null;
            var rank = inputs.rank;
            if(rank > 2)
            {
                throw new NotImplementedException("call rank > 2");
            }
            else
            {
                outputs = gen_math_ops.mat_mul(inputs, kernel);
            }

            if (use_bias)
                outputs = tf.nn.bias_add(outputs, bias);
            if (activation != null)
                outputs = activation.Activate(outputs);

            return (outputs, outputs);
        }
    }
}
