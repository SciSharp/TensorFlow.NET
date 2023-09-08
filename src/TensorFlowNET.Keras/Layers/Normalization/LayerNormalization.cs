/*****************************************************************************
   Copyright 2021 Haiping Chen. All Rights Reserved.

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
using Tensorflow.Common.Types;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Saving;
using Tensorflow.Keras.Utils;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Layers
{
    public class LayerNormalization : Layer
    {
        LayerNormalizationArgs args;

        float epsilon => args.Epsilon;
        bool center => args.Center;
        bool scale => args.Scale;
        bool _fused;
        int[] axis;
        string _data_format;
        Shape kernel_size;
        IInitializer beta_initializer => args.BetaInitializer;
        IInitializer gamma_initializer => args.GammaInitializer;
        IRegularizer gamma_regularizer => args.GammaRegularizer;
        IVariableV1 gamma;
        IVariableV1 beta;
        IVariableV1 moving_mean;
        IVariableV1 moving_variance;

        public LayerNormalization(LayerNormalizationArgs args) : base(args)
        {
            this.args = args;
            axis = args.Axis.axis;
        }

        public override void build(KerasShapesWrapper input_shape)
        {
            var single_shape = input_shape.ToSingleShape();
            var ndims = single_shape.ndim;
            foreach (var (idx, x) in enumerate(axis))
                if (x < 0)
                    axis[idx] = ndims + x;

            var axis_to_dim = new Dictionary<int, int>();
            foreach (var x in axis)
                axis_to_dim[x] = (int)single_shape[x];

            inputSpec = new InputSpec(ndim: ndims, axes: axis_to_dim);
            var param_dtype = DType == TF_DataType.DtInvalid ? TF_DataType.TF_FLOAT : DType;
            var param_shape = inputSpec.AllAxisDim;

            if (scale)
                gamma = add_weight("gamma",
                    param_shape,
                    dtype: param_dtype,
                    initializer: gamma_initializer,
                    trainable: true);

            if (center)
                beta = add_weight("beta",
                    param_shape,
                    dtype: param_dtype,
                    initializer: beta_initializer,
                    trainable: true);

            _fused = _fused_can_be_used(ndims);

            built = true;
            _buildInputShape = input_shape;
        }

        bool _fused_can_be_used(int ndims)
        {
            var can_use_fused = false;
            if (axis.Last() == ndims - 1 && axis.Last() - axis[0] == len(axis) - 1)
                can_use_fused = true;
            if (epsilon < 1.001e-5 || DType != tf.float32)
                can_use_fused = false;
            return can_use_fused;
        }

        public override Shape ComputeOutputShape(Shape input_shape)
        {
            return input_shape;
        }

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            Tensor outputs = null;
            var inputs_dtype = inputs.dtype.as_base_dtype();
            var input_shape = inputs.shape;
            var ndims = len(input_shape);
            var broadcast_shape = range(ndims).Select(x => 1).ToArray();
            foreach (var dim in axis)
                broadcast_shape[dim] = input_shape.as_int_list()[dim];

            Func<IVariableV1, Tensor> _broadcast = v =>
            {
                if (v.shape.ndim != ndims && !axis.SequenceEqual(new int[] { ndims - 1 }))
                    return tf.reshape(v.AsTensor(), broadcast_shape);
                return v.AsTensor();
            };

            if (_fused)
            {
                var tensor_shape = tf.shape(inputs);
                var pre_dim = tf.constant(1);
                var in_dim = tf.constant(1);
                foreach (var dim in range(ndims))
                {
                    var dim_tensor = tensor_shape[dim];
                    if (dim < axis[0])
                        pre_dim = pre_dim * dim_tensor;
                    else
                        in_dim = in_dim * dim_tensor;
                }
                inputs = tf.reshape(inputs, new object[] { 1, pre_dim, in_dim, 1 });

                var scale = tf.ones(new Shape((int)pre_dim), dtype: DType);
                var offset = tf.zeros(new Shape((int)pre_dim), dtype: DType);

                outputs = tf.nn.fused_batch_norm(
                  inputs,
                  scale: scale,
                  offset: offset,
                  epsilon: epsilon,
                  data_format: "NCHW")[0];

                outputs = tf.reshape(outputs, tensor_shape);

                (scale, offset) = (_broadcast(gamma), _broadcast(beta));

                outputs = outputs * tf.cast(scale, outputs.dtype);
                outputs = outputs + tf.cast(offset, outputs.dtype);
            }
            else
            {
                var input_dtype = inputs.dtype;
                if ((input_dtype == tf.float16) && DType == tf.float32) inputs = tf.cast(inputs, tf.float32);
                (Tensor mean, Tensor variance) = tf.nn.moments(inputs, axis, keep_dims: true);

                (Tensor scale, Tensor offset) = (_broadcast(gamma), _broadcast(beta));

                outputs = tf.nn.batch_normalization(
                inputs,
                mean,
                variance,
                offset: offset,
                scale: scale,
                variance_epsilon: epsilon);

                outputs = tf.cast(outputs, input_dtype);
            }
            // If some components of the shape got lost due to adjustments, fix that.
            outputs.shape = input_shape;

            return outputs;
        }
    }
}
