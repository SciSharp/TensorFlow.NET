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
using Tensorflow.Keras.Utils;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Layers
{
    public class BatchNormalization : Layer
    {
        BatchNormalizationArgs args;

        float momentum => args.Momentum;
        float epsilon => args.Epsilon;
        bool center => args.Center;
        bool scale => args.Scale;
        bool renorm => args.Renorm;
        bool fused;
        int[] axis;
        string _data_format;
        IInitializer beta_initializer => args.BetaInitializer;
        IInitializer gamma_initializer => args.GammaInitializer;
        IInitializer moving_mean_initializer => args.MovingMeanInitializer;
        IInitializer moving_variance_initializer => args.MovingVarianceInitializer;
        IRegularizer gamma_regularizer => args.GammaRegularizer;
        IVariableV1 gamma;
        IVariableV1 beta;
        IVariableV1 moving_mean;
        IVariableV1 moving_variance;

        public BatchNormalization(BatchNormalizationArgs args) : base(args)
        {
            this.args = args;
            axis = args.Axis.dims;
        }

        protected override void build(Tensors inputs)
        {
            TensorShape input_shape = inputs.shape;
            var ndims = input_shape.ndim;
            foreach (var (idx, x) in enumerate(axis))
                if (x < 0)
                    axis[idx] = ndims + x;

            fused = ndims == 4;

            if (fused)
            {
                if (Enumerable.SequenceEqual(axis, new int[] { 1 }))
                    _data_format = "NCHW";
                else if (Enumerable.SequenceEqual(axis, new int[] { 3 }))
                    _data_format = "NHWC";
                else
                    throw new ValueError($"Unsupported axis, fused batch norm only supports axis == [1] or axis == [3]");
            }

            var axis_to_dim = new Dictionary<int, int>();
            foreach (var x in axis)
                axis_to_dim[x] = input_shape[x];

            inputSpec = new InputSpec(ndim: ndims, axes: axis_to_dim);
            var param_dtype = DType == TF_DataType.DtInvalid ? TF_DataType.TF_FLOAT : DType;
            var param_shape = inputSpec.AllAxisDim;

            if (scale)
                gamma = add_weight("gamma",
                    param_shape,
                    dtype: param_dtype,
                    initializer: gamma_initializer,
                    trainable: true);
            else
                throw new NotImplementedException("add_weight gamma");

            if (center)
                beta = add_weight("beta",
                    param_shape,
                    dtype: param_dtype,
                    initializer: beta_initializer,
                    trainable: true);
            else
                throw new NotImplementedException("add_weight beta");

            moving_mean = add_weight("moving_mean",
                param_shape,
                dtype: param_dtype,
                initializer: moving_mean_initializer,
                synchronization: VariableSynchronization.OnRead,
                aggregation: VariableAggregation.Mean,
                trainable: false);

            moving_variance = add_weight("moving_variance",
                shape: param_shape,
                dtype: param_dtype,
                initializer: moving_variance_initializer,
                synchronization: VariableSynchronization.OnRead,
                aggregation: VariableAggregation.Mean,
                trainable: false);

            if (renorm)
                throw new NotImplementedException("build when renorm is true");

            built = true;
        }

        protected override Tensors Call(Tensors inputs, Tensor state = null, bool training = false)
        {
            Tensor outputs = null;
            var training_tensor = tf.logical_and(training, Trainable);
            if (fused)
            {
                // var training = tf.convert_to_tensor(training);
                outputs = _fused_batch_norm(inputs, training: training_tensor);
                return outputs;
            }

            throw new NotImplementedException("BatchNormalization call");
        }

        private Tensor _fused_batch_norm(Tensor inputs, Tensor training)
        {
            TensorShape input_batch_size = null;
            var use_fused_avg_updates = true;
            float exponential_avg_factor = 0;
            if (use_fused_avg_updates)
                exponential_avg_factor = 1.0f - momentum;

            Func<Tensor[]> _fused_batch_norm_training = () =>
            {
                return tf.nn.fused_batch_norm(
                  inputs,
                  gamma,
                  beta,
                  mean: moving_mean,
                  variance: moving_variance,
                  epsilon: epsilon, is_training: true,
                  data_format: _data_format,
                  exponential_avg_factor: exponential_avg_factor);
            };

            Func<Tensor[]> _fused_batch_norm_inference = () =>
            {
                return tf.nn.fused_batch_norm(
                  inputs,
                  gamma,
                  beta,
                  mean: moving_mean,
                  variance: moving_variance,
                  epsilon: epsilon,
                  is_training: false,
                  data_format: _data_format);
            };

            if (use_fused_avg_updates && input_batch_size != null)
                throw new NotImplementedException("");

            var results = tf_utils.smart_cond(training, _fused_batch_norm_training, _fused_batch_norm_inference);
            var (output, mean, variance) = (results[0], results[1], results[2]);
            var training_value = tf_utils.constant_value(training);

            if (!training_value.HasValue || (training_value.HasValue && training_value.Value))
            {
                Tensor momentum_tensor = null;
                if (!use_fused_avg_updates)
                {
                    if (training_value == null)
                        momentum_tensor = tf_utils.smart_cond(training,
                            () => new float[] { momentum }, 
                            () => new float[] { 1.0f })[0];
                    else
                        momentum_tensor = ops.convert_to_tensor(momentum);
                }

                if (use_fused_avg_updates)
                    _assign_new_value(moving_mean, mean);
                else
                    _assign_moving_average(moving_variance, variance, momentum_tensor);

                if (use_fused_avg_updates)
                    _assign_new_value(moving_variance, variance);
                else
                    _assign_moving_average(moving_variance, variance, momentum_tensor);

                // var mean_update = _assign_moving_average(moving_mean.AsTensor(), mean, momentum_tensor);
                // var variance_update = _assign_moving_average(moving_variance.AsTensor(), variance, momentum_tensor);
                // add_update(new Tensor[] { mean_update }, inputs: true);
                // add_update(new Tensor[] { variance_update }, inputs: true);
            }

            return output;
        }

        void _assign_new_value(IVariableV1 variable, Tensor value)
        {
            tf_with(ops.name_scope("AssignNewValue", null, new { variable, value, momentum }), scope =>
            {
                // var cm = ops.colocate_with(variable);
                variable.assign_lazy_load(value, name: scope);
            });
        }

        void _assign_moving_average(IVariableV1 variable, Tensor value, Tensor momentum)
        {
            tf_with(ops.name_scope("AssignMovingAvg", null, new { variable, value, momentum }), scope =>
            {
                // var cm = ops.colocate_with(variable);
                var decay = ops.convert_to_tensor(1.0f - momentum, name: "decay");
                var update_delta = (variable.AsTensor() - math_ops.cast(value, variable.dtype)) * decay;
                variable.assign_sub_lazy_load(update_delta, name: scope);
            });
        }
    }
}
