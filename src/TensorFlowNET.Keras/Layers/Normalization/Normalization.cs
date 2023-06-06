/*****************************************************************************
   Copyright 2023 Haiping Chen. All Rights Reserved.

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

using Tensorflow.Common.Types;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Saving;

namespace Tensorflow.Keras.Layers
{
    public class Normalization : PreprocessingLayer
    {
        NormalizationArgs _args;

        int[] axis;
        int[] _reduce_axis;
        IVariableV1 adapt_mean, adapt_variance, count;
        Tensor mean, variance;
        Shape _broadcast_shape;
        float? input_mean, input_variance;
        TF_DataType compute_dtype = tf.float32;

        public Normalization(NormalizationArgs args) : base(args)
        {
            _args = args;
            if (args.Axis == null)
            {
                axis = new int[0];
            }
            else
            {
                axis = args.Axis.axis;
            }
            input_mean = args.Mean;
            input_variance = args.Variance;
        }

        public override void build(KerasShapesWrapper input_shape)
        {
            base.build(input_shape);
            var single_shape = input_shape.ToSingleShape();
            var ndim = single_shape.ndim;
            foreach (var (idx, x) in enumerate(axis))
                if (x < 0)
                    axis[idx] = ndim + x;

            var _keep_axis = axis.Select(d => d >= 0 ? d : d + ndim).ToArray();
            _reduce_axis = range(ndim).Where(d => !_keep_axis.Contains(d)).ToArray();
            var _reduce_axis_mask = range(ndim).Select(d => _keep_axis.Contains(d) ? 0 : 1).ToArray();
            // Broadcast any reduced axes.
            _broadcast_shape = new Shape(range(ndim).Select(d => _keep_axis.Contains(d) ? single_shape.dims[d] : 1).ToArray());
            var mean_and_var_shape = _keep_axis.Select(d => single_shape.dims[d]).ToArray();

            var param_dtype = DType == TF_DataType.DtInvalid ? TF_DataType.TF_FLOAT : DType;
            var param_shape = input_shape;

            if(input_mean == null)
            {
                adapt_mean = add_weight("mean",
                    mean_and_var_shape,
                    dtype: tf.float32,
                    initializer: tf.zeros_initializer,
                    trainable: false);

                adapt_variance = add_weight("variance",
                    mean_and_var_shape,
                    dtype: tf.float32,
                    initializer: tf.ones_initializer,
                    trainable: false);

                count = add_weight("count",
                    Shape.Scalar,
                    dtype: tf.int64,
                    initializer: tf.zeros_initializer,
                    trainable: false);

                finalize_state();
            }
            else
            {
                mean = input_mean * np.ones(mean_and_var_shape);
                variance = input_variance * np.ones(mean_and_var_shape);
                mean = tf.reshape(mean, _broadcast_shape);
                variance = tf.reshape(variance, _broadcast_shape);
                mean = tf.cast(mean, compute_dtype);
                variance = tf.cast(variance, compute_dtype);
            }
        }

        public override void reset_state()
        {
            if (input_mean != null && !built)
            {
                return;
            }
            adapt_mean.assign(tf.zeros_like(adapt_mean.AsTensor()));
            adapt_variance.assign(tf.ones_like(adapt_variance.AsTensor()));
            count.assign(tf.zeros_like(count.AsTensor()));
        }

        public override void finalize_state()
        {
            if (input_mean != null && !built)
            {
                return;
            }
            mean = tf.reshape(adapt_mean.AsTensor(), _broadcast_shape);
            variance = tf.reshape(adapt_variance.AsTensor(), _broadcast_shape);
        }

        public override void update_state(Tensor data)
        {
            data = tf.cast(data, adapt_mean.dtype);
            var (batch_mean, batch_variance) = tf.nn.moments(data, axes: _reduce_axis);
            var batch_shape = tf.shape(data, out_type: count.dtype);

            var batch_count = constant_op.constant(1L);
            if (_reduce_axis != null)
            {
                var batch_reduce_shape = tf.gather(batch_shape,  constant_op.constant(_reduce_axis));
                batch_count = tf.reduce_prod(batch_reduce_shape);
            }
            var total_count = batch_count + count.AsTensor();
            var batch_weight = tf.cast(batch_count, dtype: compute_dtype) / tf.cast(
                total_count, dtype: compute_dtype);
            var existing_weight = 1.0 - batch_weight;
            var total_mean = adapt_mean.AsTensor() * existing_weight + batch_mean * batch_weight;

            var total_variance = (
                    adapt_variance.AsTensor() + tf.square(adapt_mean.AsTensor() - total_mean)
                ) * existing_weight + (
                    batch_variance + tf.square(batch_mean - total_mean)
                ) * batch_weight;
            adapt_mean.assign(total_mean);
            adapt_variance.assign(total_variance);
            count.assign(total_count);
        }

        public override Shape ComputeOutputShape(Shape input_shape)
        {
            return input_shape;
        }

        public override void adapt(Tensor data, int? batch_size = null, int? steps = null)
        {
            base.adapt(data, batch_size: batch_size, steps: steps);
        }

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            if (_args.Invert)
            {
                return mean + (
                    inputs * tf.maximum(tf.sqrt(variance), keras.backend.epsilon())
            );
            }
            else
            {
                return (inputs - mean) / tf.maximum(
                    tf.sqrt(variance), keras.backend.epsilon());
            }
        }
    }
}
