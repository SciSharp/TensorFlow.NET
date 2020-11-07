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
using static Tensorflow.Binding;

namespace Tensorflow.Operations.Activation
{
    public class sigmoid : IActivation
    {
        public Tensor Activate(Tensor x, string name = null)
        {
            return tf.sigmoid(x);
        }
    }

    public class tanh : IActivation
    {
        public Tensor Activate(Tensor x, string name = null)
        {
            return tf.tanh(x);
        }
    }

    public class leakyrelu : IActivation
    {
        private readonly float _alpha;

        public leakyrelu(float alpha = 0.3f)
        {
            _alpha = alpha;
        }

        public Tensor Activate(Tensor x, string name = null)
        {
            return nn_ops.leaky_relu(x, _alpha);
        }
    }

    public class elu : IActivation
    {
        private readonly float _alpha;

        public elu(float alpha = 0.1f)
        {
            _alpha = alpha;
        }

        public Tensor Activate(Tensor x, string name = null)
        {
            var res = gen_ops.elu(x);
            if (Math.Abs(_alpha - 0.1f) < 0.00001f)
            {
                return res;
            }

            return array_ops.@where(x > 0, res, _alpha * res);
        }
    }

    public class softmax : IActivation
    {
        private readonly int _axis;

        /// <summary>Initializes a new instance of the <see cref="T:System.Object"></see> class.</summary>
        public softmax(int axis = -1)
        {
            _axis = axis;
        }

        public Tensor Activate(Tensor x, string name = null)
        {
            return nn_ops.softmax(x, _axis);
        }
    }

    public class softplus : IActivation
    {
        public Tensor Activate(Tensor x, string name = null)
        {
            return gen_ops.softplus(x);
        }
    }

    public class softsign : IActivation
    {
        public Tensor Activate(Tensor x, string name = null)
        {
            return gen_ops.softsign(x);
        }
    }

    public class swish : IActivation
    {
        public Tensor Activate(Tensor x, string name = null)
        {
            return tf.multiply(x, tf.nn.sigmoid(x));
        }
    }

    public class linear : IActivation
    {
        public Tensor Activate(Tensor x, string name = null)
        {
            return x;
        }
    }


    public class exponential : IActivation
    {
        public Tensor Activate(Tensor x, string name = null)
        {
            return tf.exp(x, name: name);
        }
    }


    public class relu : IActivation
    {
        private readonly float _threshold;
        private readonly float _alpha;
        private readonly float? _maxValue;

        public relu(float threshold = 0f, float alpha = 0.2f, float? max_value = null)
        {
            _threshold = threshold;
            _alpha = alpha;
            _maxValue = max_value;
        }

        public Tensor Activate(Tensor x, string name = null)
        {
            //based on keras/backend.py
            if (Math.Abs(_alpha) > 0.000001f)
            {
                if (!_maxValue.HasValue && Math.Abs(_threshold) < 0.0001)
                {
                    return nn_ops.leaky_relu(x, _alpha);
                }
            }

            Tensor negative_part;
            if (Math.Abs(_threshold) > 0.000001f)
            {
                negative_part = gen_ops.relu(-x + _threshold);
            }
            else
            {
                negative_part = gen_ops.relu(-x + _threshold);
            }

            if (Math.Abs(_threshold) > 0.000001f)
            {
                x = x * math_ops.cast(tf.greater(x, _threshold), TF_DataType.TF_FLOAT);
            }
            else if (Math.Abs(_maxValue.Value - 6f) < 0.0001f)
            {
                x = gen_ops.relu6(x);
            }
            else
            {
                x = gen_ops.relu(x);
            }

            bool clip_max = _maxValue.HasValue;
            if (clip_max)
            {
                Tensor maxval = constant_op.constant(_maxValue, x.dtype.as_base_dtype());
                var zero = constant_op.constant(0.0f, x.dtype.as_base_dtype());
                x = gen_ops.clip_by_value(x, zero, maxval);
            }

            if (Math.Abs(_alpha) > 0.00001)
            {
                var a = constant_op.constant(_alpha, x.dtype.as_base_dtype());
                x -= a * negative_part;
            }

            return x;
        }
    }

    public class selu : IActivation
    {
        public Tensor Activate(Tensor x, string name = null)
        {
            const float alpha = 1.6732632423543772848170429916717f;
            const float scale = 1.0507009873554804934193349852946f;
            return scale * new elu(alpha).Activate(x, name);
        }
    }

    public class hard_sigmoid : IActivation
    {
        public Tensor Activate(Tensor x, string name = null)
        {
            x = (0.2 * x) + 0.5;
            var zero = tf.convert_to_tensor(0.0f, x.dtype.as_base_dtype());
            var one = tf.convert_to_tensor(1.0f, x.dtype.as_base_dtype());
            return tf.clip_by_value(x, zero, one);
        }
    }
}