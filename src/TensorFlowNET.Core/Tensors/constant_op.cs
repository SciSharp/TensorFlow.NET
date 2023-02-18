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

using Tensorflow.NumPy;
using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Contexts;
using Tensorflow.Eager;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class constant_op
    {
        /// <summary>
        /// Creates a constant tensor.
        /// 
        /// The resulting tensor is populated with values of type `dtype`, as
        /// specified by arguments `value` and (optionally) `shape`
        /// </summary>
        /// <param name="value">A constant value (or list) of output type `dtype`.</param>
        /// <param name="dtype">The type of the elements of the resulting tensor.</param>
        /// <param name="shape">Optional dimensions of resulting tensor.</param>
        /// <param name="name">Optional name for the tensor.</param>
        /// <returns></returns>
        public static Tensor constant(object value, TF_DataType dtype = TF_DataType.DtInvalid, 
            Shape shape = null, bool verify_shape = false,
            bool allow_broadcast = true, string name = "Const")
        {
            if (value == null)
                return null;

            if(tf.executing_eagerly())
                return convert_to_eager_tensor(value, dtype, shape, name, verify_shape: verify_shape, allow_broadcast: allow_broadcast);
            else
                return convert_to_graph_tensor(value, dtype, shape, name, verify_shape: verify_shape, allow_broadcast: allow_broadcast);
        }

        private static Tensor _eager_reshape(Tensor tensor, int[] shape, Context ctx)
        {
            var attr_t = tensor.dtype.as_datatype_enum();
            var dims_t = convert_to_eager_tensor(shape, ctx, dtypes.int32);
            var inputs_flat = new[] { tensor, dims_t };
            var attrs = new object[] { "T", attr_t, "Tshape", TF_DataType.TF_INT32 };
            var result = tf.Runner.Execute(ctx, "Reshape", 1, inputs_flat, attrs);
            return result[0];
        }

        private static Tensor _eager_fill(int[] dims, Tensor value, Context ctx)
        {
            var attr_t = value.dtype.as_datatype_enum();
            var dims_t = convert_to_eager_tensor(dims, ctx, dtypes.int32);
            var inputs_flat = new[] { dims_t, value };
            var attrs = new object[] { "T", attr_t, "index_type", TF_DataType.TF_INT32 };
            var result = tf.Runner.Execute(ctx, "Fill", 1, inputs_flat, attrs);
            return result[0];
        }

        private static Tensor convert_to_eager_tensor(object value, Context ctx, TF_DataType dtype = TF_DataType.DtInvalid)
        {
            ctx.ensure_initialized();
            // convert data type
            if (dtype != TF_DataType.DtInvalid &&
                value.GetType().Name != "NDArray" &&
                value.GetType().BaseType.Name != "Array" &&
                dtype != value.GetDataType())
            {
                switch (dtype)
                {
                    case TF_DataType.TF_DOUBLE:
                        value = Convert.ToDouble(value);
                        break;
                    case TF_DataType.TF_FLOAT:
                        value = Convert.ToSingle(value);
                        break;
                    case TF_DataType.TF_INT64:
                        value = Convert.ToInt64(value);
                        break;
                    case TF_DataType.TF_INT32:
                        value = Convert.ToInt32(value);
                        break;
                    default:
                        break;
                }
            }
            else if (dtype != TF_DataType.DtInvalid &&
                value is NDArray nd &&
                nd.dtype != dtype)
            {
                value = math_ops.cast(nd, dtype);
            }

            // non ascii char
            if (dtype == TF_DataType.TF_STRING && value is byte[] bytes)
                return new EagerTensor(bytes, Shape.Scalar, TF_DataType.TF_STRING);

            switch (value)
            {
                case EagerTensor val:
                    return val;
                case NDArray val:
                    return val;
                case Shape val:
                    return new EagerTensor(val.dims, new Shape(val.ndim));
                case Axis val:
                    return new EagerTensor(val.axis, val.IsScalar ? Shape.Scalar : new Shape(val.size));
                case string val:
                    return new EagerTensor(new[] { val }, Shape.Scalar);
                case string[] val:
                    return new EagerTensor(val, new Shape(val.Length));
                case bool val:
                    return new EagerTensor(new[] { val }, Shape.Scalar);
                case byte val:
                    return new EagerTensor(new[] { val }, Shape.Scalar);
                case int val:
                    return new EagerTensor(new[] { val }, Shape.Scalar);
                case long val:
                    return new EagerTensor(new[] { val }, Shape.Scalar);
                case ulong val:
                    return new EagerTensor(new[] { val }, Shape.Scalar);
                case float val:
                    return new EagerTensor(new[] { val }, Shape.Scalar);
                case double val:
                    return new EagerTensor(new[] { val }, Shape.Scalar);
                case IEnumerable<Tensor> val:
                    return ops.convert_to_tensor(val);
                case Array val:
                    return new EagerTensor(val, val.GetShape());
                default:
                    throw new NotImplementedException($"convert_to_eager_tensor {value.GetType()}");
            }
        }

        static Tensor convert_to_eager_tensor(object value,
            TF_DataType dtype,
            Shape shape,
            string name,
            bool verify_shape,
            bool allow_broadcast)
        {
            var t = convert_to_eager_tensor(value, tf.Context, dtype: dtype);
            if (dtype != TF_DataType.DtInvalid && dtype != t.dtype)
            {
                t = math_ops.cast(t, dtype);
            }
            if (shape is null || shape.IsNull)
                return t;

            if (t.shape.Equals(shape))
                return t;

            if (verify_shape)
                throw new TypeError($"Expected Tensor's shape: {shape}, got {t.shape}.");

            var num_t = t.shape.size;
            if (num_t == shape.size)
                return _eager_reshape(t, shape, tf.Context);
            if (num_t == 1)
            {
                if (t.dtype == dtypes.@bool)
                    throw new NotImplementedException("");
                else
                    return _eager_fill(shape, t, tf.Context);
            }

            throw new NotImplementedException("");
        }

        static Tensor convert_to_graph_tensor(object value,
            TF_DataType dtype,
            Shape shape,
            string name,
            bool verify_shape,
            bool allow_broadcast)
        {
            Graph g = ops.get_default_graph();
            var tensor_value = new AttrValue();
            tensor_value.Tensor = tensor_util.make_tensor_proto(value,
                dtype: dtype,
                shape: shape,
                verify_shape: verify_shape,
                allow_broadcast: allow_broadcast);

            var dtype_value = new AttrValue
            {
                Type = tensor_value.Tensor.Dtype,
            };

            var attrs = new Dictionary<string, AttrValue>();
            attrs["value"] = tensor_value;
            attrs["dtype"] = dtype_value;

            var op = g.create_op("Const",
                new Tensor[0],
                new TF_DataType[] { dtype_value.Type.as_tf_dtype() },
                attrs: attrs,
                name: name);

            return op.outputs[0];
        }

        /// <summary>
        /// Function to convert Shape to Tensor.
        /// </summary>
        /// <param name="s"></param>
        /// <param name="dtype"></param>
        /// <param name="name"></param>
        /// <param name="as_ref"></param>
        /// <returns></returns>
        public static Tensor _tensor_shape_tensor_conversion_function(Shape s,
            TF_DataType dtype = TF_DataType.DtInvalid,
            string name = null,
            bool as_ref = false)
        {
            var s_list = s.dims;
            var int64_value = 0L;
            foreach (var dim in s_list)
            {
                if (dim > Math.Pow(2, 31))
                {
                    int64_value = dim;
                    break;
                }
            }

            dtype = int64_value > 0 ? TF_DataType.TF_INT64 : TF_DataType.TF_INT32;

            if (string.IsNullOrEmpty(name))
                name = "shape_as_tensor";

            return constant_op.constant(s_list, dtype: dtype, name: name);
        }

        public static bool is_constant(ITensorOrOperation tensor_or_op)
        {
            if (tensor_or_op is Tensor tensor)
                return tensor.op.type == "Const";
            else if (tensor_or_op is Operation op)
                return op.type == "Const";
            else
                throw new ValueError("is_constant");
        }
    }
}
