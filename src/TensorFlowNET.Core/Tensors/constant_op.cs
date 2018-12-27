using System;
using System.Collections.Generic;
using System.Text;

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
        /// <param name="verify_shape">Boolean that enables verification of a shape of values.</param>
        /// <returns></returns>
        public static Tensor Create(object value, TF_DataType dtype = TF_DataType.DtInvalid, TensorShape shape = null, string name = "Const", bool verify_shape = false)
        {
            Graph g = ops.get_default_graph();
            var tensor_value = new AttrValue();
            var tensor_pb = tensor_util.make_tensor_proto(value, dtype, shape, verify_shape);
            tensor_value.Tensor = tensor_pb;
            var dtype_value = new AttrValue
            {
                Type = tensor_value.Tensor.Dtype,
            };

            var attrs = new Dictionary<string, AttrValue>();
            attrs["dtype"] = dtype_value;
            attrs["value"] = tensor_value;
            var const_tensor = g.create_op("Const", null, new TF_DataType[] { (TF_DataType)dtype_value.Type }, attrs: attrs).outputs[0];
            const_tensor.value = value;

            return const_tensor;
        }
    }
}
