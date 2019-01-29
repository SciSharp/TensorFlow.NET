using NumSharp.Core;
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
        public static Tensor constant(NDArray nd, string name = "Const", bool verify_shape = false)
        {
            Graph g = ops.get_default_graph();
            var tensor_pb = tensor_util.make_tensor_proto(nd, verify_shape);
            var tensor_value = new AttrValue
            {
                Type = tensor_pb.Dtype,
                Tensor = tensor_pb
            };
            
            var dtype_value = new AttrValue
            {
                Type = tensor_value.Tensor.Dtype,
            };

            var attrs = new Dictionary<string, AttrValue>();
            attrs["value"] = tensor_value;
            attrs["dtype"] = dtype_value;

            var op = g.create_op("Const",
                null,
                new TF_DataType[] { (TF_DataType)dtype_value.Type },
                attrs: attrs,
                name: name);

            return op.outputs[0];
        }

        /// <summary>
        /// Function to convert TensorShape to Tensor.
        /// </summary>
        /// <param name="s"></param>
        /// <param name="dtype"></param>
        /// <param name="name"></param>
        /// <param name="as_ref"></param>
        /// <returns></returns>
        public static Tensor _tensor_shape_tensor_conversion_function(TensorShape s, TF_DataType dtype = TF_DataType.DtInvalid, string name = "", bool as_ref = false)
        {
            var s_list = s.Dimensions;
            var int64_value = 0;
            foreach(var dim in s_list)
            {
                if (dim > Math.Pow(2, 31))
                {
                    int64_value = dim;
                    break;
                }
            }

            if(int64_value > 0)
            {
                dtype = TF_DataType.TF_INT32;
            }

            if (string.IsNullOrEmpty(name))
                name = "shape_as_tensor";

            return constant_op.constant(s_list, name);
        }
    }
}
