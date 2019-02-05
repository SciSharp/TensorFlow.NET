using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class array_ops
    {
        public static Tensor zeros(Shape shape, TF_DataType dtype = TF_DataType.TF_FLOAT, string name = "")
        {
            Tensor output = null;

            dtype = dtype.as_base_dtype();
            Python.with(new ops.name_scope(name, "zeros", shape), self =>
            {
                name = self as ops.name_scope;
                switch (dtype)
                {
                    case TF_DataType.TF_BOOL:
                        output = _constant_if_small(false, shape, dtype, name);
                        break;
                    case TF_DataType.TF_DOUBLE:
                        output = _constant_if_small(0.0D, shape, dtype, name);
                        break;
                    case TF_DataType.TF_FLOAT:
                        output = _constant_if_small(0.0F, shape, dtype, name);
                        break;
                    case TF_DataType.TF_INT32:
                        output = _constant_if_small(0, shape, dtype, name);
                        break;
                    default:
                        break;
                }
            });

            return output;
        }

        private static Tensor _constant_if_small<T>(T value, Shape shape, TF_DataType dtype, string name)
        {
            Tensor tShape = null;
            var nd = np.zeros<T>(shape);
            if (shape.Size < 1000)
            {
                return constant_op.constant(nd, name);
            }
            else
            {
                tShape = constant_op._tensor_shape_tensor_conversion_function(shape.as_shape());
                var c = constant_op.constant(0);
                return gen_array_ops.fill(tShape, c, name);
            }
        }

        public static Tensor rank(Tensor input, string name = "")
        {
            return math_ops.rank_internal(input, name, optimize: true);
        }

        /// <summary>
        /// Returns the shape of a tensor.
        /// </summary>
        /// <param name="input">A `Tensor` or `SparseTensor`.</param>
        /// <param name="name">A name for the operation (optional).</param>
        /// <param name="out_type">
        /// (Optional) The specified output type of the operation
        /// (`int32` or `int64`). Defaults to `tf.int32`.
        /// </param>
        /// <returns>A `Tensor` of type `out_type`.</returns>
        public static Tensor shape(Tensor input, string name = "", TF_DataType out_type = TF_DataType.TF_INT32)
        {
            return shape_internal(input, name, optimize: true, out_type: out_type);
        }

        public static Tensor size(Tensor input, string name = "", TF_DataType out_type = TF_DataType.TF_INT32)
        {
            return size_internal(input, name, optimize: true, out_type: out_type);
        }

        private static Tensor shape_internal(Tensor input, string name = "", bool optimize = true, TF_DataType out_type = TF_DataType.TF_INT32)
        {
            return Python.with<ops.name_scope, Tensor>(new ops.name_scope(name, "Shape", new Tensor[] { input }), scope =>
            {
                name = scope;

                if (!tf.context.executing_eagerly())
                {
                    var input_tensor = ops.convert_to_tensor(input);
                    var input_shape = tensor_util.to_shape(input_tensor.shape);
                    if (optimize && input_shape.is_fully_defined())
                    {
                        var nd = np.array(input_tensor.shape, out_type.as_numpy_datatype());
                        return constant_op.constant(nd, name);
                    }
                }

                return gen_array_ops.shape(input);
            });
        }

        private static Tensor size_internal(Tensor input, string name = "", bool optimize = true, TF_DataType out_type = TF_DataType.TF_INT32)
        {
            return Python.with<ops.name_scope, Tensor>(new ops.name_scope(name, "Size", new Tensor[] { input }), scope =>
            {
                name = scope;

                if (!tf.context.executing_eagerly())
                {
                    var input_tensor = ops.convert_to_tensor(input);
                    var input_shape = tensor_util.to_shape(input_tensor.shape);
                    if (optimize)
                    {
                        if (input_shape.is_fully_defined())
                        {
                            var nd = np.array(input_tensor.shape, out_type.as_numpy_datatype());
                            return constant_op.constant(nd, name);
                        }
                    }

                    return gen_array_ops.size(input, name: name, out_type: out_type);
                }
                else
                {
                    // result = gen_array_ops.shape();
                }

                return null;
            });
        }
    }
}
