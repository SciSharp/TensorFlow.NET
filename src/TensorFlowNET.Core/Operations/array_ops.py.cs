using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class array_ops
    {
        public static Tensor placeholder_with_default<T>(T input, int[] shape, string name = "") => gen_array_ops.placeholder_with_default(input, shape, name);

        public static Tensor zeros(Shape shape, TF_DataType dtype = TF_DataType.TF_FLOAT, string name = "")
        {
            dtype = dtype.as_base_dtype();
            return Python.with<ops.name_scope, Tensor>(new ops.name_scope(name, "zeros", shape), scope =>
            {
                name = scope;
                switch (dtype)
                {
                    case TF_DataType.TF_BOOL:
                        return _constant_if_small(false, shape, dtype, name);
                    case TF_DataType.TF_DOUBLE:
                        return _constant_if_small(0.0D, shape, dtype, name);
                    case TF_DataType.TF_FLOAT:
                        return _constant_if_small(0.0F, shape, dtype, name);
                    case TF_DataType.TF_INT32:
                        return _constant_if_small(0, shape, dtype, name);
                    default:
                        throw new TypeError("can't find type for zeros");
                }
            });
        }

        private static Tensor _constant_if_small<T>(T value, Shape shape, TF_DataType dtype, string name)
        {
            Tensor tShape = null;
            var nd = np.zeros<T>(shape);
            if (shape.Size < 1000)
            {
                return constant_op.constant(nd, name: name);
            }
            else
            {
                tShape = constant_op._tensor_shape_tensor_conversion_function(shape.as_shape());
                var c = constant_op.constant(0);
                return gen_array_ops.fill(tShape, c, name: name);
            }
        }

        public static Tensor rank(Tensor input, string name = "")
        {
            return math_ops.rank_internal(input, name, optimize: true);
        }

        public static Tensor where(Tensor condition, Tensor x = null, Tensor y = null, string name = "")
        {
            if( x == null && y == null)
            {
                throw new NotImplementedException("where");
            }
            else if(x != null && y != null)
            {
                return gen_array_ops.select(condition, x, y, name);
            }
            else
            {
                throw new ValueError("x and y must both be non-None or both be None.");
            }
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
                        return constant_op.constant(nd, name: name);
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
                            return constant_op.constant(nd, name: name);
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

        public static Tensor zeros_like(Tensor tensor, TF_DataType dtype = TF_DataType.DtInvalid, string name = "", bool optimize = true)
        {
            return Python.with<ops.name_scope, Tensor>(new ops.name_scope(name, "zeros_like", new Tensor[] { tensor }), scope =>
            {
                name = scope;
                tensor = ops.convert_to_tensor(tensor, name: "tensor");

                // is_fully_defined return unexpected value.
                if (optimize && tensor_util.to_shape(tensor.shape).is_fully_defined() && dtype != TF_DataType.TF_VARIANT)
                {

                }

                if(dtype != TF_DataType.DtInvalid && dtype != tensor.dtype && dtype != TF_DataType.TF_VARIANT)
                {
                    throw new NotImplementedException("zeros_like");
                    // return zeros(shape_internal(tensor, optimize: optimize), dtype: dtype, name: name);
                }
                else
                {
                    return gen_array_ops.zeros_like(tensor, name: name);
                }
            });
        }
    }
}
