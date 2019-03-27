using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class array_ops : Python
    {
        public static Tensor placeholder_with_default<T>(T input, int[] shape, string name = null) => gen_array_ops.placeholder_with_default(input, shape, name);

        public static Tensor zeros(Shape shape, TF_DataType dtype = TF_DataType.TF_FLOAT, string name = null)
        {
            dtype = dtype.as_base_dtype();
            return with(ops.name_scope(name, "zeros", shape), scope =>
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
            if (shape.Size < 1000)
            {
                return constant_op.constant(value, shape: shape, dtype: dtype, name: name);
            }
            else
            {
                tShape = constant_op._tensor_shape_tensor_conversion_function(shape.as_shape());
                var c = constant_op.constant(0);
                return gen_array_ops.fill(tShape, c, name: name);
            }
        }

        public static Tensor expand_dims(Tensor input, int axis = -1, string name = null, int dim = -1) => expand_dims_v2(input, axis, name);

        private static Tensor expand_dims_v2(Tensor input, int axis, string name = null) => gen_array_ops.expand_dims(input, axis, name);

        public static Tensor rank(Tensor input, string name = null)
        {
            return math_ops.rank_internal(input, name, optimize: true);
        }

        /// <summary>
        /// Creates a tensor with all elements set to 1.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="dtype"></param>
        /// <param name="name"></param>
        /// <param name="optimize"></param>
        /// <returns></returns>
        public static Tensor ones_like<T>(T tensor, TF_DataType dtype = TF_DataType.DtInvalid, string name = null, bool optimize = true)
            => ones_like_impl(tensor, dtype, name, optimize);

        public static Tensor reshape(Tensor tensor, Tensor shape, string name = null)
        {
            return gen_array_ops.reshape(tensor, shape, null);
        }

        private static Tensor ones_like_impl<T>(T tensor, TF_DataType dtype, string name, bool optimize = true)
        {
            return with(ops.name_scope(name, "ones_like", new { tensor }), scope =>
            {
                name = scope;
                var tensor1 = ops.convert_to_tensor(tensor, name: "tensor");
                var ones_shape = shape_internal(tensor1, optimize: optimize);
                if (dtype == TF_DataType.DtInvalid)
                    dtype = tensor1.dtype;
                var ret = ones(ones_shape, dtype: dtype, name: name);
                ret.shape = tensor1.shape;
                return ret;
            });
        }

        public static Tensor ones(Tensor shape, TF_DataType dtype = TF_DataType.TF_FLOAT, string name = null)
        {
            dtype = dtype.as_base_dtype();
            return with(ops.name_scope(name, "ones", new { shape }), scope =>
            {
                name = scope;
                var output = gen_array_ops.fill(shape, constant_op.constant(1.0f, dtype: dtype), name: name);
                return output;
            });
        }

        public static Tensor ones(int[] dims, TF_DataType dtype = TF_DataType.TF_FLOAT, string name = null)
        {
            dtype = dtype.as_base_dtype();
            return with(ops.name_scope(name, "ones", new { dims }), scope =>
            {
                name = scope;
                var shape = ops.convert_to_tensor(dims, dtype: TF_DataType.TF_INT32);
                var output = gen_array_ops.fill(shape, constant_op.constant(1.0f, dtype: dtype), name: name);
                return output;
            });
        }

        public static Tensor where(Tensor condition, Tensor x = null, Tensor y = null, string name = null)
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
        public static Tensor shape(Tensor input, string name = null, TF_DataType out_type = TF_DataType.TF_INT32)
        {
            return shape_internal(input, name, optimize: true, out_type: out_type);
        }

        public static Tensor size(Tensor input, string name = null, bool optimize = true, TF_DataType out_type = TF_DataType.TF_INT32)
        {
            return size_internal(input, name, optimize: optimize, out_type: out_type);
        }

        private static Tensor shape_internal(Tensor input, string name = null, bool optimize = true, TF_DataType out_type = TF_DataType.TF_INT32)
        {
            return with(ops.name_scope(name, "Shape", new { input }), scope =>
            {
                name = scope;

                if (!tf.context.executing_eagerly())
                {
                    var input_tensor = ops.convert_to_tensor(input);
                    var input_shape = tensor_util.to_shape(input_tensor.shape);
                    if (optimize && input_tensor.NDims > -1 && input_shape.is_fully_defined())
                    {
                        var nd = np.array(input_tensor.shape, out_type.as_numpy_datatype());
                        return constant_op.constant(nd, name: name);
                    }
                }

                return gen_array_ops.shape(input, name: name, out_type: out_type);
            });
        }

        private static Tensor size_internal(Tensor input, string name = null, bool optimize = true, TF_DataType out_type = TF_DataType.TF_INT32)
        {
            return with(ops.name_scope(name, "Size", new Tensor[] { input }), scope =>
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
                    throw new NotImplementedException("array_ops.size_internal");
                }

                return null;
            });
        }

        public static Tensor zeros_like(Tensor tensor, TF_DataType dtype = TF_DataType.DtInvalid, string name = null, bool optimize = true)
        {
            return with(ops.name_scope(name, "zeros_like", new Tensor[] { tensor }), scope =>
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

        /// <summary>
        ///   When building ops to compute gradients, this op prevents the contribution of
        ///   its inputs to be taken into account.Normally, the gradient generator adds ops
        ///   to a graph to compute the derivatives of a specified 'loss' by recursively
        ///   finding out inputs that contributed to its computation.If you insert this op
        ///   in the graph it inputs are masked from the gradient generator.  They are not
        ///   taken into account for computing gradients.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor stop_gradient(Tensor input, string name = null)
        {
            return gen_array_ops.stop_gradient(input,  name);
        }

        /// <summary>
        /// Removes dimensions of size 1 from the shape of a tensor.
        /// Given a tensor `input`, this operation returns a tensor of the same type with
        /// all dimensions of size 1 removed.If you don't want to remove all size 1
        /// dimensions, you can remove specific size 1 dimensions by specifying
        /// `axis`.
        /// </summary>
        /// <param name="input"> A `Tensor`. The `input` to squeeze.</param>
        /// <param name="axis"> An optional list of `ints`. Defaults to `[]`.
        /// If specified, only squeezes the dimensions listed.The dimension
        /// index starts at 0. It is an error to squeeze a dimension that is not 1.
        /// Must be in the range `[-rank(input), rank(input))`.</param>
        /// <param name="name"> A name for the operation (optional).</param>
        /// <param name="squeeze_dims" >Deprecated keyword argument that is now axis.</param>
        /// <returns>A `Tensor`. Has the same type as `input`.
        /// Contains the same data as `input`, but has one or more dimensions of
        /// size 1 removed.</returns>
        public static Tensor squeeze(Tensor input, int[] axis = null, string name = null, int[] squeeze_dims = null)
        {
            return gen_array_ops.squeeze(input, axis, name);
        }

        public static Tensor identity(Tensor input, string name = null)
        {
            return gen_array_ops.identity(input, name);
        }
        /// <summary>
        /// Computes the shape of a broadcast given symbolic shapes.
        /// When shape_x and shape_y are Tensors representing shapes(i.e.the result of
        /// calling tf.shape on another Tensor) this computes a Tensor which is the shape
        /// of the result of a broadcasting op applied in tensors of shapes shape_x and
        /// shape_y.
        /// For example, if shape_x is [1, 2, 3] and shape_y is [5, 1, 3], the result is a
        /// Tensor whose value is [5, 2, 3].
        /// This is useful when validating the result of a broadcasting operation when the
        /// tensors do not have statically known shapes.
        /// </summary>
        /// <param name="shape_x"> A rank 1 integer `Tensor`, representing the shape of x.</param>
        /// <param name="shape_y"> A rank 1 integer `Tensor`, representing the shape of y.</param>
        /// <returns> A rank 1 integer `Tensor` representing the broadcasted shape.</returns>
        public static Tensor broadcast_dynamic_shape(Tensor shape_x, Tensor shape_y)
        {
            return gen_array_ops.broadcast_args(shape_x, shape_y);
        }

        public static Tensor broadcast_static_shape(Tensor shape_x, Tensor shape_y)
        {
            return Framework.common_shapes.broadcast_shape(shape_x, shape_y);
        }

        public static Tensor gather(Tensor @params, Tensor indices, string name = null, int axis = 0)
        {
            return gen_array_ops.gather_v2(@params, indices, axis, name: name);
        }

        public static Tensor transpose(Tensor a, int[] perm = null, string name = "transpose", bool conjugate = false)
        {
            return with(ops.name_scope(name, "transpose", new { a }), scope =>
            {
                name = scope;
                return gen_array_ops.transpose(a, perm, name);
            });
        }
    }
}
