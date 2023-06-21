/*Wrappers around TensorFlow ops. This file is MACHINE GENERATED! Do not edit.*/

using Tensorflow.Eager;
using Tensorflow.Contexts;
using Tensorflow.Exceptions;
using static Tensorflow.Binding;

namespace Tensorflow;

public static class gen_array_ops
{
    /// <summary>
    /// 
    /// </summary>
    /// <param name="input"></param>
    /// <param name="num_lower"></param>
    /// <param name="num_upper"></param>
    /// <returns></returns>
    public static Tensor batch_matrix_band_part(Tensor input, Tensor num_lower, Tensor num_upper, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "BatchMatrixBandPart", name) { args = new object[] { input, num_lower, num_upper }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return batch_matrix_band_part_eager_fallback(input, num_lower, num_upper, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["num_lower"] = num_lower;
        keywords["num_upper"] = num_upper;
        var _op = tf.OpDefLib._apply_op_helper("BatchMatrixBandPart", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("BatchMatrixBandPart", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor batch_matrix_band_part_eager_fallback(Tensor input, Tensor num_lower, Tensor num_upper, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, num_lower, num_upper };
        object[] _attrs = new object[] { "T", input.dtype };
        var _result = _execute.execute("BatchMatrixBandPart", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("BatchMatrixBandPart", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="diagonal"></param>
    /// <returns></returns>
    public static Tensor batch_matrix_diag(Tensor diagonal, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "BatchMatrixDiag", name) { args = new object[] { diagonal }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return batch_matrix_diag_eager_fallback(diagonal, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["diagonal"] = diagonal;
        var _op = tf.OpDefLib._apply_op_helper("BatchMatrixDiag", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("BatchMatrixDiag", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor batch_matrix_diag_eager_fallback(Tensor diagonal, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { diagonal };
        object[] _attrs = new object[] { "T", diagonal.dtype };
        var _result = _execute.execute("BatchMatrixDiag", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("BatchMatrixDiag", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public static Tensor batch_matrix_diag_part(Tensor input, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "BatchMatrixDiagPart", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return batch_matrix_diag_part_eager_fallback(input, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        var _op = tf.OpDefLib._apply_op_helper("BatchMatrixDiagPart", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("BatchMatrixDiagPart", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor batch_matrix_diag_part_eager_fallback(Tensor input, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "T", input.dtype };
        var _result = _execute.execute("BatchMatrixDiagPart", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("BatchMatrixDiagPart", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="input"></param>
    /// <param name="diagonal"></param>
    /// <returns></returns>
    public static Tensor batch_matrix_set_diag(Tensor input, Tensor diagonal, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "BatchMatrixSetDiag", name) { args = new object[] { input, diagonal }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return batch_matrix_set_diag_eager_fallback(input, diagonal, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["diagonal"] = diagonal;
        var _op = tf.OpDefLib._apply_op_helper("BatchMatrixSetDiag", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("BatchMatrixSetDiag", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor batch_matrix_set_diag_eager_fallback(Tensor input, Tensor diagonal, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, diagonal };
        object[] _attrs = new object[] { "T", input.dtype };
        var _result = _execute.execute("BatchMatrixSetDiag", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("BatchMatrixSetDiag", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// BatchToSpace for 4-D tensors of type T.
    /// </summary>
    /// <remarks>
    /// 
    /// This is a legacy version of the more general BatchToSpaceND.
    /// 
    /// Rearranges (permutes) data from batch into blocks of spatial data, followed by
    /// cropping. This is the reverse transformation of SpaceToBatch. More specifically,
    /// this op outputs a copy of the input tensor where values from the `batch`
    /// dimension are moved in spatial blocks to the `height` and `width` dimensions,
    /// followed by cropping along the `height` and `width` dimensions.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="crops"></param>
    /// <param name="block_size"></param>
    /// <returns></returns>
    public static Tensor batch_to_space(Tensor input, Tensor crops, int block_size = 0, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "BatchToSpace", name) { args = new object[] { input, crops }, attrs = new Dictionary<string, object>() { ["block_size"] = block_size } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return batch_to_space_eager_fallback(input, crops, block_size: block_size, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["crops"] = crops;
        keywords["block_size"] = block_size;
        var _op = tf.OpDefLib._apply_op_helper("BatchToSpace", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "block_size", _op._get_attr_int("block_size"), "Tidx", _op._get_attr_type("Tidx") };
            _execute.record_gradient("BatchToSpace", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor batch_to_space_eager_fallback(Tensor input, Tensor crops, int block_size, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, crops };
        object[] _attrs = new object[] { "T", input.dtype, "block_size", block_size, "Tidx", crops.dtype };
        var _result = _execute.execute("BatchToSpace", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("BatchToSpace", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// BatchToSpace for N-D tensors of type T.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation reshapes the "batch" dimension 0 into `M + 1` dimensions of shape
    /// `block_shape + [batch]`, interleaves these blocks back into the grid defined by
    /// the spatial dimensions `[1, ..., M]`, to obtain a result with the same rank as
    /// the input.  The spatial dimensions of this intermediate result are then
    /// optionally cropped according to `crops` to produce the output.  This is the
    /// reverse of SpaceToBatch.  See below for a precise description.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="block_shape"></param>
    /// <param name="crops"></param>
    /// <returns></returns>
    public static Tensor batch_to_space_nd(Tensor input, Tensor block_shape, Tensor crops, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "BatchToSpaceND", name) { args = new object[] { input, block_shape, crops }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return batch_to_space_nd_eager_fallback(input, block_shape, crops, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["block_shape"] = block_shape;
        keywords["crops"] = crops;
        var _op = tf.OpDefLib._apply_op_helper("BatchToSpaceND", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tblock_shape", _op._get_attr_type("Tblock_shape"), "Tcrops", _op._get_attr_type("Tcrops") };
            _execute.record_gradient("BatchToSpaceND", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor batch_to_space_nd_eager_fallback(Tensor input, Tensor block_shape, Tensor crops, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, block_shape, crops };
        object[] _attrs = new object[] { "T", input.dtype, "Tblock_shape", block_shape.dtype, "Tcrops", crops.dtype };
        var _result = _execute.execute("BatchToSpaceND", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("BatchToSpaceND", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Bitcasts a tensor from one type to another without copying data.
    /// </summary>
    /// <remarks>
    /// 
    /// Given a tensor `input`, this operation returns a tensor that has the same buffer
    /// data as `input` with datatype `type`.
    /// 
    /// If the input datatype `T` is larger than the output datatype `type` then the
    /// shape changes from [...] to [..., sizeof(`T`)/sizeof(`type`)].
    /// 
    /// If `T` is smaller than `type`, the operator requires that the rightmost
    /// dimension be equal to sizeof(`type`)/sizeof(`T`). The shape then goes from
    /// [..., sizeof(`type`)/sizeof(`T`)] to [...].
    /// 
    /// tf.bitcast() and tf.cast() work differently when real dtype is casted as a complex dtype
    /// (e.g. tf.complex64 or tf.complex128) as tf.cast() make imaginary part 0 while tf.bitcast()
    /// gives module error.
    /// For example,
    /// 
    /// Example 1:
    /// 
    /// >>> a = [1., 2., 3.]
    /// >>> equality_bitcast = tf.bitcast(a, tf.complex128)
    /// Traceback (most recent call last):
    /// ...
    /// InvalidArgumentError: Cannot bitcast from 1 to 18 [Op:Bitcast]
    /// >>> equality_cast = tf.cast(a, tf.complex128)
    /// >>> print(equality_cast)
    /// tf.Tensor([1.+0.j 2.+0.j 3.+0.j], shape=(3,), dtype=complex128)
    /// 
    /// Example 2:
    /// 
    /// >>> tf.bitcast(tf.constant(0xffffffff, dtype=tf.uint32), tf.uint8)
    /// <tf.Tensor: shape=(4,), dtype=uint8, numpy=array([255, 255, 255, 255], dtype=uint8)>
    /// 
    /// Example 3:
    /// 
    /// >>> x = [1., 2., 3.]
    /// >>> y = [0., 2., 3.]
    /// >>> equality= tf.equal(x,y)
    /// >>> equality_cast = tf.cast(equality,tf.float32)
    /// >>> equality_bitcast = tf.bitcast(equality_cast,tf.uint8)
    /// >>> print(equality)
    /// tf.Tensor([False True True], shape=(3,), dtype=bool)
    /// >>> print(equality_cast)
    /// tf.Tensor([0. 1. 1.], shape=(3,), dtype=float32)
    /// >>> print(equality_bitcast)
    /// tf.Tensor(
    ///     [[  0   0   0   0]
    ///      [  0   0 128  63]
    ///      [  0   0 128  63]], shape=(3, 4), dtype=uint8)
    /// 
    /// *NOTE*: Bitcast is implemented as a low-level cast, so machines with different
    /// endian orderings will give different results.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="type"></param>
    /// <returns></returns>
    public static Tensor bitcast(Tensor input, TF_DataType type, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Bitcast", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { ["type"] = type } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return bitcast_eager_fallback(input, type: type, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["type"] = type;
        var _op = tf.OpDefLib._apply_op_helper("Bitcast", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "type", _op._get_attr_type("type") };
            _execute.record_gradient("Bitcast", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor bitcast_eager_fallback(Tensor input, TF_DataType type, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "T", input.dtype, "type", type };
        var _result = _execute.execute("Bitcast", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Bitcast", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Return the shape of s0 op s1 with broadcast.
    /// </summary>
    /// <remarks>
    /// 
    /// Given `s0` and `s1`, tensors that represent shapes, compute `r0`, the
    /// broadcasted shape. `s0`, `s1` and `r0` are all integer vectors.
    /// 
    /// </remarks>
    /// <param name="s0"></param>
    /// <param name="s1"></param>
    /// <returns></returns>
    public static Tensor broadcast_args(Tensor s0, Tensor s1, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "BroadcastArgs", name) { args = new object[] { s0, s1 }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return broadcast_args_eager_fallback(s0, s1, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["s0"] = s0;
        keywords["s1"] = s1;
        var _op = tf.OpDefLib._apply_op_helper("BroadcastArgs", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("BroadcastArgs", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor broadcast_args_eager_fallback(Tensor s0, Tensor s1, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { s0, s1 };
        object[] _attrs = new object[] { "T", s0.dtype };
        var _result = _execute.execute("BroadcastArgs", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("BroadcastArgs", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Return the reduction indices for computing gradients of s0 op s1 with broadcast.
    /// </summary>
    /// <remarks>
    /// 
    /// This is typically used by gradient computations for a broadcasting operation.
    /// 
    /// </remarks>
    /// <param name="s0"></param>
    /// <param name="s1"></param>
    /// <returns></returns>
    public static Tensor[] broadcast_gradient_args(Tensor s0, Tensor s1, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "BroadcastGradientArgs", name) { args = new object[] { s0, s1 }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return broadcast_gradient_args_eager_fallback(s0, s1, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["s0"] = s0;
        keywords["s1"] = s1;
        var _op = tf.OpDefLib._apply_op_helper("BroadcastGradientArgs", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("BroadcastGradientArgs", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] broadcast_gradient_args_eager_fallback(Tensor s0, Tensor s1, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { s0, s1 };
        object[] _attrs = new object[] { "T", s0.dtype };
        var _result = _execute.execute("BroadcastGradientArgs", 2, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("BroadcastGradientArgs", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Broadcast an array for a compatible shape.
    /// </summary>
    /// <remarks>
    /// 
    /// Broadcasting is the process of making arrays to have compatible shapes
    /// for arithmetic operations. Two shapes are compatible if for each
    /// dimension pair they are either equal or one of them is one.
    /// 
    /// For example:
    /// 
    /// >>> x = tf.constant([[1, 2, 3]])   # Shape (1, 3,)
    /// >>> y = tf.broadcast_to(x, [2, 3])
    /// >>> print(y)
    /// tf.Tensor(
    ///     [[1 2 3]
    ///      [1 2 3]], shape=(2, 3), dtype=int32)
    /// 
    /// In the above example, the input Tensor with the shape of `[1, 3]`
    /// is broadcasted to output Tensor with shape of `[2, 3]`.
    /// 
    /// When broadcasting, if a tensor has fewer axes than necessary its shape is
    /// padded on the left with ones. So this gives the same result as the previous
    /// example:
    /// 
    /// >>> x = tf.constant([1, 2, 3])   # Shape (3,)
    /// >>> y = tf.broadcast_to(x, [2, 3])
    /// 
    /// 
    /// When doing broadcasted operations such as multiplying a tensor
    /// by a scalar, broadcasting (usually) confers some time or space
    /// benefit, as the broadcasted tensor is never materialized.
    /// 
    /// However, `broadcast_to` does not carry with it any such benefits.
    /// The newly-created tensor takes the full memory of the broadcasted
    /// shape. (In a graph context, `broadcast_to` might be fused to
    /// subsequent operation and then be optimized away, however.)
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="shape"></param>
    /// <returns></returns>
    public static Tensor broadcast_to(Tensor input, Tensor shape, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "BroadcastTo", name) { args = new object[] { input, shape }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return broadcast_to_eager_fallback(input, shape, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["shape"] = shape;
        var _op = tf.OpDefLib._apply_op_helper("BroadcastTo", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tidx", _op._get_attr_type("Tidx") };
            _execute.record_gradient("BroadcastTo", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor broadcast_to_eager_fallback(Tensor input, Tensor shape, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, shape };
        object[] _attrs = new object[] { "T", input.dtype, "Tidx", shape.dtype };
        var _result = _execute.execute("BroadcastTo", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("BroadcastTo", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Checks a tensor for NaN and Inf values.
    /// </summary>
    /// <remarks>
    /// 
    /// When run, reports an `InvalidArgument` error if `tensor` has any values
    /// that are not a number (NaN) or infinity (Inf). Otherwise, returns the input
    /// tensor.
    /// 
    /// Example usage:
    /// 
    /// ``` python
    /// a = tf.Variable(1.0)
    /// tf.debugging.check_numerics(a, message='')
    /// 
    /// b = tf.Variable(np.nan)
    /// try:
    ///   tf.debugging.check_numerics(b, message='Checking b')
    /// except Exception as e:
    ///   assert "Checking b : Tensor had NaN values" in e.message
    /// 
    /// c = tf.Variable(np.inf)
    /// try:
    ///   tf.debugging.check_numerics(c, message='Checking c')
    /// except Exception as e:
    ///   assert "Checking c : Tensor had Inf values" in e.message
    /// ```
    /// 
    /// 
    /// </remarks>
    /// <param name="tensor"></param>
    /// <param name="message">
    /// 
    /// Prefix of the error message.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor check_numerics(Tensor tensor, string message, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "CheckNumerics", name) { args = new object[] { tensor }, attrs = new Dictionary<string, object>() { ["message"] = message } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return check_numerics_eager_fallback(tensor, message: message, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["tensor"] = tensor;
        keywords["message"] = message;
        var _op = tf.OpDefLib._apply_op_helper("CheckNumerics", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "message", _op.get_attr("message") };
            _execute.record_gradient("CheckNumerics", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor check_numerics_eager_fallback(Tensor tensor, string message, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { tensor };
        object[] _attrs = new object[] { "T", tensor.dtype, "message", message };
        var _result = _execute.execute("CheckNumerics", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("CheckNumerics", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Checks a tensor for NaN, -Inf and +Inf values.
    /// </summary>
    /// <remarks>
    /// 
    /// When run, reports an `InvalidArgument` error if `tensor` has any values
    /// that are not a number (NaN) or infinity (Inf). Otherwise, returns the input
    /// tensor. Unlike CheckNumerics (V1), CheckNumericsV2 distinguishes -Inf and +Inf
    /// in the errors it throws.
    /// 
    /// </remarks>
    /// <param name="tensor"></param>
    /// <param name="message">
    /// 
    /// Prefix of the error message.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor check_numerics_v2(Tensor tensor, string message, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "CheckNumericsV2", name) { args = new object[] { tensor }, attrs = new Dictionary<string, object>() { ["message"] = message } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return check_numerics_v2_eager_fallback(tensor, message: message, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["tensor"] = tensor;
        keywords["message"] = message;
        var _op = tf.OpDefLib._apply_op_helper("CheckNumericsV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "message", _op.get_attr("message") };
            _execute.record_gradient("CheckNumericsV2", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor check_numerics_v2_eager_fallback(Tensor tensor, string message, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { tensor };
        object[] _attrs = new object[] { "T", tensor.dtype, "message", message };
        var _result = _execute.execute("CheckNumericsV2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("CheckNumericsV2", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Concatenates tensors along one dimension.
    /// </summary>
    /// <param name="concat_dim"></param>
    /// <param name="values"></param>
    /// <returns></returns>
    public static Tensor concat(Tensor concat_dim, Tensors values, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Concat", name) { args = new object[] { concat_dim, values }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return concat_eager_fallback(concat_dim, values, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["concat_dim"] = concat_dim;
        keywords["values"] = values;
        var _op = tf.OpDefLib._apply_op_helper("Concat", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "N", _op._get_attr_int("N"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("Concat", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor concat_eager_fallback(Tensor concat_dim, Tensors values, string name, Context ctx)
    {
        List<Tensor> _inputs_flat_list = new();
        _inputs_flat_list.Add(concat_dim);
        _inputs_flat_list.AddRange(values);
        var _inputs_flat = _inputs_flat_list.ToArray();
        object[] _attrs = new object[] { "N", values.Length, "T", values.dtype };
        var _result = _execute.execute("Concat", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Concat", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes offsets of concat inputs within its output.
    /// </summary>
    /// <remarks>
    /// 
    /// For example:
    /// 
    /// >>> x = [2, 2, 7]
    /// >>> y = [2, 3, 7]
    /// >>> z = [2, 9, 7]
    /// >>> offsets = concat_offset(1, [x, y, z])
    /// >>> [list(off.numpy()) for off in offsets]
    /// [[0, 0, 0], [0, 2, 0], [0, 5, 0]]
    /// 
    /// This is typically used by gradient computations for a concat operation.
    /// 
    /// </remarks>
    /// <param name="concat_dim"></param>
    /// <param name="shape"></param>
    /// <returns></returns>
    public static Tensor[] concat_offset(Tensor concat_dim, Tensors shape, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ConcatOffset", name) { args = new object[] { concat_dim, shape }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return concat_offset_eager_fallback(concat_dim, shape, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["concat_dim"] = concat_dim;
        keywords["shape"] = shape;
        var _op = tf.OpDefLib._apply_op_helper("ConcatOffset", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "N", _op._get_attr_int("N") };
            _execute.record_gradient("ConcatOffset", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] concat_offset_eager_fallback(Tensor concat_dim, Tensors shape, string name, Context ctx)
    {
        List<Tensor> _inputs_flat_list = new();
        _inputs_flat_list.Add(concat_dim);
        _inputs_flat_list.AddRange(shape);
        var _inputs_flat = _inputs_flat_list.ToArray();
        object[] _attrs = new object[] { "N", shape.Length };
        var _result = _execute.execute("ConcatOffset", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ConcatOffset", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Concatenates tensors along one dimension.
    /// </summary>
    /// <param name="values"></param>
    /// <param name="axis"></param>
    /// <returns></returns>
    public static Tensor concat_v2(Tensors values, Tensor axis, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ConcatV2", name) { args = new object[] { values, axis }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return concat_v2_eager_fallback(values, axis, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["values"] = values;
        keywords["axis"] = axis;
        var _op = tf.OpDefLib._apply_op_helper("ConcatV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "N", _op._get_attr_int("N"), "T", _op._get_attr_type("T"), "Tidx", _op._get_attr_type("Tidx") };
            _execute.record_gradient("ConcatV2", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor concat_v2_eager_fallback(Tensors values, Tensor axis, string name, Context ctx)
    {
        List<Tensor> _inputs_flat_list = new();
        _inputs_flat_list.AddRange(values);
        _inputs_flat_list.Add(axis);
        var _inputs_flat = _inputs_flat_list.ToArray();
        object[] _attrs = new object[] { "N", values.Length, "T", values.dtype, "Tidx", axis.dtype };
        var _result = _execute.execute("ConcatV2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ConcatV2", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Shuffle dimensions of x according to a permutation and conjugate the result.
    /// </summary>
    /// <remarks>
    /// 
    /// The output `y` has the same rank as `x`. The shapes of `x` and `y` satisfy:
    ///   `y.shape[i] == x.shape[perm[i]] for i in [0, 1, ..., rank(x) - 1]`
    ///   `y[i,j,k,...,s,t,u] == conj(x[perm[i], perm[j], perm[k],...,perm[s], perm[t], perm[u]])`
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="perm"></param>
    /// <returns></returns>
    public static Tensor conjugate_transpose(Tensor x, Tensor perm, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ConjugateTranspose", name) { args = new object[] { x, perm }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return conjugate_transpose_eager_fallback(x, perm, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["perm"] = perm;
        var _op = tf.OpDefLib._apply_op_helper("ConjugateTranspose", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tperm", _op._get_attr_type("Tperm") };
            _execute.record_gradient("ConjugateTranspose", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor conjugate_transpose_eager_fallback(Tensor x, Tensor perm, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, perm };
        object[] _attrs = new object[] { "T", x.dtype, "Tperm", perm.dtype };
        var _result = _execute.execute("ConjugateTranspose", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ConjugateTranspose", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns a constant tensor.
    /// </summary>
    /// <param name="value">
    /// 
    /// Attr `value` is the tensor to return.
    /// 
    /// </param>
    /// <param name="dtype"></param>
    /// <returns></returns>
    public static Tensor _const(TensorProto value, TF_DataType dtype, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Const", name) { args = new object[] { }, attrs = new Dictionary<string, object>() { ["value"] = value, ["dtype"] = dtype } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return const_eager_fallback(value: value, dtype: dtype, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["value"] = value;
        keywords["dtype"] = dtype;
        var _op = tf.OpDefLib._apply_op_helper("Const", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "value", _op.get_attr("value"), "dtype", _op._get_attr_type("dtype") };
            _execute.record_gradient("Const", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor const_eager_fallback(TensorProto value, TF_DataType dtype, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { };
        object[] _attrs = new object[] { "value", value, "dtype", dtype };
        var _result = _execute.execute("Const", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Const", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Identity op for gradient debugging.
    /// </summary>
    /// <remarks>
    /// 
    /// This op is hidden from public in Python. It is used by TensorFlow Debugger to
    /// register gradient tensors for gradient debugging.
    /// This op operates on non-reference-type tensors.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <returns></returns>
    public static Tensor debug_gradient_identity(Tensor input, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "DebugGradientIdentity", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return debug_gradient_identity_eager_fallback(input, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        var _op = tf.OpDefLib._apply_op_helper("DebugGradientIdentity", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("DebugGradientIdentity", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor debug_gradient_identity_eager_fallback(Tensor input, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "T", input.dtype };
        var _result = _execute.execute("DebugGradientIdentity", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("DebugGradientIdentity", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Identity op for gradient debugging.
    /// </summary>
    /// <remarks>
    /// 
    /// This op is hidden from public in Python. It is used by TensorFlow Debugger to
    /// register gradient tensors for gradient debugging.
    /// This op operates on reference-type tensors.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <returns></returns>
    public static Tensor debug_gradient_ref_identity(Tensor input, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            throw new RuntimeError("debug_gradient_ref_identity op does not support eager execution. Arg input is a ref.");
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        var _op = tf.OpDefLib._apply_op_helper("DebugGradientRefIdentity", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("DebugGradientRefIdentity", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor debug_gradient_ref_identity_eager_fallback(Tensor input, string name, Context ctx)
    {
        throw new RuntimeError($"debug_gradient_ref_identity op does not support eager execution. Arg 'input' is a ref.");
    }
    /// <summary>
    /// Makes a copy of `x`.
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor deep_copy(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "DeepCopy", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return deep_copy_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("DeepCopy", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("DeepCopy", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor deep_copy_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("DeepCopy", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("DeepCopy", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// DepthToSpace for tensors of type T.
    /// </summary>
    /// <remarks>
    /// 
    /// Rearranges data from depth into blocks of spatial data.
    /// This is the reverse transformation of SpaceToDepth. More specifically,
    /// this op outputs a copy of the input tensor where values from the `depth`
    /// dimension are moved in spatial blocks to the `height` and `width` dimensions.
    /// The attr `block_size` indicates the input block size and how the data is moved.
    /// 
    ///   * Chunks of data of size `block_size * block_size` from depth are rearranged
    ///     into non-overlapping blocks of size `block_size x block_size`
    ///   * The width of the output tensor is `input_depth * block_size`, whereas the
    ///     height is `input_height * block_size`.
    ///   * The Y, X coordinates within each block of the output image are determined
    ///     by the high order component of the input channel index.
    ///   * The depth of the input tensor must be divisible by
    ///     `block_size * block_size`.
    /// 
    /// The `data_format` attr specifies the layout of the input and output tensors
    /// with the following options:
    ///   "NHWC": `[ batch, height, width, channels ]`
    ///   "NCHW": `[ batch, channels, height, width ]`
    ///   "NCHW_VECT_C":
    ///       `qint8 [ batch, channels / 4, height, width, 4 ]`
    /// 
    /// It is useful to consider the operation as transforming a 6-D Tensor.
    /// e.g. for data_format = NHWC,
    ///      Each element in the input tensor can be specified via 6 coordinates,
    ///      ordered by decreasing memory layout significance as:
    ///      n,iY,iX,bY,bX,oC  (where n=batch index, iX, iY means X or Y coordinates
    ///                         within the input image, bX, bY means coordinates
    ///                         within the output block, oC means output channels).
    ///      The output would be the input transposed to the following layout:
    ///      n,iY,bY,iX,bX,oC
    /// 
    /// This operation is useful for resizing the activations between convolutions
    /// (but keeping all data), e.g. instead of pooling. It is also useful for training
    /// purely convolutional models.
    /// 
    /// For example, given an input of shape `[1, 1, 1, 4]`, data_format = "NHWC" and
    /// block_size = 2:
    /// 
    /// ```
    /// x = [[[[1, 2, 3, 4]]]]
    /// 
    /// ```
    /// 
    /// This operation will output a tensor of shape `[1, 2, 2, 1]`:
    /// 
    /// ```
    ///    [[[[1], [2]],
    ///      [[3], [4]]]]
    /// ```
    /// 
    /// Here, the input has a batch of 1 and each batch element has shape `[1, 1, 4]`,
    /// the corresponding output will have 2x2 elements and will have a depth of
    /// 1 channel (1 = `4 / (block_size * block_size)`).
    /// The output element shape is `[2, 2, 1]`.
    /// 
    /// For an input tensor with larger depth, here of shape `[1, 1, 1, 12]`, e.g.
    /// 
    /// ```
    /// x = [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
    /// ```
    /// 
    /// This operation, for block size of 2, will return the following tensor of shape
    /// `[1, 2, 2, 3]`
    /// 
    /// ```
    ///    [[[[1, 2, 3], [4, 5, 6]],
    ///      [[7, 8, 9], [10, 11, 12]]]]
    /// 
    /// ```
    /// 
    /// Similarly, for the following input of shape `[1 2 2 4]`, and a block size of 2:
    /// 
    /// ```
    /// x =  [[[[1, 2, 3, 4],
    ///        [5, 6, 7, 8]],
    ///       [[9, 10, 11, 12],
    ///        [13, 14, 15, 16]]]]
    /// ```
    /// 
    /// the operator will return the following tensor of shape `[1 4 4 1]`:
    /// 
    /// ```
    /// x = [[[ [1],   [2],  [5],  [6]],
    ///       [ [3],   [4],  [7],  [8]],
    ///       [ [9],  [10], [13],  [14]],
    ///       [ [11], [12], [15],  [16]]]]
    /// 
    /// ```
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="block_size">
    /// 
    /// The size of the spatial block, same as in Space2Depth.
    /// 
    /// </param>
    /// <param name="data_format"></param>
    /// <returns></returns>
    public static Tensor depth_to_space(Tensor input, int block_size = 0, string data_format = "NHWC", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "DepthToSpace", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { ["block_size"] = block_size, ["data_format"] = data_format } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return depth_to_space_eager_fallback(input, block_size: block_size, data_format: data_format, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        if (data_format is null)
        {
            data_format = "NHWC";
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["block_size"] = block_size;
        keywords["data_format"] = data_format;
        var _op = tf.OpDefLib._apply_op_helper("DepthToSpace", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "block_size", _op._get_attr_int("block_size"), "data_format", _op.get_attr("data_format") };
            _execute.record_gradient("DepthToSpace", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor depth_to_space_eager_fallback(Tensor input, int block_size, string data_format, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "T", input.dtype, "block_size", block_size, "data_format", data_format };
        var _result = _execute.execute("DepthToSpace", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("DepthToSpace", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Dequantize the 'input' tensor into a float or bfloat16 Tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// [min_range, max_range] are scalar floats that specify the range for
    /// the output. The 'mode' attribute controls exactly which calculations are
    /// used to convert the float values to their quantized equivalents.
    /// 
    /// In 'MIN_COMBINED' mode, each value of the tensor will undergo the following:
    /// 
    /// ```
    /// if T == qint8: in[i] += (range(T) + 1)/ 2.0
    /// out[i] = min_range + (in[i]* (max_range - min_range) / range(T))
    /// ```
    /// here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`
    /// 
    /// *MIN_COMBINED Mode Example*
    /// 
    /// If the input comes from a QuantizedRelu6, the output type is
    /// quint8 (range of 0-255) but the possible range of QuantizedRelu6 is
    /// 0-6.  The min_range and max_range values are therefore 0.0 and 6.0.
    /// Dequantize on quint8 will take each value, cast to float, and multiply
    /// by 6 / 255.
    /// Note that if quantizedtype is qint8, the operation will additionally add
    /// each value by 128 prior to casting.
    /// 
    /// If the mode is 'MIN_FIRST', then this approach is used:
    /// 
    /// ```c++
    /// num_discrete_values = 1 << (# of bits in T)
    /// range_adjust = num_discrete_values / (num_discrete_values - 1)
    /// range = (range_max - range_min) * range_adjust
    /// range_scale = range / num_discrete_values
    /// const double offset_input = static_cast<double>(input) - lowest_quantized;
    /// result = range_min + ((input - numeric_limits<T>::min()) * range_scale)
    /// ```
    /// 
    /// If the mode is `SCALED`, dequantization is performed by multiplying each
    /// input value by a scaling_factor. (Thus an input of 0 always maps to 0.0).
    /// 
    /// The scaling_factor is determined from `min_range`, `max_range`, and
    /// `narrow_range` in a way that is compatible with `QuantizeAndDequantize{V2|V3}`
    /// and `QuantizeV2`, using the following algorithm:
    /// 
    /// ```c++
    /// 
    ///   const int min_expected_T = std::numeric_limits<T>::min() +
    ///     (narrow_range ? 1 : 0);
    ///   const int max_expected_T = std::numeric_limits<T>::max();
    ///   const float max_expected_T = std::numeric_limits<float>::max();
    /// 
    ///   const float scale_factor =
    ///     (std::numeric_limits<T>::min() == 0) ? (max_range / max_expected_T)
    ///                                          : std::max(min_range / min_expected_T,
    ///                                                     max_range / max_expected_T);
    /// ```
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="min_range"></param>
    /// <param name="max_range"></param>
    /// <param name="mode"></param>
    /// <param name="narrow_range"></param>
    /// <param name="axis"></param>
    /// <param name="dtype">
    /// 
    /// Type of the output tensor. Currently Dequantize supports float and bfloat16.
    /// If 'dtype' is 'bfloat16', it only supports 'MIN_COMBINED' mode.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor dequantize(Tensor input, Tensor min_range, Tensor max_range, string mode = "MIN_COMBINED", bool narrow_range = false, int axis = -1, TF_DataType dtype = TF_DataType.TF_FLOAT, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Dequantize", name) { args = new object[] { input, min_range, max_range }, attrs = new Dictionary<string, object>() { ["mode"] = mode, ["narrow_range"] = narrow_range, ["axis"] = axis, ["dtype"] = dtype } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return dequantize_eager_fallback(input, min_range, max_range, mode: mode, narrow_range: narrow_range, axis: axis, dtype: dtype, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        if (mode is null)
        {
            mode = "MIN_COMBINED";
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["min_range"] = min_range;
        keywords["max_range"] = max_range;
        keywords["mode"] = mode;
        keywords["narrow_range"] = narrow_range;
        keywords["axis"] = axis;
        keywords["dtype"] = dtype;
        var _op = tf.OpDefLib._apply_op_helper("Dequantize", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "mode", _op.get_attr("mode"), "narrow_range", _op._get_attr_bool("narrow_range"), "axis", _op._get_attr_int("axis"), "dtype", _op._get_attr_type("dtype") };
            _execute.record_gradient("Dequantize", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor dequantize_eager_fallback(Tensor input, Tensor min_range, Tensor max_range, string mode, bool narrow_range, int axis, TF_DataType dtype, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, min_range, max_range };
        object[] _attrs = new object[] { "T", input.dtype, "mode", mode, "narrow_range", narrow_range, "axis", axis, "dtype", dtype };
        var _result = _execute.execute("Dequantize", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Dequantize", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns a diagonal tensor with a given diagonal values.
    /// </summary>
    /// <remarks>
    /// 
    /// Given a `diagonal`, this operation returns a tensor with the `diagonal` and
    /// everything else padded with zeros. The diagonal is computed as follows:
    /// 
    /// Assume `diagonal` has dimensions [D1,..., Dk], then the output is a tensor of
    /// rank 2k with dimensions [D1,..., Dk, D1,..., Dk] where:
    /// 
    /// `output[i1,..., ik, i1,..., ik] = diagonal[i1, ..., ik]` and 0 everywhere else.
    /// 
    /// For example:
    /// 
    /// ```
    /// # 'diagonal' is [1, 2, 3, 4]
    /// tf.diag(diagonal) ==> [[1, 0, 0, 0]
    ///                        [0, 2, 0, 0]
    ///                        [0, 0, 3, 0]
    ///                        [0, 0, 0, 4]]
    /// ```
    /// 
    /// </remarks>
    /// <param name="diagonal"></param>
    /// <returns></returns>
    public static Tensor diag(Tensor diagonal, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Diag", name) { args = new object[] { diagonal }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return diag_eager_fallback(diagonal, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["diagonal"] = diagonal;
        var _op = tf.OpDefLib._apply_op_helper("Diag", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Diag", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor diag_eager_fallback(Tensor diagonal, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { diagonal };
        object[] _attrs = new object[] { "T", diagonal.dtype };
        var _result = _execute.execute("Diag", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Diag", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns the diagonal part of the tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation returns a tensor with the `diagonal` part
    /// of the `input`. The `diagonal` part is computed as follows:
    /// 
    /// Assume `input` has dimensions `[D1,..., Dk, D1,..., Dk]`, then the output is a
    /// tensor of rank `k` with dimensions `[D1,..., Dk]` where:
    /// 
    /// `diagonal[i1,..., ik] = input[i1, ..., ik, i1,..., ik]`.
    /// 
    /// For example:
    /// 
    /// ```
    /// # 'input' is [[1, 0, 0, 0]
    ///               [0, 2, 0, 0]
    ///               [0, 0, 3, 0]
    ///               [0, 0, 0, 4]]
    /// 
    /// tf.diag_part(input) ==> [1, 2, 3, 4]
    /// ```
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <returns></returns>
    public static Tensor diag_part(Tensor input, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "DiagPart", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return diag_part_eager_fallback(input, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        var _op = tf.OpDefLib._apply_op_helper("DiagPart", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("DiagPart", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor diag_part_eager_fallback(Tensor input, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "T", input.dtype };
        var _result = _execute.execute("DiagPart", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("DiagPart", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the (possibly normalized) Levenshtein Edit Distance.
    /// </summary>
    /// <remarks>
    /// 
    /// The inputs are variable-length sequences provided by SparseTensors
    ///   (hypothesis_indices, hypothesis_values, hypothesis_shape)
    /// and
    ///   (truth_indices, truth_values, truth_shape).
    /// 
    /// The inputs are:
    /// 
    /// </remarks>
    /// <param name="hypothesis_indices"></param>
    /// <param name="hypothesis_values"></param>
    /// <param name="hypothesis_shape"></param>
    /// <param name="truth_indices"></param>
    /// <param name="truth_values"></param>
    /// <param name="truth_shape"></param>
    /// <param name="normalize">
    /// 
    /// boolean (if true, edit distances are normalized by length of truth).
    /// 
    /// The output is:
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor edit_distance(Tensor hypothesis_indices, Tensor hypothesis_values, Tensor hypothesis_shape, Tensor truth_indices, Tensor truth_values, Tensor truth_shape, bool normalize = true, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "EditDistance", name) { args = new object[] { hypothesis_indices, hypothesis_values, hypothesis_shape, truth_indices, truth_values, truth_shape }, attrs = new Dictionary<string, object>() { ["normalize"] = normalize } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return edit_distance_eager_fallback(hypothesis_indices, hypothesis_values, hypothesis_shape, truth_indices, truth_values, truth_shape, normalize: normalize, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["hypothesis_indices"] = hypothesis_indices;
        keywords["hypothesis_values"] = hypothesis_values;
        keywords["hypothesis_shape"] = hypothesis_shape;
        keywords["truth_indices"] = truth_indices;
        keywords["truth_values"] = truth_values;
        keywords["truth_shape"] = truth_shape;
        keywords["normalize"] = normalize;
        var _op = tf.OpDefLib._apply_op_helper("EditDistance", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "normalize", _op._get_attr_bool("normalize"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("EditDistance", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor edit_distance_eager_fallback(Tensor hypothesis_indices, Tensor hypothesis_values, Tensor hypothesis_shape, Tensor truth_indices, Tensor truth_values, Tensor truth_shape, bool normalize, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { hypothesis_indices, hypothesis_values, hypothesis_shape, truth_indices, truth_values, truth_shape };
        object[] _attrs = new object[] { "normalize", normalize, "T", hypothesis_values.dtype };
        var _result = _execute.execute("EditDistance", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("EditDistance", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    ///
    /// </summary>
    /// <param name="shape"></param>
    /// <param name="dtype"></param>
    /// <param name="init"></param>
    /// <returns></returns>
    public static Tensor empty(Tensor shape, TF_DataType dtype, bool init = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Empty", name) { args = new object[] { shape }, attrs = new Dictionary<string, object>() { ["dtype"] = dtype, ["init"] = init } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return empty_eager_fallback(shape, dtype: dtype, init: init, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["shape"] = shape;
        keywords["dtype"] = dtype;
        keywords["init"] = init;
        var _op = tf.OpDefLib._apply_op_helper("Empty", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "dtype", _op._get_attr_type("dtype"), "init", _op._get_attr_bool("init") };
            _execute.record_gradient("Empty", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor empty_eager_fallback(Tensor shape, TF_DataType dtype, bool init, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { shape };
        object[] _attrs = new object[] { "dtype", dtype, "init", init };
        var _result = _execute.execute("Empty", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Empty", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Ensures that the tensor's shape matches the expected shape.
    /// </summary>
    /// <remarks>
    /// 
    /// Raises an error if the input tensor's shape does not match the specified shape.
    /// Returns the input tensor otherwise.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="shape">
    /// 
    /// The expected (possibly partially specified) shape of the input tensor.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor ensure_shape(Tensor input, Shape shape, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "EnsureShape", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { ["shape"] = shape } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return ensure_shape_eager_fallback(input, shape: shape, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["shape"] = shape;
        var _op = tf.OpDefLib._apply_op_helper("EnsureShape", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "shape", _op.get_attr("shape"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("EnsureShape", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor ensure_shape_eager_fallback(Tensor input, Shape shape, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "shape", shape, "T", input.dtype };
        var _result = _execute.execute("EnsureShape", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("EnsureShape", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Inserts a dimension of 1 into a tensor's shape.
    /// </summary>
    /// <remarks>
    /// 
    /// Given a tensor `input`, this operation inserts a dimension of 1 at the
    /// dimension index `dim` of `input`'s shape. The dimension index `dim` starts at
    /// zero; if you specify a negative number for `dim` it is counted backward from
    /// the end.
    /// 
    /// This operation is useful if you want to add a batch dimension to a single
    /// element. For example, if you have a single image of shape `[height, width,
    /// channels]`, you can make it a batch of 1 image with `expand_dims(image, 0)`,
    /// which will make the shape `[1, height, width, channels]`.
    /// 
    /// Other examples:
    /// 
    /// ```
    /// # 't' is a tensor of shape [2]
    /// shape(expand_dims(t, 0)) ==> [1, 2]
    /// shape(expand_dims(t, 1)) ==> [2, 1]
    /// shape(expand_dims(t, -1)) ==> [2, 1]
    /// 
    /// # 't2' is a tensor of shape [2, 3, 5]
    /// shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
    /// shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
    /// shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
    /// ```
    /// 
    /// This operation requires that:
    /// 
    /// `-1-input.dims() <= dim <= input.dims()`
    /// 
    /// This operation is related to `squeeze()`, which removes dimensions of
    /// size 1.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="dim"></param>
    /// <returns></returns>
    public static Tensor expand_dims(Tensor input, Tensor dim, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ExpandDims", name) { args = new object[] { input, dim }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return expand_dims_eager_fallback(input, dim, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["dim"] = dim;
        var _op = tf.OpDefLib._apply_op_helper("ExpandDims", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tdim", _op._get_attr_type("Tdim") };
            _execute.record_gradient("ExpandDims", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor expand_dims_eager_fallback(Tensor input, Tensor dim, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, dim };
        object[] _attrs = new object[] { "T", input.dtype, "Tdim", dim.dtype };
        var _result = _execute.execute("ExpandDims", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ExpandDims", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Extract `patches` from `images` and put them in the "depth" output dimension.
    /// </summary>
    /// <param name="images"></param>
    /// <param name="ksizes">
    /// 
    /// The size of the sliding window for each dimension of `images`.
    /// 
    /// </param>
    /// <param name="strides">
    /// 
    /// How far the centers of two consecutive patches are in
    /// the images. Must be: `[1, stride_rows, stride_cols, 1]`.
    /// 
    /// </param>
    /// <param name="rates">
    /// 
    /// Must be: `[1, rate_rows, rate_cols, 1]`. This is the
    /// input stride, specifying how far two consecutive patch samples are in the
    /// input. Equivalent to extracting patches with
    /// `patch_sizes_eff = patch_sizes + (patch_sizes - 1) * (rates - 1)`, followed by
    /// subsampling them spatially by a factor of `rates`. This is equivalent to
    /// `rate` in dilated (a.k.a. Atrous) convolutions.
    /// 
    /// </param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor extract_image_patches(Tensor images, int[] ksizes, int[] strides, int[] rates, string padding, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ExtractImagePatches", name) { args = new object[] { images }, attrs = new Dictionary<string, object>() { ["ksizes"] = ksizes, ["strides"] = strides, ["rates"] = rates, ["padding"] = padding } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return extract_image_patches_eager_fallback(images, ksizes: ksizes, strides: strides, rates: rates, padding: padding, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["images"] = images;
        keywords["ksizes"] = ksizes;
        keywords["strides"] = strides;
        keywords["rates"] = rates;
        keywords["padding"] = padding;
        var _op = tf.OpDefLib._apply_op_helper("ExtractImagePatches", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "ksizes", _op.get_attr("ksizes"), "strides", _op.get_attr("strides"), "rates", _op.get_attr("rates"), "T", _op._get_attr_type("T"), "padding", _op.get_attr("padding") };
            _execute.record_gradient("ExtractImagePatches", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor extract_image_patches_eager_fallback(Tensor images, int[] ksizes, int[] strides, int[] rates, string padding, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { images };
        object[] _attrs = new object[] { "ksizes", ksizes, "strides", strides, "rates", rates, "T", images.dtype, "padding", padding };
        var _result = _execute.execute("ExtractImagePatches", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ExtractImagePatches", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Extract `patches` from `input` and put them in the `"depth"` output dimension. 3D extension of `extract_image_patches`.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="ksizes">
    /// 
    /// The size of the sliding window for each dimension of `input`.
    /// 
    /// </param>
    /// <param name="strides">
    /// 
    /// 1-D of length 5. How far the centers of two consecutive patches are in
    /// `input`. Must be: `[1, stride_planes, stride_rows, stride_cols, 1]`.
    /// 
    /// </param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// The size-related attributes are specified as follows:
    /// 
    /// ```python
    /// ksizes = [1, ksize_planes, ksize_rows, ksize_cols, 1]
    /// strides = [1, stride_planes, strides_rows, strides_cols, 1]
    /// ```
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor extract_volume_patches(Tensor input, int[] ksizes, int[] strides, string padding, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ExtractVolumePatches", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { ["ksizes"] = ksizes, ["strides"] = strides, ["padding"] = padding } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return extract_volume_patches_eager_fallback(input, ksizes: ksizes, strides: strides, padding: padding, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["ksizes"] = ksizes;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        var _op = tf.OpDefLib._apply_op_helper("ExtractVolumePatches", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "ksizes", _op.get_attr("ksizes"), "strides", _op.get_attr("strides"), "T", _op._get_attr_type("T"), "padding", _op.get_attr("padding") };
            _execute.record_gradient("ExtractVolumePatches", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor extract_volume_patches_eager_fallback(Tensor input, int[] ksizes, int[] strides, string padding, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "ksizes", ksizes, "strides", strides, "T", input.dtype, "padding", padding };
        var _result = _execute.execute("ExtractVolumePatches", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ExtractVolumePatches", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Fake-quantize the 'inputs' tensor, type float to 'outputs' tensor of same type.
    /// </summary>
    /// <remarks>
    /// 
    /// Attributes
    /// 
    /// *   `[min; max]` define the clamping range for the `inputs` data.
    /// *   `inputs` values are quantized into the quantization range (
    /// `[0; 2^num_bits - 1]` when `narrow_range` is false and `[1; 2^num_bits - 1]`
    /// when it is true) and then de-quantized and output as floats in `[min; max]`
    /// interval.
    /// *   `num_bits` is the bitwidth of the quantization; between 2 and 16, inclusive.
    /// 
    /// Before quantization, `min` and `max` values are adjusted with the following
    /// logic.
    /// It is suggested to have `min <= 0 <= max`. If `0` is not in the range of values,
    /// the behavior can be unexpected:
    /// 
    /// *   If `0 < min < max`: `min_adj = 0` and `max_adj = max - min`.
    /// *   If `min < max < 0`: `min_adj = min - max` and `max_adj = 0`.
    /// *   If `min <= 0 <= max`: `scale = (max - min) / (2^num_bits - 1) `,
    /// `min_adj = scale * round(min / scale)` and `max_adj = max + min_adj - min`.
    /// 
    /// Quantization is called fake since the output is still in floating point.
    /// 
    /// </remarks>
    /// <param name="inputs"></param>
    /// <param name="min"></param>
    /// <param name="max"></param>
    /// <param name="num_bits"></param>
    /// <param name="narrow_range"></param>
    /// <returns></returns>
    public static Tensor fake_quant_with_min_max_args(Tensor inputs, float min = -6f, float max = 6f, int num_bits = 8, bool narrow_range = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "FakeQuantWithMinMaxArgs", name) { args = new object[] { inputs }, attrs = new Dictionary<string, object>() { ["min"] = min, ["max"] = max, ["num_bits"] = num_bits, ["narrow_range"] = narrow_range } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return fake_quant_with_min_max_args_eager_fallback(inputs, min: min, max: max, num_bits: num_bits, narrow_range: narrow_range, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["inputs"] = inputs;
        keywords["min"] = min;
        keywords["max"] = max;
        keywords["num_bits"] = num_bits;
        keywords["narrow_range"] = narrow_range;
        var _op = tf.OpDefLib._apply_op_helper("FakeQuantWithMinMaxArgs", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "min", _op.get_attr("min"), "max", _op.get_attr("max"), "num_bits", _op._get_attr_int("num_bits"), "narrow_range", _op._get_attr_bool("narrow_range") };
            _execute.record_gradient("FakeQuantWithMinMaxArgs", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor fake_quant_with_min_max_args_eager_fallback(Tensor inputs, float min, float max, int num_bits, bool narrow_range, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { inputs };
        object[] _attrs = new object[] { "min", min, "max", max, "num_bits", num_bits, "narrow_range", narrow_range };
        var _result = _execute.execute("FakeQuantWithMinMaxArgs", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("FakeQuantWithMinMaxArgs", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Compute gradients for a FakeQuantWithMinMaxArgs operation.
    /// </summary>
    /// <param name="gradients"></param>
    /// <param name="inputs"></param>
    /// <param name="min"></param>
    /// <param name="max"></param>
    /// <param name="num_bits"></param>
    /// <param name="narrow_range"></param>
    /// <returns></returns>
    public static Tensor fake_quant_with_min_max_args_gradient(Tensor gradients, Tensor inputs, float min = -6f, float max = 6f, int num_bits = 8, bool narrow_range = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "FakeQuantWithMinMaxArgsGradient", name) { args = new object[] { gradients, inputs }, attrs = new Dictionary<string, object>() { ["min"] = min, ["max"] = max, ["num_bits"] = num_bits, ["narrow_range"] = narrow_range } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return fake_quant_with_min_max_args_gradient_eager_fallback(gradients, inputs, min: min, max: max, num_bits: num_bits, narrow_range: narrow_range, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["gradients"] = gradients;
        keywords["inputs"] = inputs;
        keywords["min"] = min;
        keywords["max"] = max;
        keywords["num_bits"] = num_bits;
        keywords["narrow_range"] = narrow_range;
        var _op = tf.OpDefLib._apply_op_helper("FakeQuantWithMinMaxArgsGradient", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "min", _op.get_attr("min"), "max", _op.get_attr("max"), "num_bits", _op._get_attr_int("num_bits"), "narrow_range", _op._get_attr_bool("narrow_range") };
            _execute.record_gradient("FakeQuantWithMinMaxArgsGradient", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor fake_quant_with_min_max_args_gradient_eager_fallback(Tensor gradients, Tensor inputs, float min, float max, int num_bits, bool narrow_range, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { gradients, inputs };
        object[] _attrs = new object[] { "min", min, "max", max, "num_bits", num_bits, "narrow_range", narrow_range };
        var _result = _execute.execute("FakeQuantWithMinMaxArgsGradient", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("FakeQuantWithMinMaxArgsGradient", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Fake-quantize the 'inputs' tensor of type float via global float scalars
    /// </summary>
    /// <remarks>
    /// 
    /// Fake-quantize the `inputs` tensor of type float via global float scalars
    /// `min` and `max` to `outputs` tensor of same shape as `inputs`.
    /// 
    /// Attributes
    /// 
    /// *   `[min; max]` define the clamping range for the `inputs` data.
    /// *   `inputs` values are quantized into the quantization range (
    /// `[0; 2^num_bits - 1]` when `narrow_range` is false and `[1; 2^num_bits - 1]`
    /// when it is true) and then de-quantized and output as floats in `[min; max]`
    /// interval.
    /// *   `num_bits` is the bitwidth of the quantization; between 2 and 16, inclusive.
    /// 
    /// Before quantization, `min` and `max` values are adjusted with the following
    /// logic.
    /// It is suggested to have `min <= 0 <= max`. If `0` is not in the range of values,
    /// the behavior can be unexpected:
    /// 
    /// *   If `0 < min < max`: `min_adj = 0` and `max_adj = max - min`.
    /// *   If `min < max < 0`: `min_adj = min - max` and `max_adj = 0`.
    /// *   If `min <= 0 <= max`: `scale = (max - min) / (2^num_bits - 1) `,
    /// `min_adj = scale * round(min / scale)` and `max_adj = max + min_adj - min`.
    /// 
    /// This operation has a gradient and thus allows for training `min` and `max`
    /// values.
    /// 
    /// </remarks>
    /// <param name="inputs"></param>
    /// <param name="min"></param>
    /// <param name="max"></param>
    /// <param name="num_bits"></param>
    /// <param name="narrow_range"></param>
    /// <returns></returns>
    public static Tensor fake_quant_with_min_max_vars(Tensor inputs, Tensor min, Tensor max, int num_bits = 8, bool narrow_range = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "FakeQuantWithMinMaxVars", name) { args = new object[] { inputs, min, max }, attrs = new Dictionary<string, object>() { ["num_bits"] = num_bits, ["narrow_range"] = narrow_range } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return fake_quant_with_min_max_vars_eager_fallback(inputs, min, max, num_bits: num_bits, narrow_range: narrow_range, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["inputs"] = inputs;
        keywords["min"] = min;
        keywords["max"] = max;
        keywords["num_bits"] = num_bits;
        keywords["narrow_range"] = narrow_range;
        var _op = tf.OpDefLib._apply_op_helper("FakeQuantWithMinMaxVars", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "num_bits", _op._get_attr_int("num_bits"), "narrow_range", _op._get_attr_bool("narrow_range") };
            _execute.record_gradient("FakeQuantWithMinMaxVars", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor fake_quant_with_min_max_vars_eager_fallback(Tensor inputs, Tensor min, Tensor max, int num_bits, bool narrow_range, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { inputs, min, max };
        object[] _attrs = new object[] { "num_bits", num_bits, "narrow_range", narrow_range };
        var _result = _execute.execute("FakeQuantWithMinMaxVars", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("FakeQuantWithMinMaxVars", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Compute gradients for a FakeQuantWithMinMaxVars operation.
    /// </summary>
    /// <param name="gradients"></param>
    /// <param name="inputs"></param>
    /// <param name="min"></param>
    /// <param name="max"></param>
    /// <param name="num_bits">
    /// 
    /// The bitwidth of the quantization; between 2 and 8, inclusive.
    /// 
    /// </param>
    /// <param name="narrow_range">
    /// 
    /// Whether to quantize into 2^num_bits - 1 distinct values.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] fake_quant_with_min_max_vars_gradient(Tensor gradients, Tensor inputs, Tensor min, Tensor max, int num_bits = 8, bool narrow_range = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "FakeQuantWithMinMaxVarsGradient", name) { args = new object[] { gradients, inputs, min, max }, attrs = new Dictionary<string, object>() { ["num_bits"] = num_bits, ["narrow_range"] = narrow_range } });
                return _fast_path_result;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return fake_quant_with_min_max_vars_gradient_eager_fallback(gradients, inputs, min, max, num_bits: num_bits, narrow_range: narrow_range, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["gradients"] = gradients;
        keywords["inputs"] = inputs;
        keywords["min"] = min;
        keywords["max"] = max;
        keywords["num_bits"] = num_bits;
        keywords["narrow_range"] = narrow_range;
        var _op = tf.OpDefLib._apply_op_helper("FakeQuantWithMinMaxVarsGradient", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "num_bits", _op._get_attr_int("num_bits"), "narrow_range", _op._get_attr_bool("narrow_range") };
            _execute.record_gradient("FakeQuantWithMinMaxVarsGradient", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] fake_quant_with_min_max_vars_gradient_eager_fallback(Tensor gradients, Tensor inputs, Tensor min, Tensor max, int num_bits, bool narrow_range, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { gradients, inputs, min, max };
        object[] _attrs = new object[] { "num_bits", num_bits, "narrow_range", narrow_range };
        var _result = _execute.execute("FakeQuantWithMinMaxVarsGradient", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("FakeQuantWithMinMaxVarsGradient", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Fake-quantize the 'inputs' tensor of type float via per-channel floats
    /// </summary>
    /// <remarks>
    /// 
    /// Fake-quantize the `inputs` tensor of type float per-channel and one of the
    /// shapes: `[d]`, `[b, d]` `[b, h, w, d]` via per-channel floats `min` and `max`
    /// of shape `[d]` to `outputs` tensor of same shape as `inputs`.
    /// 
    /// Attributes
    /// 
    /// *   `[min; max]` define the clamping range for the `inputs` data.
    /// *   `inputs` values are quantized into the quantization range (
    /// `[0; 2^num_bits - 1]` when `narrow_range` is false and `[1; 2^num_bits - 1]`
    /// when it is true) and then de-quantized and output as floats in `[min; max]`
    /// interval.
    /// *   `num_bits` is the bitwidth of the quantization; between 2 and 16, inclusive.
    /// 
    /// Before quantization, `min` and `max` values are adjusted with the following
    /// logic.
    /// It is suggested to have `min <= 0 <= max`. If `0` is not in the range of values,
    /// the behavior can be unexpected:
    /// 
    /// *   If `0 < min < max`: `min_adj = 0` and `max_adj = max - min`.
    /// *   If `min < max < 0`: `min_adj = min - max` and `max_adj = 0`.
    /// *   If `min <= 0 <= max`: `scale = (max - min) / (2^num_bits - 1) `,
    /// `min_adj = scale * round(min / scale)` and `max_adj = max + min_adj - min`.
    /// 
    /// This operation has a gradient and thus allows for training `min` and `max`
    /// values.
    /// 
    /// </remarks>
    /// <param name="inputs"></param>
    /// <param name="min"></param>
    /// <param name="max"></param>
    /// <param name="num_bits"></param>
    /// <param name="narrow_range"></param>
    /// <returns></returns>
    public static Tensor fake_quant_with_min_max_vars_per_channel(Tensor inputs, Tensor min, Tensor max, int num_bits = 8, bool narrow_range = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "FakeQuantWithMinMaxVarsPerChannel", name) { args = new object[] { inputs, min, max }, attrs = new Dictionary<string, object>() { ["num_bits"] = num_bits, ["narrow_range"] = narrow_range } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return fake_quant_with_min_max_vars_per_channel_eager_fallback(inputs, min, max, num_bits: num_bits, narrow_range: narrow_range, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["inputs"] = inputs;
        keywords["min"] = min;
        keywords["max"] = max;
        keywords["num_bits"] = num_bits;
        keywords["narrow_range"] = narrow_range;
        var _op = tf.OpDefLib._apply_op_helper("FakeQuantWithMinMaxVarsPerChannel", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "num_bits", _op._get_attr_int("num_bits"), "narrow_range", _op._get_attr_bool("narrow_range") };
            _execute.record_gradient("FakeQuantWithMinMaxVarsPerChannel", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor fake_quant_with_min_max_vars_per_channel_eager_fallback(Tensor inputs, Tensor min, Tensor max, int num_bits, bool narrow_range, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { inputs, min, max };
        object[] _attrs = new object[] { "num_bits", num_bits, "narrow_range", narrow_range };
        var _result = _execute.execute("FakeQuantWithMinMaxVarsPerChannel", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("FakeQuantWithMinMaxVarsPerChannel", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Compute gradients for a FakeQuantWithMinMaxVarsPerChannel operation.
    /// </summary>
    /// <param name="gradients"></param>
    /// <param name="inputs"></param>
    /// <param name="min"></param>
    /// <param name="max"></param>
    /// <param name="num_bits">
    /// 
    /// The bitwidth of the quantization; between 2 and 16, inclusive.
    /// 
    /// </param>
    /// <param name="narrow_range">
    /// 
    /// Whether to quantize into 2^num_bits - 1 distinct values.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] fake_quant_with_min_max_vars_per_channel_gradient(Tensor gradients, Tensor inputs, Tensor min, Tensor max, int num_bits = 8, bool narrow_range = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "FakeQuantWithMinMaxVarsPerChannelGradient", name) { args = new object[] { gradients, inputs, min, max }, attrs = new Dictionary<string, object>() { ["num_bits"] = num_bits, ["narrow_range"] = narrow_range } });
                return _fast_path_result;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return fake_quant_with_min_max_vars_per_channel_gradient_eager_fallback(gradients, inputs, min, max, num_bits: num_bits, narrow_range: narrow_range, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["gradients"] = gradients;
        keywords["inputs"] = inputs;
        keywords["min"] = min;
        keywords["max"] = max;
        keywords["num_bits"] = num_bits;
        keywords["narrow_range"] = narrow_range;
        var _op = tf.OpDefLib._apply_op_helper("FakeQuantWithMinMaxVarsPerChannelGradient", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "num_bits", _op._get_attr_int("num_bits"), "narrow_range", _op._get_attr_bool("narrow_range") };
            _execute.record_gradient("FakeQuantWithMinMaxVarsPerChannelGradient", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] fake_quant_with_min_max_vars_per_channel_gradient_eager_fallback(Tensor gradients, Tensor inputs, Tensor min, Tensor max, int num_bits, bool narrow_range, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { gradients, inputs, min, max };
        object[] _attrs = new object[] { "num_bits", num_bits, "narrow_range", narrow_range };
        var _result = _execute.execute("FakeQuantWithMinMaxVarsPerChannelGradient", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("FakeQuantWithMinMaxVarsPerChannelGradient", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Creates a tensor filled with a scalar value.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation creates a tensor of shape `dims` and fills it with `value`.
    /// 
    /// For example:
    /// 
    /// ```
    /// # Output tensor has shape [2, 3].
    /// fill([2, 3], 9) ==> [[9, 9, 9]
    ///                      [9, 9, 9]]
    /// ```
    /// 
    /// `tf.fill` differs from `tf.constant` in a few ways:
    /// 
    /// *   `tf.fill` only supports scalar contents, whereas `tf.constant` supports
    ///     Tensor values.
    /// *   `tf.fill` creates an Op in the computation graph that constructs the actual
    ///     Tensor value at runtime. This is in contrast to `tf.constant` which embeds
    ///     the entire Tensor into the graph with a `Const` node.
    /// *   Because `tf.fill` evaluates at graph runtime, it supports dynamic shapes
    ///     based on other runtime Tensors, unlike `tf.constant`.
    /// 
    /// </remarks>
    /// <param name="dims"></param>
    /// <param name="value"></param>
    /// <returns></returns>
    public static Tensor fill(Tensor dims, Tensor value, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Fill", name) { args = new object[] { dims, value }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return fill_eager_fallback(dims, value, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["dims"] = dims;
        keywords["value"] = value;
        var _op = tf.OpDefLib._apply_op_helper("Fill", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "index_type", _op._get_attr_type("index_type") };
            _execute.record_gradient("Fill", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor fill_eager_fallback(Tensor dims, Tensor value, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { dims, value };
        object[] _attrs = new object[] { "T", value.dtype, "index_type", dims.dtype };
        var _result = _execute.execute("Fill", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Fill", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Generates fingerprint values.
    /// </summary>
    /// <remarks>
    /// 
    /// Generates fingerprint values of `data`.
    /// 
    /// Fingerprint op considers the first dimension of `data` as the batch dimension,
    /// and `output[i]` contains the fingerprint value generated from contents in
    /// `data[i, ...]` for all `i`.
    /// 
    /// Fingerprint op writes fingerprint values as byte arrays. For example, the
    /// default method `farmhash64` generates a 64-bit fingerprint value at a time.
    /// This 8-byte value is written out as an `uint8` array of size 8, in little-endian
    /// order.
    /// 
    /// For example, suppose that `data` has data type `DT_INT32` and shape (2, 3, 4),
    /// and that the fingerprint method is `farmhash64`. In this case, the output shape
    /// is (2, 8), where 2 is the batch dimension size of `data`, and 8 is the size of
    /// each fingerprint value in bytes. `output[0, :]` is generated from 12 integers in
    /// `data[0, :, :]` and similarly `output[1, :]` is generated from other 12 integers
    /// in `data[1, :, :]`.
    /// 
    /// Note that this op fingerprints the raw underlying buffer, and it does not
    /// fingerprint Tensor's metadata such as data type and/or shape. For example, the
    /// fingerprint values are invariant under reshapes and bitcasts as long as the
    /// batch dimension remain the same:
    /// 
    /// ```
    /// Fingerprint(data) == Fingerprint(Reshape(data, ...))
    /// Fingerprint(data) == Fingerprint(Bitcast(data, ...))
    /// ```
    /// 
    /// For string data, one should expect `Fingerprint(data) !=
    /// Fingerprint(ReduceJoin(data))` in general.
    /// 
    /// </remarks>
    /// <param name="data"></param>
    /// <param name="method"></param>
    /// <returns></returns>
    public static Tensor fingerprint(Tensor data, Tensor method, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Fingerprint", name) { args = new object[] { data, method }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return fingerprint_eager_fallback(data, method, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["data"] = data;
        keywords["method"] = method;
        var _op = tf.OpDefLib._apply_op_helper("Fingerprint", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Fingerprint", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor fingerprint_eager_fallback(Tensor data, Tensor method, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { data, method };
        object[] _attrs = new object[] { "T", data.dtype };
        var _result = _execute.execute("Fingerprint", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Fingerprint", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Gather slices from `params` according to `indices`.
    /// </summary>
    /// <remarks>
    /// 
    /// `indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
    /// Produces an output tensor with shape `indices.shape + params.shape[1:]` where:
    /// 
    /// ```python
    ///     # Scalar indices
    ///     output[:, ..., :] = params[indices, :, ... :]
    /// 
    ///     # Vector indices
    ///     output[i, :, ..., :] = params[indices[i], :, ... :]
    /// 
    ///     # Higher rank indices
    ///     output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]
    /// ```
    /// 
    /// If `indices` is a permutation and `len(indices) == params.shape[0]` then
    /// this operation will permute `params` accordingly.
    /// 
    /// `validate_indices`: DEPRECATED. If this operation is assigned to CPU, values in
    /// `indices` are always validated to be within range. If assigned to GPU,
    /// out-of-bound indices result in safe but unspecified behavior, which may include
    /// raising an error.
    /// 
    /// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
    /// <img style="width:100%" src="https://www.tensorflow.org/images/Gather.png" alt>
    /// </div>
    /// 
    /// </remarks>
    /// <param name="params_"></param>
    /// <param name="indices"></param>
    /// <param name="validate_indices"></param>
    /// <returns></returns>
    public static Tensor gather(Tensor params_, Tensor indices, bool validate_indices = true, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Gather", name) { args = new object[] { params_, indices }, attrs = new Dictionary<string, object>() { ["validate_indices"] = validate_indices } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return gather_eager_fallback(params_, indices, validate_indices: validate_indices, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["params"] = params_;
        keywords["indices"] = indices;
        keywords["validate_indices"] = validate_indices;
        var _op = tf.OpDefLib._apply_op_helper("Gather", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "validate_indices", _op._get_attr_bool("validate_indices"), "Tparams", _op._get_attr_type("Tparams"), "Tindices", _op._get_attr_type("Tindices") };
            _execute.record_gradient("Gather", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor gather_eager_fallback(Tensor params_, Tensor indices, bool validate_indices, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { params_, indices };
        object[] _attrs = new object[] { "validate_indices", validate_indices, "Tparams", params_.dtype, "Tindices", indices.dtype };
        var _result = _execute.execute("Gather", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Gather", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Gather slices from `params` into a Tensor with shape specified by `indices`.
    /// </summary>
    /// <remarks>
    /// 
    /// `indices` is a K-dimensional integer tensor, best thought of as a
    /// (K-1)-dimensional tensor of indices into `params`, where each element defines a
    /// slice of `params`:
    /// 
    ///     output[\(i_0, ..., i_{K-2}\)] = params[indices[\(i_0, ..., i_{K-2}\)]]
    /// 
    /// Whereas in `tf.gather` `indices` defines slices into the `axis`
    /// dimension of `params`, in `tf.gather_nd`, `indices` defines slices into the
    /// first `N` dimensions of `params`, where `N = indices.shape[-1]`.
    /// 
    /// The last dimension of `indices` can be at most the rank of
    /// `params`:
    /// 
    ///     indices.shape[-1] <= params.rank
    /// 
    /// The last dimension of `indices` corresponds to elements
    /// (if `indices.shape[-1] == params.rank`) or slices
    /// (if `indices.shape[-1] < params.rank`) along dimension `indices.shape[-1]`
    /// of `params`.  The output tensor has shape
    /// 
    ///     indices.shape[:-1] + params.shape[indices.shape[-1]:]
    /// 
    /// Note that on CPU, if an out of bound index is found, an error is returned.
    /// On GPU, if an out of bound index is found, a 0 is stored in the
    /// corresponding output value.
    /// 
    /// Some examples below.
    /// 
    /// Simple indexing into a matrix:
    /// 
    /// ```python
    ///     indices = [[0, 0], [1, 1]]
    ///     params = [['a', 'b'], ['c', 'd']]
    ///     output = ['a', 'd']
    /// ```
    /// 
    /// Slice indexing into a matrix:
    /// 
    /// ```python
    ///     indices = [[1], [0]]
    ///     params = [['a', 'b'], ['c', 'd']]
    ///     output = [['c', 'd'], ['a', 'b']]
    /// ```
    /// 
    /// Indexing into a 3-tensor:
    /// 
    /// ```python
    ///     indices = [[1]]
    ///     params = [[['a0', 'b0'], ['c0', 'd0']],
    ///               [['a1', 'b1'], ['c1', 'd1']]]
    ///     output = [[['a1', 'b1'], ['c1', 'd1']]]
    /// 
    /// 
    ///     indices = [[0, 1], [1, 0]]
    ///     params = [[['a0', 'b0'], ['c0', 'd0']],
    ///               [['a1', 'b1'], ['c1', 'd1']]]
    ///     output = [['c0', 'd0'], ['a1', 'b1']]
    /// 
    /// 
    ///     indices = [[0, 0, 1], [1, 0, 1]]
    ///     params = [[['a0', 'b0'], ['c0', 'd0']],
    ///               [['a1', 'b1'], ['c1', 'd1']]]
    ///     output = ['b0', 'b1']
    /// ```
    /// 
    /// Batched indexing into a matrix:
    /// 
    /// ```python
    ///     indices = [[[0, 0]], [[0, 1]]]
    ///     params = [['a', 'b'], ['c', 'd']]
    ///     output = [['a'], ['b']]
    /// ```
    /// 
    /// Batched slice indexing into a matrix:
    /// 
    /// ```python
    ///     indices = [[[1]], [[0]]]
    ///     params = [['a', 'b'], ['c', 'd']]
    ///     output = [[['c', 'd']], [['a', 'b']]]
    /// ```
    /// 
    /// Batched indexing into a 3-tensor:
    /// 
    /// ```python
    ///     indices = [[[1]], [[0]]]
    ///     params = [[['a0', 'b0'], ['c0', 'd0']],
    ///               [['a1', 'b1'], ['c1', 'd1']]]
    ///     output = [[[['a1', 'b1'], ['c1', 'd1']]],
    ///               [[['a0', 'b0'], ['c0', 'd0']]]]
    /// 
    ///     indices = [[[0, 1], [1, 0]], [[0, 0], [1, 1]]]
    ///     params = [[['a0', 'b0'], ['c0', 'd0']],
    ///               [['a1', 'b1'], ['c1', 'd1']]]
    ///     output = [[['c0', 'd0'], ['a1', 'b1']],
    ///               [['a0', 'b0'], ['c1', 'd1']]]
    /// 
    /// 
    ///     indices = [[[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 0]]]
    ///     params = [[['a0', 'b0'], ['c0', 'd0']],
    ///               [['a1', 'b1'], ['c1', 'd1']]]
    ///     output = [['b0', 'b1'], ['d0', 'c1']]
    /// ```
    /// 
    /// See also `tf.gather` and `tf.batch_gather`.
    /// 
    /// </remarks>
    /// <param name="params_"></param>
    /// <param name="indices"></param>
    /// <returns></returns>
    public static Tensor gather_nd(Tensor params_, Tensor indices, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "GatherNd", name) { args = new object[] { params_, indices }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return gather_nd_eager_fallback(params_, indices, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["params"] = params_;
        keywords["indices"] = indices;
        var _op = tf.OpDefLib._apply_op_helper("GatherNd", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tparams", _op._get_attr_type("Tparams"), "Tindices", _op._get_attr_type("Tindices") };
            _execute.record_gradient("GatherNd", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor gather_nd_eager_fallback(Tensor params_, Tensor indices, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { params_, indices };
        object[] _attrs = new object[] { "Tparams", params_.dtype, "Tindices", indices.dtype };
        var _result = _execute.execute("GatherNd", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("GatherNd", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Gather slices from `params` axis `axis` according to `indices`.
    /// </summary>
    /// <remarks>
    /// 
    /// `indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
    /// Produces an output tensor with shape `params.shape[:axis] +
    /// indices.shape[batch_dims:] + params.shape[axis + 1:]` where:
    /// 
    /// ```python
    ///     # Scalar indices (output is rank(params) - 1).
    ///     output[a_0, ..., a_n, b_0, ..., b_n] =
    ///       params[a_0, ..., a_n, indices, b_0, ..., b_n]
    /// 
    ///     # Vector indices (output is rank(params)).
    ///     output[a_0, ..., a_n, i, b_0, ..., b_n] =
    ///       params[a_0, ..., a_n, indices[i], b_0, ..., b_n]
    /// 
    ///     # Higher rank indices (output is rank(params) + rank(indices) - 1).
    ///     output[a_0, ..., a_n, i, ..., j, b_0, ... b_n] =
    ///       params[a_0, ..., a_n, indices[i, ..., j], b_0, ..., b_n]
    /// ```
    /// 
    /// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
    /// <img style="width:100%" src="https://www.tensorflow.org/images/Gather.png" alt>
    /// </div>
    /// 
    /// Note that on CPU, if an out of bound index is found, an error is returned.
    /// On GPU, if an out of bound index is found, a 0 is stored in the
    /// corresponding output value.
    /// 
    /// See also `tf.batch_gather` and `tf.gather_nd`.
    /// 
    /// </remarks>
    /// <param name="params_"></param>
    /// <param name="indices"></param>
    /// <param name="axis"></param>
    /// <param name="batch_dims"></param>
    /// <returns></returns>
    public static Tensor gather_v2(Tensor params_, Tensor indices, Tensor axis, int batch_dims = 0, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "GatherV2", name) { args = new object[] { params_, indices, axis }, attrs = new Dictionary<string, object>() { ["batch_dims"] = batch_dims } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return gather_v2_eager_fallback(params_, indices, axis, batch_dims: batch_dims, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["params"] = params_;
        keywords["indices"] = indices;
        keywords["axis"] = axis;
        keywords["batch_dims"] = batch_dims;
        var _op = tf.OpDefLib._apply_op_helper("GatherV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "batch_dims", _op._get_attr_int("batch_dims"), "Tparams", _op._get_attr_type("Tparams"), "Tindices", _op._get_attr_type("Tindices"), "Taxis", _op._get_attr_type("Taxis") };
            _execute.record_gradient("GatherV2", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor gather_v2_eager_fallback(Tensor params_, Tensor indices, Tensor axis, int batch_dims, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { params_, indices, axis };
        object[] _attrs = new object[] { "batch_dims", batch_dims, "Tparams", params_.dtype, "Tindices", indices.dtype, "Taxis", axis.dtype };
        var _result = _execute.execute("GatherV2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("GatherV2", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Gives a guarantee to the TF runtime that the input tensor is a constant.
    /// </summary>
    /// <remarks>
    /// 
    /// The runtime is then free to make optimizations based on this.
    /// 
    /// Only accepts value typed tensors as inputs and rejects resource variable handles
    /// as input.
    /// 
    /// Returns the input tensor without modification.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <returns></returns>
    public static Tensor guarantee_const(Tensor input, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "GuaranteeConst", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return guarantee_const_eager_fallback(input, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        var _op = tf.OpDefLib._apply_op_helper("GuaranteeConst", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("GuaranteeConst", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor guarantee_const_eager_fallback(Tensor input, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "T", input.dtype };
        var _result = _execute.execute("GuaranteeConst", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("GuaranteeConst", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Return a tensor with the same shape and contents as the input tensor or value.
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public static Tensor identity(Tensor input, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Identity", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return identity_eager_fallback(input, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        var _op = tf.OpDefLib._apply_op_helper("Identity", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Identity", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor identity_eager_fallback(Tensor input, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "T", input.dtype };
        var _result = _execute.execute("Identity", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Identity", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns a list of tensors with the same shapes and contents as the input
    /// </summary>
    /// <remarks>
    /// 
    /// tensors.
    /// 
    /// This op can be used to override the gradient for complicated functions. For
    /// example, suppose y = f(x) and we wish to apply a custom function g for backprop
    /// such that dx = g(dy). In Python,
    /// 
    /// ```python
    /// with tf.get_default_graph().gradient_override_map(
    ///     {'IdentityN': 'OverrideGradientWithG'}):
    ///   y, _ = identity_n([f(x), x])
    /// 
    /// @tf.RegisterGradient('OverrideGradientWithG')
    /// def ApplyG(op, dy, _):
    ///   return [None, g(dy)]  # Do not backprop to f(x).
    /// ```
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <returns></returns>
    public static Tensor[] identity_n(Tensors input, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "IdentityN", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return identity_n_eager_fallback(input, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        var _op = tf.OpDefLib._apply_op_helper("IdentityN", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op.get_attr("T") };
            _execute.record_gradient("IdentityN", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] identity_n_eager_fallback(Tensor input, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { };
        var _result = _execute.execute("IdentityN", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("IdentityN", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Returns immutable tensor from memory region.
    /// </summary>
    /// <remarks>
    /// 
    /// The current implementation memmaps the tensor from a file.
    /// 
    /// </remarks>
    /// <param name="dtype">
    /// 
    /// Type of the returned tensor.
    /// 
    /// </param>
    /// <param name="shape">
    /// 
    /// Shape of the returned tensor.
    /// 
    /// </param>
    /// <param name="memory_region_name">
    /// 
    /// Name of readonly memory region used by the tensor, see
    /// NewReadOnlyMemoryRegionFromFile in tensorflow::Env.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor immutable_const(TF_DataType dtype, Shape shape, string memory_region_name, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ImmutableConst", name) { args = new object[] { }, attrs = new Dictionary<string, object>() { ["dtype"] = dtype, ["shape"] = shape, ["memory_region_name"] = memory_region_name } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return immutable_const_eager_fallback(dtype: dtype, shape: shape, memory_region_name: memory_region_name, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["dtype"] = dtype;
        keywords["shape"] = shape;
        keywords["memory_region_name"] = memory_region_name;
        var _op = tf.OpDefLib._apply_op_helper("ImmutableConst", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "dtype", _op._get_attr_type("dtype"), "shape", _op.get_attr("shape"), "memory_region_name", _op.get_attr("memory_region_name") };
            _execute.record_gradient("ImmutableConst", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor immutable_const_eager_fallback(TF_DataType dtype, Shape shape, string memory_region_name, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { };
        object[] _attrs = new object[] { "dtype", dtype, "shape", shape, "memory_region_name", memory_region_name };
        var _result = _execute.execute("ImmutableConst", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ImmutableConst", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    ///
    /// </summary>
    /// <param name="x"></param>
    /// <param name="i"></param>
    /// <param name="v"></param>
    /// <returns></returns>
    public static Tensor inplace_add(Tensor x, Tensor i, Tensor v, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "InplaceAdd", name) { args = new object[] { x, i, v }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return inplace_add_eager_fallback(x, i, v, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["i"] = i;
        keywords["v"] = v;
        var _op = tf.OpDefLib._apply_op_helper("InplaceAdd", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("InplaceAdd", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor inplace_add_eager_fallback(Tensor x, Tensor i, Tensor v, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, i, v };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("InplaceAdd", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("InplaceAdd", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    ///
    /// </summary>
    /// <param name="x"></param>
    /// <param name="i"></param>
    /// <param name="v"></param>
    /// <returns></returns>
    public static Tensor inplace_sub(Tensor x, Tensor i, Tensor v, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "InplaceSub", name) { args = new object[] { x, i, v }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return inplace_sub_eager_fallback(x, i, v, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["i"] = i;
        keywords["v"] = v;
        var _op = tf.OpDefLib._apply_op_helper("InplaceSub", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("InplaceSub", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor inplace_sub_eager_fallback(Tensor x, Tensor i, Tensor v, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, i, v };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("InplaceSub", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("InplaceSub", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    ///
    /// </summary>
    /// <param name="x"></param>
    /// <param name="i"></param>
    /// <param name="v"></param>
    /// <returns></returns>
    public static Tensor inplace_update(Tensor x, Tensor i, Tensor v, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "InplaceUpdate", name) { args = new object[] { x, i, v }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return inplace_update_eager_fallback(x, i, v, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["i"] = i;
        keywords["v"] = v;
        var _op = tf.OpDefLib._apply_op_helper("InplaceUpdate", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("InplaceUpdate", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor inplace_update_eager_fallback(Tensor x, Tensor i, Tensor v, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, i, v };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("InplaceUpdate", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("InplaceUpdate", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the inverse permutation of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation computes the inverse of an index permutation. It takes a 1-D
    /// integer tensor `x`, which represents the indices of a zero-based array, and
    /// swaps each value with its index position. In other words, for an output tensor
    /// `y` and an input tensor `x`, this operation computes the following:
    /// 
    /// `y[x[i]] = i for i in [0, 1, ..., len(x) - 1]`
    /// 
    /// The values must include 0. There can be no duplicate values or negative values.
    /// 
    /// For example:
    /// 
    /// ```
    /// # tensor `x` is [3, 4, 0, 2, 1]
    /// invert_permutation(x) ==> [2, 4, 3, 0, 1]
    /// ```
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor invert_permutation(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "InvertPermutation", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return invert_permutation_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("InvertPermutation", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("InvertPermutation", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor invert_permutation_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("InvertPermutation", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("InvertPermutation", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the difference between two lists of numbers or strings.
    /// </summary>
    /// <remarks>
    /// 
    /// Given a list `x` and a list `y`, this operation returns a list `out` that
    /// represents all values that are in `x` but not in `y`. The returned list `out`
    /// is sorted in the same order that the numbers appear in `x` (duplicates are
    /// preserved). This operation also returns a list `idx` that represents the
    /// position of each `out` element in `x`. In other words:
    /// 
    /// `out[i] = x[idx[i]] for i in [0, 1, ..., len(out) - 1]`
    /// 
    /// For example, given this input:
    /// 
    /// ```
    /// x = [1, 2, 3, 4, 5, 6]
    /// y = [1, 3, 5]
    /// ```
    /// 
    /// This operation would return:
    /// 
    /// ```
    /// out ==> [2, 4, 6]
    /// idx ==> [1, 3, 5]
    /// ```
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <param name="out_idx"></param>
    /// <returns></returns>
    public static Tensor[] list_diff(Tensor x, Tensor y, TF_DataType out_idx = TF_DataType.TF_INT32, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ListDiff", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { ["out_idx"] = out_idx } });
                return _fast_path_result;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return list_diff_eager_fallback(x, y, out_idx: out_idx, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        keywords["out_idx"] = out_idx;
        var _op = tf.OpDefLib._apply_op_helper("ListDiff", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "out_idx", _op._get_attr_type("out_idx") };
            _execute.record_gradient("ListDiff", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] list_diff_eager_fallback(Tensor x, Tensor y, TF_DataType out_idx, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { "T", x.dtype, "out_idx", out_idx };
        var _result = _execute.execute("ListDiff", 2, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ListDiff", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Applies lower_bound(sorted_search_values, values) along each row.
    /// </summary>
    /// <remarks>
    /// 
    /// Each set of rows with the same index in (sorted_inputs, values) is treated
    /// independently.  The resulting row is the equivalent of calling
    /// `np.searchsorted(sorted_inputs, values, side='left')`.
    /// 
    /// The result is not a global index to the entire
    /// `Tensor`, but rather just the index in the last dimension.
    /// 
    /// A 2-D example:
    ///   sorted_sequence = [[0, 3, 9, 9, 10],
    ///                      [1, 2, 3, 4, 5]]
    ///   values = [[2, 4, 9],
    ///             [0, 2, 6]]
    /// 
    ///   result = LowerBound(sorted_sequence, values)
    /// 
    ///   result == [[1, 2, 2],
    ///              [0, 1, 5]]
    /// 
    /// </remarks>
    /// <param name="sorted_inputs"></param>
    /// <param name="values"></param>
    /// <param name="out_type"></param>
    /// <returns></returns>
    public static Tensor lower_bound(Tensor sorted_inputs, Tensor values, TF_DataType out_type = TF_DataType.TF_INT32, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "LowerBound", name) { args = new object[] { sorted_inputs, values }, attrs = new Dictionary<string, object>() { ["out_type"] = out_type } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return lower_bound_eager_fallback(sorted_inputs, values, out_type: out_type, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["sorted_inputs"] = sorted_inputs;
        keywords["values"] = values;
        keywords["out_type"] = out_type;
        var _op = tf.OpDefLib._apply_op_helper("LowerBound", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "out_type", _op._get_attr_type("out_type") };
            _execute.record_gradient("LowerBound", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor lower_bound_eager_fallback(Tensor sorted_inputs, Tensor values, TF_DataType out_type, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { sorted_inputs, values };
        object[] _attrs = new object[] { "T", sorted_inputs.dtype, "out_type", out_type };
        var _result = _execute.execute("LowerBound", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("LowerBound", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Copy a tensor setting everything outside a central band in each innermost matrix to zero.
    /// </summary>
    /// <remarks>
    /// 
    /// The `band` part is computed as follows:
    /// Assume `input` has `k` dimensions `[I, J, K, ..., M, N]`, then the output is a
    /// tensor with the same shape where
    /// 
    /// `band[i, j, k, ..., m, n] = in_band(m, n) * input[i, j, k, ..., m, n]`.
    /// 
    /// The indicator function
    /// 
    /// `in_band(m, n) = (num_lower < 0 || (m-n) <= num_lower)) &&
    ///                  (num_upper < 0 || (n-m) <= num_upper)`.
    /// 
    /// For example:
    /// 
    /// ```
    /// # if 'input' is [[ 0,  1,  2, 3]
    /// #                [-1,  0,  1, 2]
    /// #                [-2, -1,  0, 1]
    /// #                [-3, -2, -1, 0]],
    /// 
    /// tf.linalg.band_part(input, 1, -1) ==> [[ 0,  1,  2, 3]
    ///                                        [-1,  0,  1, 2]
    ///                                        [ 0, -1,  0, 1]
    ///                                        [ 0,  0, -1, 0]],
    /// 
    /// tf.linalg.band_part(input, 2, 1) ==> [[ 0,  1,  0, 0]
    ///                                       [-1,  0,  1, 0]
    ///                                       [-2, -1,  0, 1]
    ///                                       [ 0, -2, -1, 0]]
    /// ```
    /// 
    /// Useful special cases:
    /// 
    /// ```
    ///  tf.linalg.band_part(input, 0, -1) ==> Upper triangular part.
    ///  tf.linalg.band_part(input, -1, 0) ==> Lower triangular part.
    ///  tf.linalg.band_part(input, 0, 0) ==> Diagonal.
    /// ```
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="num_lower"></param>
    /// <param name="num_upper"></param>
    /// <returns></returns>
    public static Tensor matrix_band_part(Tensor input, Tensor num_lower, Tensor num_upper, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "MatrixBandPart", name) { args = new object[] { input, num_lower, num_upper }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return matrix_band_part_eager_fallback(input, num_lower, num_upper, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["num_lower"] = num_lower;
        keywords["num_upper"] = num_upper;
        var _op = tf.OpDefLib._apply_op_helper("MatrixBandPart", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tindex", _op._get_attr_type("Tindex") };
            _execute.record_gradient("MatrixBandPart", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor matrix_band_part_eager_fallback(Tensor input, Tensor num_lower, Tensor num_upper, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, num_lower, num_upper };
        object[] _attrs = new object[] { "T", input.dtype, "Tindex", num_lower.dtype };
        var _result = _execute.execute("MatrixBandPart", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("MatrixBandPart", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns a batched diagonal tensor with a given batched diagonal values.
    /// </summary>
    /// <remarks>
    /// 
    /// Given a `diagonal`, this operation returns a tensor with the `diagonal` and
    /// everything else padded with zeros. The diagonal is computed as follows:
    /// 
    /// Assume `diagonal` has `k` dimensions `[I, J, K, ..., N]`, then the output is a
    /// tensor of rank `k+1` with dimensions [I, J, K, ..., N, N]` where:
    /// 
    /// `output[i, j, k, ..., m, n] = 1{m=n} * diagonal[i, j, k, ..., n]`.
    /// 
    /// For example:
    /// 
    /// ```
    /// # 'diagonal' is [[1, 2, 3, 4], [5, 6, 7, 8]]
    /// 
    /// and diagonal.shape = (2, 4)
    /// 
    /// tf.matrix_diag(diagonal) ==> [[[1, 0, 0, 0]
    ///                                      [0, 2, 0, 0]
    ///                                      [0, 0, 3, 0]
    ///                                      [0, 0, 0, 4]],
    ///                                     [[5, 0, 0, 0]
    ///                                      [0, 6, 0, 0]
    ///                                      [0, 0, 7, 0]
    ///                                      [0, 0, 0, 8]]]
    /// 
    /// which has shape (2, 4, 4)
    /// ```
    /// 
    /// </remarks>
    /// <param name="diagonal"></param>
    /// <returns></returns>
    public static Tensor matrix_diag(Tensor diagonal, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "MatrixDiag", name) { args = new object[] { diagonal }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return matrix_diag_eager_fallback(diagonal, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["diagonal"] = diagonal;
        var _op = tf.OpDefLib._apply_op_helper("MatrixDiag", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("MatrixDiag", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor matrix_diag_eager_fallback(Tensor diagonal, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { diagonal };
        object[] _attrs = new object[] { "T", diagonal.dtype };
        var _result = _execute.execute("MatrixDiag", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("MatrixDiag", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns the batched diagonal part of a batched tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation returns a tensor with the `diagonal` part
    /// of the batched `input`. The `diagonal` part is computed as follows:
    /// 
    /// Assume `input` has `k` dimensions `[I, J, K, ..., M, N]`, then the output is a
    /// tensor of rank `k - 1` with dimensions `[I, J, K, ..., min(M, N)]` where:
    /// 
    /// `diagonal[i, j, k, ..., n] = input[i, j, k, ..., n, n]`.
    /// 
    /// The input must be at least a matrix.
    /// 
    /// For example:
    /// 
    /// ```
    /// # 'input' is [[[1, 0, 0, 0]
    ///                [0, 2, 0, 0]
    ///                [0, 0, 3, 0]
    ///                [0, 0, 0, 4]],
    ///               [[5, 0, 0, 0]
    ///                [0, 6, 0, 0]
    ///                [0, 0, 7, 0]
    ///                [0, 0, 0, 8]]]
    /// 
    /// and input.shape = (2, 4, 4)
    /// 
    /// tf.matrix_diag_part(input) ==> [[1, 2, 3, 4], [5, 6, 7, 8]]
    /// 
    /// which has shape (2, 4)
    /// ```
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <returns></returns>
    public static Tensor matrix_diag_part(Tensor input, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "MatrixDiagPart", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return matrix_diag_part_eager_fallback(input, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        var _op = tf.OpDefLib._apply_op_helper("MatrixDiagPart", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("MatrixDiagPart", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor matrix_diag_part_eager_fallback(Tensor input, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "T", input.dtype };
        var _result = _execute.execute("MatrixDiagPart", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("MatrixDiagPart", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns the batched diagonal part of a batched tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Returns a tensor with the `k[0]`-th to `k[1]`-th diagonals of the batched
    /// `input`.
    /// 
    /// Assume `input` has `r` dimensions `[I, J, ..., L, M, N]`.
    /// Let `max_diag_len` be the maximum length among all diagonals to be extracted,
    /// `max_diag_len = min(M + min(k[1], 0), N + min(-k[0], 0))`
    /// Let `num_diags` be the number of diagonals to extract,
    /// `num_diags = k[1] - k[0] + 1`.
    /// 
    /// If `num_diags == 1`, the output tensor is of rank `r - 1` with shape
    /// `[I, J, ..., L, max_diag_len]` and values:
    /// 
    /// ```
    /// diagonal[i, j, ..., l, n]
    ///   = input[i, j, ..., l, n+y, n+x] ; if 0 <= n+y < M and 0 <= n+x < N,
    ///     padding_value                 ; otherwise.
    /// ```
    /// where `y = max(-k[1], 0)`, `x = max(k[1], 0)`.
    /// 
    /// Otherwise, the output tensor has rank `r` with dimensions
    /// `[I, J, ..., L, num_diags, max_diag_len]` with values:
    /// 
    /// ```
    /// diagonal[i, j, ..., l, m, n]
    ///   = input[i, j, ..., l, n+y, n+x] ; if 0 <= n+y < M and 0 <= n+x < N,
    ///     padding_value                 ; otherwise.
    /// ```
    /// where `d = k[1] - m`, `y = max(-d, 0)`, and `x = max(d, 0)`.
    /// 
    /// The input must be at least a matrix.
    /// 
    /// For example:
    /// 
    /// ```
    /// input = np.array([[[1, 2, 3, 4],  # Input shape: (2, 3, 4)
    ///                    [5, 6, 7, 8],
    ///                    [9, 8, 7, 6]],
    ///                   [[5, 4, 3, 2],
    ///                    [1, 2, 3, 4],
    ///                    [5, 6, 7, 8]]])
    /// 
    /// # A main diagonal from each batch.
    /// tf.matrix_diag_part(input) ==> [[1, 6, 7],  # Output shape: (2, 3)
    ///                                 [5, 2, 7]]
    /// 
    /// # A superdiagonal from each batch.
    /// tf.matrix_diag_part(input, k = 1)
    ///   ==> [[2, 7, 6],  # Output shape: (2, 3)
    ///        [4, 3, 8]]
    /// 
    /// # A tridiagonal band from each batch.
    /// tf.matrix_diag_part(input, k = (-1, 1))
    ///   ==> [[[2, 7, 6],  # Output shape: (2, 3, 3)
    ///         [1, 6, 7],
    ///         [5, 8, 0]],
    ///        [[4, 3, 8],
    ///         [5, 2, 7],
    ///         [1, 6, 0]]]
    /// 
    /// # Padding value = 9
    /// tf.matrix_diag_part(input, k = (1, 3), padding_value = 9)
    ///   ==> [[[4, 9, 9],  # Output shape: (2, 3, 3)
    ///         [3, 8, 9],
    ///         [2, 7, 6]],
    ///        [[2, 9, 9],
    ///         [3, 4, 9],
    ///         [4, 3, 8]]]
    /// ```
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="k"></param>
    /// <param name="padding_value"></param>
    /// <returns></returns>
    public static Tensor matrix_diag_part_v2(Tensor input, Tensor k, Tensor padding_value, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "MatrixDiagPartV2", name) { args = new object[] { input, k, padding_value }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return matrix_diag_part_v2_eager_fallback(input, k, padding_value, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["k"] = k;
        keywords["padding_value"] = padding_value;
        var _op = tf.OpDefLib._apply_op_helper("MatrixDiagPartV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("MatrixDiagPartV2", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor matrix_diag_part_v2_eager_fallback(Tensor input, Tensor k, Tensor padding_value, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, k, padding_value };
        object[] _attrs = new object[] { "T", input.dtype };
        var _result = _execute.execute("MatrixDiagPartV2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("MatrixDiagPartV2", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns the batched diagonal part of a batched tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Returns a tensor with the `k[0]`-th to `k[1]`-th diagonals of the batched
    /// `input`.
    /// 
    /// Assume `input` has `r` dimensions `[I, J, ..., L, M, N]`.
    /// Let `max_diag_len` be the maximum length among all diagonals to be extracted,
    /// `max_diag_len = min(M + min(k[1], 0), N + min(-k[0], 0))`
    /// Let `num_diags` be the number of diagonals to extract,
    /// `num_diags = k[1] - k[0] + 1`.
    /// 
    /// If `num_diags == 1`, the output tensor is of rank `r - 1` with shape
    /// `[I, J, ..., L, max_diag_len]` and values:
    /// 
    /// ```
    /// diagonal[i, j, ..., l, n]
    ///   = input[i, j, ..., l, n+y, n+x] ; if 0 <= n+y < M and 0 <= n+x < N,
    ///     padding_value                 ; otherwise.
    /// ```
    /// where `y = max(-k[1], 0)`, `x = max(k[1], 0)`.
    /// 
    /// Otherwise, the output tensor has rank `r` with dimensions
    /// `[I, J, ..., L, num_diags, max_diag_len]` with values:
    /// 
    /// ```
    /// diagonal[i, j, ..., l, m, n]
    ///   = input[i, j, ..., l, n+y, n+x] ; if 0 <= n+y < M and 0 <= n+x < N,
    ///     padding_value                 ; otherwise.
    /// ```
    /// where `d = k[1] - m`, `y = max(-d, 0) - offset`, and `x = max(d, 0) - offset`.
    /// 
    /// `offset` is zero except when the alignment of the diagonal is to the right.
    /// ```
    /// offset = max_diag_len - diag_len(d) ; if (`align` in {RIGHT_LEFT, RIGHT_RIGHT}
    ///                                            and `d >= 0`) or
    ///                                          (`align` in {LEFT_RIGHT, RIGHT_RIGHT}
    ///                                            and `d <= 0`)
    ///          0                          ; otherwise
    /// ```
    /// where `diag_len(d) = min(cols - max(d, 0), rows + min(d, 0))`.
    /// 
    /// The input must be at least a matrix.
    /// 
    /// For example:
    /// 
    /// ```
    /// input = np.array([[[1, 2, 3, 4],  # Input shape: (2, 3, 4)
    ///                    [5, 6, 7, 8],
    ///                    [9, 8, 7, 6]],
    ///                   [[5, 4, 3, 2],
    ///                    [1, 2, 3, 4],
    ///                    [5, 6, 7, 8]]])
    /// 
    /// # A main diagonal from each batch.
    /// tf.matrix_diag_part(input) ==> [[1, 6, 7],  # Output shape: (2, 3)
    ///                                 [5, 2, 7]]
    /// 
    /// # A superdiagonal from each batch.
    /// tf.matrix_diag_part(input, k = 1)
    ///   ==> [[2, 7, 6],  # Output shape: (2, 3)
    ///        [4, 3, 8]]
    /// 
    /// # A band from each batch.
    /// tf.matrix_diag_part(input, k = (-1, 2))
    ///   ==> [[[0, 3, 8],  # Output shape: (2, 4, 3)
    ///         [2, 7, 6],
    ///         [1, 6, 7],
    ///         [5, 8, 0]],
    ///        [[0, 3, 4],
    ///         [4, 3, 8],
    ///         [5, 2, 7],
    ///         [1, 6, 0]]]
    /// 
    /// # LEFT_RIGHT alignment.
    /// tf.matrix_diag_part(input, k = (-1, 2), align="LEFT_RIGHT")
    ///   ==> [[[3, 8, 0],  # Output shape: (2, 4, 3)
    ///         [2, 7, 6],
    ///         [1, 6, 7],
    ///         [0, 5, 8]],
    ///        [[3, 4, 0],
    ///         [4, 3, 8],
    ///         [5, 2, 7],
    ///         [0, 1, 6]]]
    /// 
    /// # max_diag_len can be shorter than the main diagonal.
    /// tf.matrix_diag_part(input, k = (-2, -1))
    ///   ==> [[[5, 8],
    ///         [9, 0]],
    ///        [[1, 6],
    ///         [5, 0]]]
    /// 
    /// # padding_value = 9
    /// tf.matrix_diag_part(input, k = (1, 3), padding_value = 9)
    ///   ==> [[[9, 9, 4],  # Output shape: (2, 3, 3)
    ///         [9, 3, 8],
    ///         [2, 7, 6]],
    ///        [[9, 9, 2],
    ///         [9, 3, 4],
    ///         [4, 3, 8]]]
    /// 
    /// ```
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="k"></param>
    /// <param name="padding_value"></param>
    /// <param name="align">
    /// 
    /// Some diagonals are shorter than `max_diag_len` and need to be padded. `align` is
    /// a string specifying how superdiagonals and subdiagonals should be aligned,
    /// respectively. There are four possible alignments: "RIGHT_LEFT" (default),
    /// "LEFT_RIGHT", "LEFT_LEFT", and "RIGHT_RIGHT". "RIGHT_LEFT" aligns superdiagonals
    /// to the right (left-pads the row) and subdiagonals to the left (right-pads the
    /// row). It is the packing format LAPACK uses. cuSPARSE uses "LEFT_RIGHT", which is
    /// the opposite alignment.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor matrix_diag_part_v3(Tensor input, Tensor k, Tensor padding_value, string align = "RIGHT_LEFT", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "MatrixDiagPartV3", name) { args = new object[] { input, k, padding_value }, attrs = new Dictionary<string, object>() { ["align"] = align } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return matrix_diag_part_v3_eager_fallback(input, k, padding_value, align: align, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        if (align is null)
        {
            align = "RIGHT_LEFT";
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["k"] = k;
        keywords["padding_value"] = padding_value;
        keywords["align"] = align;
        var _op = tf.OpDefLib._apply_op_helper("MatrixDiagPartV3", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "align", _op.get_attr("align") };
            _execute.record_gradient("MatrixDiagPartV3", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor matrix_diag_part_v3_eager_fallback(Tensor input, Tensor k, Tensor padding_value, string align, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, k, padding_value };
        object[] _attrs = new object[] { "T", input.dtype, "align", align };
        var _result = _execute.execute("MatrixDiagPartV3", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("MatrixDiagPartV3", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns a batched diagonal tensor with given batched diagonal values.
    /// </summary>
    /// <remarks>
    /// 
    /// Returns a tensor with the contents in `diagonal` as `k[0]`-th to `k[1]`-th
    /// diagonals of a matrix, with everything else padded with `padding`. `num_rows`
    /// and `num_cols` specify the dimension of the innermost matrix of the output. If
    /// both are not specified, the op assumes the innermost matrix is square and infers
    /// its size from `k` and the innermost dimension of `diagonal`. If only one of them
    /// is specified, the op assumes the unspecified value is the smallest possible
    /// based on other criteria.
    /// 
    /// Let `diagonal` have `r` dimensions `[I, J, ..., L, M, N]`. The output tensor has
    /// rank `r+1` with shape `[I, J, ..., L, M, num_rows, num_cols]` when only one
    /// diagonal is given (`k` is an integer or `k[0] == k[1]`). Otherwise, it has rank
    /// `r` with shape `[I, J, ..., L, num_rows, num_cols]`.
    /// 
    /// The second innermost dimension of `diagonal` has double meaning.
    /// When `k` is scalar or `k[0] == k[1]`, `M` is part of the batch size
    /// [I, J, ..., M], and the output tensor is:
    /// 
    /// ```
    /// output[i, j, ..., l, m, n]
    ///   = diagonal[i, j, ..., l, n-max(d_upper, 0)] ; if n - m == d_upper
    ///     padding_value                             ; otherwise
    /// ```
    /// 
    /// Otherwise, `M` is treated as the number of diagonals for the matrix in the
    /// same batch (`M = k[1]-k[0]+1`), and the output tensor is:
    /// 
    /// ```
    /// output[i, j, ..., l, m, n]
    ///   = diagonal[i, j, ..., l, diag_index, index_in_diag] ; if k[0] <= d <= k[1]
    ///     padding_value                                     ; otherwise
    /// ```
    /// where `d = n - m`, `diag_index = k[1] - d`, and `index_in_diag = n - max(d, 0)`.
    /// 
    /// For example:
    /// 
    /// ```
    /// # The main diagonal.
    /// diagonal = np.array([[1, 2, 3, 4],            # Input shape: (2, 4)
    ///                      [5, 6, 7, 8]])
    /// tf.matrix_diag(diagonal) ==> [[[1, 0, 0, 0],  # Output shape: (2, 4, 4)
    ///                                [0, 2, 0, 0],
    ///                                [0, 0, 3, 0],
    ///                                [0, 0, 0, 4]],
    ///                               [[5, 0, 0, 0],
    ///                                [0, 6, 0, 0],
    ///                                [0, 0, 7, 0],
    ///                                [0, 0, 0, 8]]]
    /// 
    /// # A superdiagonal (per batch).
    /// diagonal = np.array([[1, 2, 3],  # Input shape: (2, 3)
    ///                      [4, 5, 6]])
    /// tf.matrix_diag(diagonal, k = 1)
    ///   ==> [[[0, 1, 0, 0],  # Output shape: (2, 4, 4)
    ///         [0, 0, 2, 0],
    ///         [0, 0, 0, 3],
    ///         [0, 0, 0, 0]],
    ///        [[0, 4, 0, 0],
    ///         [0, 0, 5, 0],
    ///         [0, 0, 0, 6],
    ///         [0, 0, 0, 0]]]
    /// 
    /// # A band of diagonals.
    /// diagonals = np.array([[[1, 2, 3],  # Input shape: (2, 2, 3)
    ///                        [4, 5, 0]],
    ///                       [[6, 7, 9],
    ///                        [9, 1, 0]]])
    /// tf.matrix_diag(diagonals, k = (-1, 0))
    ///   ==> [[[1, 0, 0],  # Output shape: (2, 3, 3)
    ///         [4, 2, 0],
    ///         [0, 5, 3]],
    ///        [[6, 0, 0],
    ///         [9, 7, 0],
    ///         [0, 1, 9]]]
    /// 
    /// # Rectangular matrix.
    /// diagonal = np.array([1, 2])  # Input shape: (2)
    /// tf.matrix_diag(diagonal, k = -1, num_rows = 3, num_cols = 4)
    ///   ==> [[0, 0, 0, 0],  # Output shape: (3, 4)
    ///        [1, 0, 0, 0],
    ///        [0, 2, 0, 0]]
    /// 
    /// # Rectangular matrix with inferred num_cols and padding_value = 9.
    /// tf.matrix_diag(diagonal, k = -1, num_rows = 3, padding_value = 9)
    ///   ==> [[9, 9],  # Output shape: (3, 2)
    ///        [1, 9],
    ///        [9, 2]]
    /// ```
    /// 
    /// </remarks>
    /// <param name="diagonal"></param>
    /// <param name="k"></param>
    /// <param name="num_rows"></param>
    /// <param name="num_cols"></param>
    /// <param name="padding_value"></param>
    /// <returns></returns>
    public static Tensor matrix_diag_v2(Tensor diagonal, Tensor k, Tensor num_rows, Tensor num_cols, Tensor padding_value, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "MatrixDiagV2", name) { args = new object[] { diagonal, k, num_rows, num_cols, padding_value }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return matrix_diag_v2_eager_fallback(diagonal, k, num_rows, num_cols, padding_value, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["diagonal"] = diagonal;
        keywords["k"] = k;
        keywords["num_rows"] = num_rows;
        keywords["num_cols"] = num_cols;
        keywords["padding_value"] = padding_value;
        var _op = tf.OpDefLib._apply_op_helper("MatrixDiagV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("MatrixDiagV2", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor matrix_diag_v2_eager_fallback(Tensor diagonal, Tensor k, Tensor num_rows, Tensor num_cols, Tensor padding_value, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { diagonal, k, num_rows, num_cols, padding_value };
        object[] _attrs = new object[] { "T", diagonal.dtype };
        var _result = _execute.execute("MatrixDiagV2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("MatrixDiagV2", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns a batched diagonal tensor with given batched diagonal values.
    /// </summary>
    /// <remarks>
    /// 
    /// Returns a tensor with the contents in `diagonal` as `k[0]`-th to `k[1]`-th
    /// diagonals of a matrix, with everything else padded with `padding`. `num_rows`
    /// and `num_cols` specify the dimension of the innermost matrix of the output. If
    /// both are not specified, the op assumes the innermost matrix is square and infers
    /// its size from `k` and the innermost dimension of `diagonal`. If only one of them
    /// is specified, the op assumes the unspecified value is the smallest possible
    /// based on other criteria.
    /// 
    /// Let `diagonal` have `r` dimensions `[I, J, ..., L, M, N]`. The output tensor has
    /// rank `r+1` with shape `[I, J, ..., L, M, num_rows, num_cols]` when only one
    /// diagonal is given (`k` is an integer or `k[0] == k[1]`). Otherwise, it has rank
    /// `r` with shape `[I, J, ..., L, num_rows, num_cols]`.
    /// 
    /// The second innermost dimension of `diagonal` has double meaning.
    /// When `k` is scalar or `k[0] == k[1]`, `M` is part of the batch size
    /// [I, J, ..., M], and the output tensor is:
    /// 
    /// ```
    /// output[i, j, ..., l, m, n]
    ///   = diagonal[i, j, ..., l, n-max(d_upper, 0)] ; if n - m == d_upper
    ///     padding_value                             ; otherwise
    /// ```
    /// 
    /// Otherwise, `M` is treated as the number of diagonals for the matrix in the
    /// same batch (`M = k[1]-k[0]+1`), and the output tensor is:
    /// 
    /// ```
    /// output[i, j, ..., l, m, n]
    ///   = diagonal[i, j, ..., l, diag_index, index_in_diag] ; if k[0] <= d <= k[1]
    ///     padding_value                                     ; otherwise
    /// ```
    /// where `d = n - m`, `diag_index = [k] - d`, and
    /// `index_in_diag = n - max(d, 0) + offset`.
    /// 
    /// `offset` is zero except when the alignment of the diagonal is to the right.
    /// ```
    /// offset = max_diag_len - diag_len(d) ; if (`align` in {RIGHT_LEFT, RIGHT_RIGHT}
    ///                                            and `d >= 0`) or
    ///                                          (`align` in {LEFT_RIGHT, RIGHT_RIGHT}
    ///                                            and `d <= 0`)
    ///          0                          ; otherwise
    /// ```
    /// where `diag_len(d) = min(cols - max(d, 0), rows + min(d, 0))`.
    /// 
    /// For example:
    /// 
    /// ```
    /// # The main diagonal.
    /// diagonal = np.array([[1, 2, 3, 4],            # Input shape: (2, 4)
    ///                      [5, 6, 7, 8]])
    /// tf.matrix_diag(diagonal) ==> [[[1, 0, 0, 0],  # Output shape: (2, 4, 4)
    ///                                [0, 2, 0, 0],
    ///                                [0, 0, 3, 0],
    ///                                [0, 0, 0, 4]],
    ///                               [[5, 0, 0, 0],
    ///                                [0, 6, 0, 0],
    ///                                [0, 0, 7, 0],
    ///                                [0, 0, 0, 8]]]
    /// 
    /// # A superdiagonal (per batch).
    /// diagonal = np.array([[1, 2, 3],  # Input shape: (2, 3)
    ///                      [4, 5, 6]])
    /// tf.matrix_diag(diagonal, k = 1)
    ///   ==> [[[0, 1, 0, 0],  # Output shape: (2, 4, 4)
    ///         [0, 0, 2, 0],
    ///         [0, 0, 0, 3],
    ///         [0, 0, 0, 0]],
    ///        [[0, 4, 0, 0],
    ///         [0, 0, 5, 0],
    ///         [0, 0, 0, 6],
    ///         [0, 0, 0, 0]]]
    /// 
    /// # A tridiagonal band (per batch).
    /// diagonals = np.array([[[0, 8, 9],  # Input shape: (2, 2, 3)
    ///                        [1, 2, 3],
    ///                        [4, 5, 0]],
    ///                       [[0, 2, 3],
    ///                        [6, 7, 9],
    ///                        [9, 1, 0]]])
    /// tf.matrix_diag(diagonals, k = (-1, 1))
    ///   ==> [[[1, 8, 0],  # Output shape: (2, 3, 3)
    ///         [4, 2, 9],
    ///         [0, 5, 3]],
    ///        [[6, 2, 0],
    ///         [9, 7, 3],
    ///         [0, 1, 9]]]
    /// 
    /// # LEFT_RIGHT alignment.
    /// diagonals = np.array([[[8, 9, 0],  # Input shape: (2, 2, 3)
    ///                        [1, 2, 3],
    ///                        [0, 4, 5]],
    ///                       [[2, 3, 0],
    ///                        [6, 7, 9],
    ///                        [0, 9, 1]]])
    /// tf.matrix_diag(diagonals, k = (-1, 1), align="LEFT_RIGHT")
    ///   ==> [[[1, 8, 0],  # Output shape: (2, 3, 3)
    ///         [4, 2, 9],
    ///         [0, 5, 3]],
    ///        [[6, 2, 0],
    ///         [9, 7, 3],
    ///         [0, 1, 9]]]
    /// 
    /// # Rectangular matrix.
    /// diagonal = np.array([1, 2])  # Input shape: (2)
    /// tf.matrix_diag(diagonal, k = -1, num_rows = 3, num_cols = 4)
    ///   ==> [[0, 0, 0, 0],  # Output shape: (3, 4)
    ///        [1, 0, 0, 0],
    ///        [0, 2, 0, 0]]
    /// 
    /// # Rectangular matrix with inferred num_cols and padding_value = 9.
    /// tf.matrix_diag(diagonal, k = -1, num_rows = 3, padding_value = 9)
    ///   ==> [[9, 9],  # Output shape: (3, 2)
    ///        [1, 9],
    ///        [9, 2]]
    /// 
    /// ```
    /// 
    /// </remarks>
    /// <param name="diagonal"></param>
    /// <param name="k"></param>
    /// <param name="num_rows"></param>
    /// <param name="num_cols"></param>
    /// <param name="padding_value"></param>
    /// <param name="align">
    /// 
    /// Some diagonals are shorter than `max_diag_len` and need to be padded. `align` is
    /// a string specifying how superdiagonals and subdiagonals should be aligned,
    /// respectively. There are four possible alignments: "RIGHT_LEFT" (default),
    /// "LEFT_RIGHT", "LEFT_LEFT", and "RIGHT_RIGHT". "RIGHT_LEFT" aligns superdiagonals
    /// to the right (left-pads the row) and subdiagonals to the left (right-pads the
    /// row). It is the packing format LAPACK uses. cuSPARSE uses "LEFT_RIGHT", which is
    /// the opposite alignment.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor matrix_diag_v3(Tensor diagonal, Tensor k, Tensor num_rows, Tensor num_cols, Tensor padding_value, string align = "RIGHT_LEFT", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "MatrixDiagV3", name) { args = new object[] { diagonal, k, num_rows, num_cols, padding_value }, attrs = new Dictionary<string, object>() { ["align"] = align } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return matrix_diag_v3_eager_fallback(diagonal, k, num_rows, num_cols, padding_value, align: align, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        if (align is null)
        {
            align = "RIGHT_LEFT";
        }
        Dictionary<string, object> keywords = new();
        keywords["diagonal"] = diagonal;
        keywords["k"] = k;
        keywords["num_rows"] = num_rows;
        keywords["num_cols"] = num_cols;
        keywords["padding_value"] = padding_value;
        keywords["align"] = align;
        var _op = tf.OpDefLib._apply_op_helper("MatrixDiagV3", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "align", _op.get_attr("align") };
            _execute.record_gradient("MatrixDiagV3", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor matrix_diag_v3_eager_fallback(Tensor diagonal, Tensor k, Tensor num_rows, Tensor num_cols, Tensor padding_value, string align, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { diagonal, k, num_rows, num_cols, padding_value };
        object[] _attrs = new object[] { "T", diagonal.dtype, "align", align };
        var _result = _execute.execute("MatrixDiagV3", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("MatrixDiagV3", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns a batched matrix tensor with new batched diagonal values.
    /// </summary>
    /// <remarks>
    /// 
    /// Given `input` and `diagonal`, this operation returns a tensor with the
    /// same shape and values as `input`, except for the main diagonal of the
    /// innermost matrices.  These will be overwritten by the values in `diagonal`.
    /// 
    /// The output is computed as follows:
    /// 
    /// Assume `input` has `k+1` dimensions `[I, J, K, ..., M, N]` and `diagonal` has
    /// `k` dimensions `[I, J, K, ..., min(M, N)]`.  Then the output is a
    /// tensor of rank `k+1` with dimensions `[I, J, K, ..., M, N]` where:
    /// 
    ///   * `output[i, j, k, ..., m, n] = diagonal[i, j, k, ..., n]` for `m == n`.
    ///   * `output[i, j, k, ..., m, n] = input[i, j, k, ..., m, n]` for `m != n`.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="diagonal"></param>
    /// <returns></returns>
    public static Tensor matrix_set_diag(Tensor input, Tensor diagonal, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "MatrixSetDiag", name) { args = new object[] { input, diagonal }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return matrix_set_diag_eager_fallback(input, diagonal, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["diagonal"] = diagonal;
        var _op = tf.OpDefLib._apply_op_helper("MatrixSetDiag", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("MatrixSetDiag", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor matrix_set_diag_eager_fallback(Tensor input, Tensor diagonal, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, diagonal };
        object[] _attrs = new object[] { "T", input.dtype };
        var _result = _execute.execute("MatrixSetDiag", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("MatrixSetDiag", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns a batched matrix tensor with new batched diagonal values.
    /// </summary>
    /// <remarks>
    /// 
    /// Given `input` and `diagonal`, this operation returns a tensor with the
    /// same shape and values as `input`, except for the specified diagonals of the
    /// innermost matrices. These will be overwritten by the values in `diagonal`.
    /// 
    /// `input` has `r+1` dimensions `[I, J, ..., L, M, N]`. When `k` is scalar or
    /// `k[0] == k[1]`, `diagonal` has `r` dimensions `[I, J, ..., L, max_diag_len]`.
    /// Otherwise, it has `r+1` dimensions `[I, J, ..., L, num_diags, max_diag_len]`.
    /// `num_diags` is the number of diagonals, `num_diags = k[1] - k[0] + 1`.
    /// `max_diag_len` is the longest diagonal in the range `[k[0], k[1]]`,
    /// `max_diag_len = min(M + min(k[1], 0), N + min(-k[0], 0))`
    /// 
    /// The output is a tensor of rank `k+1` with dimensions `[I, J, ..., L, M, N]`.
    /// If `k` is scalar or `k[0] == k[1]`:
    /// 
    /// ```
    /// output[i, j, ..., l, m, n]
    ///   = diagonal[i, j, ..., l, n-max(k[1], 0)] ; if n - m == k[1]
    ///     input[i, j, ..., l, m, n]              ; otherwise
    /// ```
    /// 
    /// Otherwise,
    /// 
    /// ```
    /// output[i, j, ..., l, m, n]
    ///   = diagonal[i, j, ..., l, diag_index, index_in_diag] ; if k[0] <= d <= k[1]
    ///     input[i, j, ..., l, m, n]                         ; otherwise
    /// ```
    /// where `d = n - m`, `diag_index = k[1] - d`, and `index_in_diag = n - max(d, 0)`.
    /// 
    /// For example:
    /// 
    /// ```
    /// # The main diagonal.
    /// input = np.array([[[7, 7, 7, 7],              # Input shape: (2, 3, 4)
    ///                    [7, 7, 7, 7],
    ///                    [7, 7, 7, 7]],
    ///                   [[7, 7, 7, 7],
    ///                    [7, 7, 7, 7],
    ///                    [7, 7, 7, 7]]])
    /// diagonal = np.array([[1, 2, 3],               # Diagonal shape: (2, 3)
    ///                      [4, 5, 6]])
    /// tf.matrix_set_diag(diagonal) ==> [[[1, 7, 7, 7],  # Output shape: (2, 3, 4)
    ///                                    [7, 2, 7, 7],
    ///                                    [7, 7, 3, 7]],
    ///                                   [[4, 7, 7, 7],
    ///                                    [7, 5, 7, 7],
    ///                                    [7, 7, 6, 7]]]
    /// 
    /// # A superdiagonal (per batch).
    /// tf.matrix_set_diag(diagonal, k = 1)
    ///   ==> [[[7, 1, 7, 7],  # Output shape: (2, 3, 4)
    ///         [7, 7, 2, 7],
    ///         [7, 7, 7, 3]],
    ///        [[7, 4, 7, 7],
    ///         [7, 7, 5, 7],
    ///         [7, 7, 7, 6]]]
    /// 
    /// # A band of diagonals.
    /// diagonals = np.array([[[1, 2, 3],  # Diagonal shape: (2, 2, 3)
    ///                        [4, 5, 0]],
    ///                       [[6, 1, 2],
    ///                        [3, 4, 0]]])
    /// tf.matrix_set_diag(diagonals, k = (-1, 0))
    ///   ==> [[[1, 7, 7, 7],  # Output shape: (2, 3, 4)
    ///         [4, 2, 7, 7],
    ///         [0, 5, 3, 7]],
    ///        [[6, 7, 7, 7],
    ///         [3, 1, 7, 7],
    ///         [7, 4, 2, 7]]]
    /// 
    /// ```
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="diagonal"></param>
    /// <param name="k"></param>
    /// <returns></returns>
    public static Tensor matrix_set_diag_v2(Tensor input, Tensor diagonal, Tensor k, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "MatrixSetDiagV2", name) { args = new object[] { input, diagonal, k }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return matrix_set_diag_v2_eager_fallback(input, diagonal, k, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["diagonal"] = diagonal;
        keywords["k"] = k;
        var _op = tf.OpDefLib._apply_op_helper("MatrixSetDiagV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("MatrixSetDiagV2", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor matrix_set_diag_v2_eager_fallback(Tensor input, Tensor diagonal, Tensor k, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, diagonal, k };
        object[] _attrs = new object[] { "T", input.dtype };
        var _result = _execute.execute("MatrixSetDiagV2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("MatrixSetDiagV2", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns a batched matrix tensor with new batched diagonal values.
    /// </summary>
    /// <remarks>
    /// 
    /// Given `input` and `diagonal`, this operation returns a tensor with the
    /// same shape and values as `input`, except for the specified diagonals of the
    /// innermost matrices. These will be overwritten by the values in `diagonal`.
    /// 
    /// `input` has `r+1` dimensions `[I, J, ..., L, M, N]`. When `k` is scalar or
    /// `k[0] == k[1]`, `diagonal` has `r` dimensions `[I, J, ..., L, max_diag_len]`.
    /// Otherwise, it has `r+1` dimensions `[I, J, ..., L, num_diags, max_diag_len]`.
    /// `num_diags` is the number of diagonals, `num_diags = k[1] - k[0] + 1`.
    /// `max_diag_len` is the longest diagonal in the range `[k[0], k[1]]`,
    /// `max_diag_len = min(M + min(k[1], 0), N + min(-k[0], 0))`
    /// 
    /// The output is a tensor of rank `k+1` with dimensions `[I, J, ..., L, M, N]`.
    /// If `k` is scalar or `k[0] == k[1]`:
    /// 
    /// ```
    /// output[i, j, ..., l, m, n]
    ///   = diagonal[i, j, ..., l, n-max(k[1], 0)] ; if n - m == k[1]
    ///     input[i, j, ..., l, m, n]              ; otherwise
    /// ```
    /// 
    /// Otherwise,
    /// 
    /// ```
    /// output[i, j, ..., l, m, n]
    ///   = diagonal[i, j, ..., l, diag_index, index_in_diag] ; if k[0] <= d <= k[1]
    ///     input[i, j, ..., l, m, n]                         ; otherwise
    /// ```
    /// where `d = n - m`, `diag_index = k[1] - d`, and
    /// `index_in_diag = n - max(d, 0) + offset`.
    /// 
    /// `offset` is zero except when the alignment of the diagonal is to the right.
    /// ```
    /// offset = max_diag_len - diag_len(d) ; if (`align` in {RIGHT_LEFT, RIGHT_RIGHT}
    ///                                            and `d >= 0`) or
    ///                                          (`align` in {LEFT_RIGHT, RIGHT_RIGHT}
    ///                                            and `d <= 0`)
    ///          0                          ; otherwise
    /// ```
    /// where `diag_len(d) = min(cols - max(d, 0), rows + min(d, 0))`.
    /// 
    /// For example:
    /// 
    /// ```
    /// # The main diagonal.
    /// input = np.array([[[7, 7, 7, 7],              # Input shape: (2, 3, 4)
    ///                    [7, 7, 7, 7],
    ///                    [7, 7, 7, 7]],
    ///                   [[7, 7, 7, 7],
    ///                    [7, 7, 7, 7],
    ///                    [7, 7, 7, 7]]])
    /// diagonal = np.array([[1, 2, 3],               # Diagonal shape: (2, 3)
    ///                      [4, 5, 6]])
    /// tf.matrix_set_diag(input, diagonal)
    ///   ==> [[[1, 7, 7, 7],  # Output shape: (2, 3, 4)
    ///         [7, 2, 7, 7],
    ///         [7, 7, 3, 7]],
    ///        [[4, 7, 7, 7],
    ///         [7, 5, 7, 7],
    ///         [7, 7, 6, 7]]]
    /// 
    /// # A superdiagonal (per batch).
    /// tf.matrix_set_diag(input, diagonal, k = 1)
    ///   ==> [[[7, 1, 7, 7],  # Output shape: (2, 3, 4)
    ///         [7, 7, 2, 7],
    ///         [7, 7, 7, 3]],
    ///        [[7, 4, 7, 7],
    ///         [7, 7, 5, 7],
    ///         [7, 7, 7, 6]]]
    /// 
    /// # A band of diagonals.
    /// diagonals = np.array([[[0, 9, 1],  # Diagonal shape: (2, 4, 3)
    ///                        [6, 5, 8],
    ///                        [1, 2, 3],
    ///                        [4, 5, 0]],
    ///                       [[0, 1, 2],
    ///                        [5, 6, 4],
    ///                        [6, 1, 2],
    ///                        [3, 4, 0]]])
    /// tf.matrix_set_diag(input, diagonals, k = (-1, 2))
    ///   ==> [[[1, 6, 9, 7],  # Output shape: (2, 3, 4)
    ///         [4, 2, 5, 1],
    ///         [7, 5, 3, 8]],
    ///        [[6, 5, 1, 7],
    ///         [3, 1, 6, 2],
    ///         [7, 4, 2, 4]]]
    /// 
    /// # LEFT_RIGHT alignment.
    /// diagonals = np.array([[[9, 1, 0],  # Diagonal shape: (2, 4, 3)
    ///                        [6, 5, 8],
    ///                        [1, 2, 3],
    ///                        [0, 4, 5]],
    ///                       [[1, 2, 0],
    ///                        [5, 6, 4],
    ///                        [6, 1, 2],
    ///                        [0, 3, 4]]])
    /// tf.matrix_set_diag(input, diagonals, k = (-1, 2), align="LEFT_RIGHT")
    ///   ==> [[[1, 6, 9, 7],  # Output shape: (2, 3, 4)
    ///         [4, 2, 5, 1],
    ///         [7, 5, 3, 8]],
    ///        [[6, 5, 1, 7],
    ///         [3, 1, 6, 2],
    ///         [7, 4, 2, 4]]]
    /// 
    /// ```
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="diagonal"></param>
    /// <param name="k"></param>
    /// <param name="align">
    /// 
    /// Some diagonals are shorter than `max_diag_len` and need to be padded. `align` is
    /// a string specifying how superdiagonals and subdiagonals should be aligned,
    /// respectively. There are four possible alignments: "RIGHT_LEFT" (default),
    /// "LEFT_RIGHT", "LEFT_LEFT", and "RIGHT_RIGHT". "RIGHT_LEFT" aligns superdiagonals
    /// to the right (left-pads the row) and subdiagonals to the left (right-pads the
    /// row). It is the packing format LAPACK uses. cuSPARSE uses "LEFT_RIGHT", which is
    /// the opposite alignment.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor matrix_set_diag_v3(Tensor input, Tensor diagonal, Tensor k, string align = "RIGHT_LEFT", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "MatrixSetDiagV3", name) { args = new object[] { input, diagonal, k }, attrs = new Dictionary<string, object>() { ["align"] = align } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return matrix_set_diag_v3_eager_fallback(input, diagonal, k, align: align, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        if (align is null)
        {
            align = "RIGHT_LEFT";
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["diagonal"] = diagonal;
        keywords["k"] = k;
        keywords["align"] = align;
        var _op = tf.OpDefLib._apply_op_helper("MatrixSetDiagV3", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "align", _op.get_attr("align") };
            _execute.record_gradient("MatrixSetDiagV3", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor matrix_set_diag_v3_eager_fallback(Tensor input, Tensor diagonal, Tensor k, string align, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, diagonal, k };
        object[] _attrs = new object[] { "T", input.dtype, "align", align };
        var _result = _execute.execute("MatrixSetDiagV3", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("MatrixSetDiagV3", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Pads a tensor with mirrored values.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation pads a `input` with mirrored values according to the `paddings`
    /// you specify. `paddings` is an integer tensor with shape `[n, 2]`, where n is
    /// the rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
    /// how many values to add before the contents of `input` in that dimension, and
    /// `paddings[D, 1]` indicates how many values to add after the contents of `input`
    /// in that dimension. Both `paddings[D, 0]` and `paddings[D, 1]` must be no greater
    /// than `input.dim_size(D)` (or `input.dim_size(D) - 1`) if `copy_border` is true
    /// (if false, respectively).
    /// 
    /// The padded size of each dimension D of the output is:
    /// 
    /// `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`
    /// 
    /// For example:
    /// 
    /// ```
    /// # 't' is [[1, 2, 3], [4, 5, 6]].
    /// # 'paddings' is [[1, 1]], [2, 2]].
    /// # 'mode' is SYMMETRIC.
    /// # rank of 't' is 2.
    /// pad(t, paddings) ==> [[2, 1, 1, 2, 3, 3, 2]
    ///                       [2, 1, 1, 2, 3, 3, 2]
    ///                       [5, 4, 4, 5, 6, 6, 5]
    ///                       [5, 4, 4, 5, 6, 6, 5]]
    /// ```
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="paddings"></param>
    /// <param name="mode">
    /// 
    /// Either `REFLECT` or `SYMMETRIC`. In reflect mode the padded regions
    /// do not include the borders, while in symmetric mode the padded regions
    /// do include the borders. For example, if `input` is `[1, 2, 3]` and `paddings`
    /// is `[0, 2]`, then the output is `[1, 2, 3, 2, 1]` in reflect mode, and
    /// it is `[1, 2, 3, 3, 2]` in symmetric mode.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor mirror_pad(Tensor input, Tensor paddings, string mode, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "MirrorPad", name) { args = new object[] { input, paddings }, attrs = new Dictionary<string, object>() { ["mode"] = mode } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return mirror_pad_eager_fallback(input, paddings, mode: mode, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["paddings"] = paddings;
        keywords["mode"] = mode;
        var _op = tf.OpDefLib._apply_op_helper("MirrorPad", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tpaddings", _op._get_attr_type("Tpaddings"), "mode", _op.get_attr("mode") };
            _execute.record_gradient("MirrorPad", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor mirror_pad_eager_fallback(Tensor input, Tensor paddings, string mode, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, paddings };
        object[] _attrs = new object[] { "T", input.dtype, "Tpaddings", paddings.dtype, "mode", mode };
        var _result = _execute.execute("MirrorPad", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("MirrorPad", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Gradient op for `MirrorPad` op. This op folds a mirror-padded tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation folds the padded areas of `input` by `MirrorPad` according to the
    /// `paddings` you specify. `paddings` must be the same as `paddings` argument
    /// given to the corresponding `MirrorPad` op.
    /// 
    /// The folded size of each dimension D of the output is:
    /// 
    /// `input.dim_size(D) - paddings(D, 0) - paddings(D, 1)`
    /// 
    /// For example:
    /// 
    /// ```
    /// # 't' is [[1, 2, 3], [4, 5, 6], [7, 8, 9]].
    /// # 'paddings' is [[0, 1]], [0, 1]].
    /// # 'mode' is SYMMETRIC.
    /// # rank of 't' is 2.
    /// pad(t, paddings) ==> [[ 1,  5]
    ///                       [11, 28]]
    /// ```
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="paddings"></param>
    /// <param name="mode">
    /// 
    /// The mode used in the `MirrorPad` op.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor mirror_pad_grad(Tensor input, Tensor paddings, string mode, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "MirrorPadGrad", name) { args = new object[] { input, paddings }, attrs = new Dictionary<string, object>() { ["mode"] = mode } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return mirror_pad_grad_eager_fallback(input, paddings, mode: mode, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["paddings"] = paddings;
        keywords["mode"] = mode;
        var _op = tf.OpDefLib._apply_op_helper("MirrorPadGrad", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tpaddings", _op._get_attr_type("Tpaddings"), "mode", _op.get_attr("mode") };
            _execute.record_gradient("MirrorPadGrad", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor mirror_pad_grad_eager_fallback(Tensor input, Tensor paddings, string mode, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, paddings };
        object[] _attrs = new object[] { "T", input.dtype, "Tpaddings", paddings.dtype, "mode", mode };
        var _result = _execute.execute("MirrorPadGrad", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("MirrorPadGrad", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns a one-hot tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// The locations represented by indices in `indices` take value `on_value`,
    /// while all other locations take value `off_value`.
    /// 
    /// If the input `indices` is rank `N`, the output will have rank `N+1`,
    /// The new axis is created at dimension `axis` (default: the new axis is
    /// appended at the end).
    /// 
    /// If `indices` is a scalar the output shape will be a vector of length `depth`.
    /// 
    /// If `indices` is a vector of length `features`, the output shape will be:
    /// ```
    ///   features x depth if axis == -1
    ///   depth x features if axis == 0
    /// ```
    /// 
    /// If `indices` is a matrix (batch) with shape `[batch, features]`,
    /// the output shape will be:
    /// ```
    ///   batch x features x depth if axis == -1
    ///   batch x depth x features if axis == 1
    ///   depth x batch x features if axis == 0
    /// ```
    /// 
    /// 
    /// Examples
    /// =========
    /// 
    /// Suppose that
    /// ```
    ///   indices = [0, 2, -1, 1]
    ///   depth = 3
    ///   on_value = 5.0
    ///   off_value = 0.0
    ///   axis = -1
    /// ```
    /// 
    /// Then output is `[4 x 3]`:
    /// ```
    /// output =
    ///   [5.0 0.0 0.0]  // one_hot(0)
    ///   [0.0 0.0 5.0]  // one_hot(2)
    ///   [0.0 0.0 0.0]  // one_hot(-1)
    ///   [0.0 5.0 0.0]  // one_hot(1)
    /// ```
    /// 
    /// Suppose that
    /// ```
    ///   indices = [0, 2, -1, 1]
    ///   depth = 3
    ///   on_value = 0.0
    ///   off_value = 3.0
    ///   axis = 0
    /// ```
    /// 
    /// Then output is `[3 x 4]`:
    /// ```
    /// output =
    ///   [0.0 3.0 3.0 3.0]
    ///   [3.0 3.0 3.0 0.0]
    ///   [3.0 3.0 3.0 3.0]
    ///   [3.0 0.0 3.0 3.0]
    /// //  ^                one_hot(0)
    /// //      ^            one_hot(2)
    /// //          ^        one_hot(-1)
    /// //              ^    one_hot(1)
    /// ```
    /// 
    /// Suppose that
    /// ```
    ///   indices = [[0, 2], [1, -1]]
    ///   depth = 3
    ///   on_value = 1.0
    ///   off_value = 0.0
    ///   axis = -1
    /// ```
    /// 
    /// Then output is `[2 x 2 x 3]`:
    /// ```
    /// output =
    ///   [
    ///     [1.0, 0.0, 0.0]  // one_hot(0)
    ///     [0.0, 0.0, 1.0]  // one_hot(2)
    ///   ][
    ///     [0.0, 1.0, 0.0]  // one_hot(1)
    ///     [0.0, 0.0, 0.0]  // one_hot(-1)
    ///   ]
    /// ```
    /// 
    /// </remarks>
    /// <param name="indices"></param>
    /// <param name="depth"></param>
    /// <param name="on_value"></param>
    /// <param name="off_value"></param>
    /// <param name="axis">
    /// 
    /// The axis to fill (default: -1, a new inner-most axis).
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor one_hot(Tensor indices, Tensor depth, Tensor on_value, Tensor off_value, int axis = -1, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "OneHot", name) { args = new object[] { indices, depth, on_value, off_value }, attrs = new Dictionary<string, object>() { ["axis"] = axis } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return one_hot_eager_fallback(indices, depth, on_value, off_value, axis: axis, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["indices"] = indices;
        keywords["depth"] = depth;
        keywords["on_value"] = on_value;
        keywords["off_value"] = off_value;
        keywords["axis"] = axis;
        var _op = tf.OpDefLib._apply_op_helper("OneHot", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "axis", _op._get_attr_int("axis"), "T", _op._get_attr_type("T"), "TI", _op._get_attr_type("TI") };
            _execute.record_gradient("OneHot", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor one_hot_eager_fallback(Tensor indices, Tensor depth, Tensor on_value, Tensor off_value, int axis, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { indices, depth, on_value, off_value };
        object[] _attrs = new object[] { "axis", axis, "T", on_value.dtype, "TI", indices.dtype };
        var _result = _execute.execute("OneHot", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("OneHot", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns a tensor of ones with the same shape and type as x.
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor ones_like(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "OnesLike", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return ones_like_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("OnesLike", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("OnesLike", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor ones_like_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("OnesLike", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("OnesLike", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Packs a list of `N` rank-`R` tensors into one rank-`(R+1)` tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Packs the `N` tensors in `values` into a tensor with rank one higher than each
    /// tensor in `values`, by packing them along the `axis` dimension.
    /// Given a list of tensors of shape `(A, B, C)`;
    /// 
    /// if `axis == 0` then the `output` tensor will have the shape `(N, A, B, C)`.
    /// if `axis == 1` then the `output` tensor will have the shape `(A, N, B, C)`.
    /// Etc.
    /// 
    /// For example:
    /// 
    /// ```
    /// # 'x' is [1, 4]
    /// # 'y' is [2, 5]
    /// # 'z' is [3, 6]
    /// pack([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
    /// pack([x, y, z], axis=1) => [[1, 2, 3], [4, 5, 6]]
    /// ```
    /// 
    /// This is the opposite of `unpack`.
    /// 
    /// </remarks>
    /// <param name="values"></param>
    /// <param name="axis">
    /// 
    /// Dimension along which to pack.  Negative values wrap around, so the
    /// valid range is `[-(R+1), R+1)`.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor pack(Tensors values, int axis = 0, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Pack", name) { args = new object[] { values }, attrs = new Dictionary<string, object>() { ["axis"] = axis } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return pack_eager_fallback(values, axis: axis, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["values"] = values;
        keywords["axis"] = axis;
        var _op = tf.OpDefLib._apply_op_helper("Pack", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "N", _op._get_attr_int("N"), "T", _op._get_attr_type("T"), "axis", _op._get_attr_int("axis") };
            _execute.record_gradient("Pack", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor pack_eager_fallback(Tensors values, int axis, string name, Context ctx)
    {
        List<Tensor> _inputs_flat_list = new();
        _inputs_flat_list.AddRange(values);
        var _inputs_flat = _inputs_flat_list.ToArray();
        object[] _attrs = new object[] { "N", values.Length, "T", values.dtype, "axis", axis };
        var _result = _execute.execute("Pack", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Pack", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Pads a tensor with zeros.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation pads a `input` with zeros according to the `paddings` you
    /// specify. `paddings` is an integer tensor with shape `[Dn, 2]`, where n is the
    /// rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
    /// how many zeros to add before the contents of `input` in that dimension, and
    /// `paddings[D, 1]` indicates how many zeros to add after the contents of `input`
    /// in that dimension.
    /// 
    /// The padded size of each dimension D of the output is:
    /// 
    /// `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`
    /// 
    /// For example:
    /// 
    /// ```
    /// # 't' is [[1, 1], [2, 2]]
    /// # 'paddings' is [[1, 1], [2, 2]]
    /// # rank of 't' is 2
    /// pad(t, paddings) ==> [[0, 0, 0, 0, 0, 0]
    ///                       [0, 0, 1, 1, 0, 0]
    ///                       [0, 0, 2, 2, 0, 0]
    ///                       [0, 0, 0, 0, 0, 0]]
    /// ```
    /// 
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="paddings"></param>
    /// <returns></returns>
    public static Tensor pad(Tensor input, Tensor paddings, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Pad", name) { args = new object[] { input, paddings }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return pad_eager_fallback(input, paddings, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["paddings"] = paddings;
        var _op = tf.OpDefLib._apply_op_helper("Pad", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tpaddings", _op._get_attr_type("Tpaddings") };
            _execute.record_gradient("Pad", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor pad_eager_fallback(Tensor input, Tensor paddings, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, paddings };
        object[] _attrs = new object[] { "T", input.dtype, "Tpaddings", paddings.dtype };
        var _result = _execute.execute("Pad", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Pad", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Pads a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation pads `input` according to the `paddings` and `constant_values`
    /// you specify. `paddings` is an integer tensor with shape `[Dn, 2]`, where n is
    /// the rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
    /// how many padding values to add before the contents of `input` in that dimension,
    /// and `paddings[D, 1]` indicates how many padding values to add after the contents
    /// of `input` in that dimension. `constant_values` is a scalar tensor of the same
    /// type as `input` that indicates the value to use for padding `input`.
    /// 
    /// The padded size of each dimension D of the output is:
    /// 
    /// `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`
    /// 
    /// For example:
    /// 
    /// ```
    /// # 't' is [[1, 1], [2, 2]]
    /// # 'paddings' is [[1, 1], [2, 2]]
    /// # 'constant_values' is 0
    /// # rank of 't' is 2
    /// pad(t, paddings) ==> [[0, 0, 0, 0, 0, 0]
    ///                       [0, 0, 1, 1, 0, 0]
    ///                       [0, 0, 2, 2, 0, 0]
    ///                       [0, 0, 0, 0, 0, 0]]
    /// ```
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="paddings"></param>
    /// <param name="constant_values"></param>
    /// <returns></returns>
    public static Tensor pad_v2(Tensor input, Tensor paddings, Tensor constant_values, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "PadV2", name) { args = new object[] { input, paddings, constant_values }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return pad_v2_eager_fallback(input, paddings, constant_values, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["paddings"] = paddings;
        keywords["constant_values"] = constant_values;
        var _op = tf.OpDefLib._apply_op_helper("PadV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tpaddings", _op._get_attr_type("Tpaddings") };
            _execute.record_gradient("PadV2", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor pad_v2_eager_fallback(Tensor input, Tensor paddings, Tensor constant_values, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, paddings, constant_values };
        object[] _attrs = new object[] { "T", input.dtype, "Tpaddings", paddings.dtype };
        var _result = _execute.execute("PadV2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("PadV2", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Concatenates a list of `N` tensors along the first dimension.
    /// </summary>
    /// <remarks>
    /// 
    /// The input tensors are all required to have size 1 in the first dimension.
    /// 
    /// For example:
    /// 
    /// ```
    /// # 'x' is [[1, 4]]
    /// # 'y' is [[2, 5]]
    /// # 'z' is [[3, 6]]
    /// parallel_concat([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
    /// ```
    /// 
    /// The difference between concat and parallel_concat is that concat requires all
    /// of the inputs be computed before the operation will begin but doesn't require
    /// that the input shapes be known during graph construction.  Parallel concat
    /// will copy pieces of the input into the output as they become available, in
    /// some situations this can provide a performance benefit.
    /// 
    /// </remarks>
    /// <param name="values"></param>
    /// <param name="shape">
    /// 
    /// the final shape of the result; should be equal to the shapes of any input
    /// but with the number of input values in the first dimension.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor parallel_concat(Tensors values, Shape shape, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ParallelConcat", name) { args = new object[] { values }, attrs = new Dictionary<string, object>() { ["shape"] = shape } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return parallel_concat_eager_fallback(values, shape: shape, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["values"] = values;
        keywords["shape"] = shape;
        var _op = tf.OpDefLib._apply_op_helper("ParallelConcat", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "N", _op._get_attr_int("N"), "T", _op._get_attr_type("T"), "shape", _op.get_attr("shape") };
            _execute.record_gradient("ParallelConcat", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor parallel_concat_eager_fallback(Tensors values, Shape shape, string name, Context ctx)
    {
        List<Tensor> _inputs_flat_list = new();
        _inputs_flat_list.AddRange(values);
        var _inputs_flat = _inputs_flat_list.ToArray();
        object[] _attrs = new object[] { "N", values.Length, "T", values.dtype, "shape", shape };
        var _result = _execute.execute("ParallelConcat", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ParallelConcat", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// A placeholder op for a value that will be fed into the computation.
    /// </summary>
    /// <remarks>
    /// 
    /// N.B. This operation will fail with an error if it is executed. It is
    /// intended as a way to represent a value that will always be fed, and to
    /// provide attrs that enable the fed value to be checked at runtime.
    /// 
    /// </remarks>
    /// <param name="dtype">
    /// 
    /// The type of elements in the tensor.
    /// 
    /// </param>
    /// <param name="shape">
    /// 
    /// (Optional) The shape of the tensor. If the shape has 0 dimensions, the
    /// shape is unconstrained.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor placeholder(TF_DataType dtype, Shape shape = null, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Placeholder", name) { args = new object[] { }, attrs = new Dictionary<string, object>() { ["dtype"] = dtype, ["shape"] = shape } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return placeholder_eager_fallback(dtype: dtype, shape: shape, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["dtype"] = dtype;
        keywords["shape"] = shape;
        var _op = tf.OpDefLib._apply_op_helper("Placeholder", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "dtype", _op._get_attr_type("dtype"), "shape", _op.get_attr("shape") };
            _execute.record_gradient("Placeholder", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor placeholder_eager_fallback(TF_DataType dtype, Shape shape, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { };
        object[] _attrs = new object[] { "dtype", dtype, "shape", shape };
        var _result = _execute.execute("Placeholder", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Placeholder", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// A placeholder op for a value that will be fed into the computation.
    /// </summary>
    /// <remarks>
    /// 
    /// N.B. This operation will fail with an error if it is executed. It is
    /// intended as a way to represent a value that will always be fed, and to
    /// provide attrs that enable the fed value to be checked at runtime.
    /// 
    /// </remarks>
    /// <param name="dtype">
    /// 
    /// The type of elements in the tensor.
    /// 
    /// </param>
    /// <param name="shape">
    /// 
    /// The shape of the tensor. The shape can be any partially-specified
    /// shape.  To be unconstrained, pass in a shape with unknown rank.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor placeholder_v2(TF_DataType dtype, Shape shape, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "PlaceholderV2", name) { args = new object[] { }, attrs = new Dictionary<string, object>() { ["dtype"] = dtype, ["shape"] = shape } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return placeholder_v2_eager_fallback(dtype: dtype, shape: shape, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["dtype"] = dtype;
        keywords["shape"] = shape;
        var _op = tf.OpDefLib._apply_op_helper("PlaceholderV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "dtype", _op._get_attr_type("dtype"), "shape", _op.get_attr("shape") };
            _execute.record_gradient("PlaceholderV2", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor placeholder_v2_eager_fallback(TF_DataType dtype, Shape shape, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { };
        object[] _attrs = new object[] { "dtype", dtype, "shape", shape };
        var _result = _execute.execute("PlaceholderV2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("PlaceholderV2", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// A placeholder op that passes through `input` when its output is not fed.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="shape">
    /// 
    /// The (possibly partial) shape of the tensor.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor placeholder_with_default(Tensor input, Shape shape, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "PlaceholderWithDefault", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { ["shape"] = shape } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return placeholder_with_default_eager_fallback(input, shape: shape, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["shape"] = shape;
        var _op = tf.OpDefLib._apply_op_helper("PlaceholderWithDefault", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "dtype", _op._get_attr_type("dtype"), "shape", _op.get_attr("shape") };
            _execute.record_gradient("PlaceholderWithDefault", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor placeholder_with_default_eager_fallback(Tensor input, Shape shape, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "dtype", input.dtype, "shape", shape };
        var _result = _execute.execute("PlaceholderWithDefault", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("PlaceholderWithDefault", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// An identity op that triggers an error if a gradient is requested.
    /// </summary>
    /// <remarks>
    /// 
    /// When executed in a graph, this op outputs its input tensor as-is.
    /// 
    /// When building ops to compute gradients, the TensorFlow gradient system
    /// will return an error when trying to lookup the gradient of this op,
    /// because no gradient must ever be registered for this function.  This
    /// op exists to prevent subtle bugs from silently returning unimplemented
    /// gradients in some corner cases.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="message">
    /// 
    /// Will be printed in the error when anyone tries to differentiate
    /// this operation.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor prevent_gradient(Tensor input, string message = "", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "PreventGradient", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { ["message"] = message } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return prevent_gradient_eager_fallback(input, message: message, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        if (message is null)
        {
            message = "";
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["message"] = message;
        var _op = tf.OpDefLib._apply_op_helper("PreventGradient", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "message", _op.get_attr("message") };
            _execute.record_gradient("PreventGradient", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor prevent_gradient_eager_fallback(Tensor input, string message, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "T", input.dtype, "message", message };
        var _result = _execute.execute("PreventGradient", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("PreventGradient", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Use QuantizeAndDequantizeV2 instead.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="signed_input"></param>
    /// <param name="num_bits"></param>
    /// <param name="range_given"></param>
    /// <param name="input_min"></param>
    /// <param name="input_max"></param>
    /// <returns></returns>
    public static Tensor quantize_and_dequantize(Tensor input, bool signed_input = true, int num_bits = 8, bool range_given = false, float input_min = 0f, float input_max = 0f, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizeAndDequantize", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { ["signed_input"] = signed_input, ["num_bits"] = num_bits, ["range_given"] = range_given, ["input_min"] = input_min, ["input_max"] = input_max } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return quantize_and_dequantize_eager_fallback(input, signed_input: signed_input, num_bits: num_bits, range_given: range_given, input_min: input_min, input_max: input_max, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["signed_input"] = signed_input;
        keywords["num_bits"] = num_bits;
        keywords["range_given"] = range_given;
        keywords["input_min"] = input_min;
        keywords["input_max"] = input_max;
        var _op = tf.OpDefLib._apply_op_helper("QuantizeAndDequantize", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "signed_input", _op._get_attr_bool("signed_input"), "num_bits", _op._get_attr_int("num_bits"), "range_given", _op._get_attr_bool("range_given"), "input_min", _op.get_attr("input_min"), "input_max", _op.get_attr("input_max"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("QuantizeAndDequantize", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor quantize_and_dequantize_eager_fallback(Tensor input, bool signed_input, int num_bits, bool range_given, float input_min, float input_max, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "signed_input", signed_input, "num_bits", num_bits, "range_given", range_given, "input_min", input_min, "input_max", input_max, "T", input.dtype };
        var _result = _execute.execute("QuantizeAndDequantize", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizeAndDequantize", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Quantizes then dequantizes a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// This op simulates the precision loss from the quantized forward pass by:
    /// 
    /// 1. Quantizing the tensor to fixed point numbers, which should match the target
    ///    quantization method when it is used in inference.
    /// 2. Dequantizing it back to floating point numbers for the following ops, most
    ///    likely matmul.
    /// 
    /// There are different ways to quantize. This version uses only scaling, so 0.0
    /// maps to 0.
    /// 
    /// From the specified 'num_bits' in the quantized output type, it determines
    /// minimum and maximum representable quantized values.
    /// 
    /// e.g.
    /// 
    /// *   [-128, 127] for signed, num_bits = 8, or
    /// *   [0, 255] for unsigned, num_bits = 8.
    /// 
    /// If range_given == False, the initial input_min, input_max will be determined
    /// automatically as the minimum and maximum values in the input tensor, otherwise
    /// the specified values of input_min, input_max are used.
    /// 
    /// Note: If the input_min, input_max are specified, they do not need to equal the
    /// actual minimum and maximum values in the tensor. e.g. in some cases it may be
    /// beneficial to specify these values such that the low probability extremes of the
    /// input distribution are clipped.
    /// 
    /// This op determines the maximum scale_factor that would map the initial
    /// [input_min, input_max] range to a range that lies within the representable
    /// quantized range.
    /// 
    /// It determines the scale from one of input_min and input_max, then updates the
    /// other one to maximize the representable range.
    /// 
    /// e.g.
    /// 
    /// *   if the output is signed, num_bits = 8, [input_min, input_max] = [-10.0,
    ///     5.0]: it would use a scale_factor of -128 / -10.0 = 12.8 In this case, it
    ///     would update input_max to be 127 / 12.8 = 9.921875
    /// *   if the output is signed, num_bits = 8, [input_min, input_max] = [-10.0,
    ///     10.0]: it would use a scale_factor of 127 / 10.0 = 12.7 In this case, it
    ///     would update input_min to be 128.0 / 12.7 = -10.07874
    /// *   if the output is unsigned, input_min is forced to be 0, and only the
    ///     specified input_max is used.
    /// 
    /// After determining the scale_factor and updating the input range, it applies the
    /// following to each value in the 'input' tensor.
    /// 
    /// output = round(clamp(value, input_min, input_max) * scale_factor) / scale_factor.
    /// 
    /// The above round function rounds the value based on the given round_mode.
    /// 
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="input_min"></param>
    /// <param name="input_max"></param>
    /// <param name="signed_input">
    /// 
    /// Whether the quantization is signed or unsigned. (actually this parameter should
    /// have been called <b>`signed_output`</b>)
    /// 
    /// </param>
    /// <param name="num_bits">
    /// 
    /// The bitwidth of the quantization.
    /// 
    /// </param>
    /// <param name="range_given">
    /// 
    /// Whether the range is given or should be determined from the `input` tensor.
    /// 
    /// </param>
    /// <param name="round_mode">
    /// 
    /// The 'round_mode' attribute controls which rounding tie-breaking algorithm is
    /// used when rounding float values to their quantized equivalents. The following
    /// rounding modes are currently supported:
    /// 
    /// *   HALF_TO_EVEN: this is the default round_mode.
    /// *   HALF_UP: round towards positive. In this mode 7.5 rounds up to 8 and -7.5
    ///     rounds up to -7.
    /// 
    /// 
    /// </param>
    /// <param name="narrow_range">
    /// 
    /// If True, then the absolute value of the quantized minimum value is the same as
    /// the quantized maximum value, instead of 1 greater.
    /// i.e. for 8 bit quantization, the minimum value is -127 instead of -128.
    /// 
    /// </param>
    /// <param name="axis">
    /// 
    /// If specified, this axis is treated as a channel or slice axis, and a separate
    /// quantization range is used for each channel or slice along this axis.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor quantize_and_dequantize_v2(Tensor input, Tensor input_min, Tensor input_max, bool signed_input = true, int num_bits = 8, bool range_given = false, string round_mode = "HALF_TO_EVEN", bool narrow_range = false, int axis = -1, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizeAndDequantizeV2", name) { args = new object[] { input, input_min, input_max }, attrs = new Dictionary<string, object>() { ["signed_input"] = signed_input, ["num_bits"] = num_bits, ["range_given"] = range_given, ["round_mode"] = round_mode, ["narrow_range"] = narrow_range, ["axis"] = axis } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return quantize_and_dequantize_v2_eager_fallback(input, input_min, input_max, signed_input: signed_input, num_bits: num_bits, range_given: range_given, round_mode: round_mode, narrow_range: narrow_range, axis: axis, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        if (round_mode is null)
        {
            round_mode = "HALF_TO_EVEN";
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["input_min"] = input_min;
        keywords["input_max"] = input_max;
        keywords["signed_input"] = signed_input;
        keywords["num_bits"] = num_bits;
        keywords["range_given"] = range_given;
        keywords["round_mode"] = round_mode;
        keywords["narrow_range"] = narrow_range;
        keywords["axis"] = axis;
        var _op = tf.OpDefLib._apply_op_helper("QuantizeAndDequantizeV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "signed_input", _op._get_attr_bool("signed_input"), "num_bits", _op._get_attr_int("num_bits"), "range_given", _op._get_attr_bool("range_given"), "T", _op._get_attr_type("T"), "round_mode", _op.get_attr("round_mode"), "narrow_range", _op._get_attr_bool("narrow_range"), "axis", _op._get_attr_int("axis") };
            _execute.record_gradient("QuantizeAndDequantizeV2", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor quantize_and_dequantize_v2_eager_fallback(Tensor input, Tensor input_min, Tensor input_max, bool signed_input, int num_bits, bool range_given, string round_mode, bool narrow_range, int axis, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, input_min, input_max };
        object[] _attrs = new object[] { "signed_input", signed_input, "num_bits", num_bits, "range_given", range_given, "T", input.dtype, "round_mode", round_mode, "narrow_range", narrow_range, "axis", axis };
        var _result = _execute.execute("QuantizeAndDequantizeV2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizeAndDequantizeV2", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Quantizes then dequantizes a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// This is almost identical to QuantizeAndDequantizeV2, except that num_bits is a
    /// tensor, so its value can change during training.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="input_min"></param>
    /// <param name="input_max"></param>
    /// <param name="num_bits"></param>
    /// <param name="signed_input"></param>
    /// <param name="range_given"></param>
    /// <param name="narrow_range"></param>
    /// <param name="axis"></param>
    /// <returns></returns>
    public static Tensor quantize_and_dequantize_v3(Tensor input, Tensor input_min, Tensor input_max, Tensor num_bits, bool signed_input = true, bool range_given = true, bool narrow_range = false, int axis = -1, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizeAndDequantizeV3", name) { args = new object[] { input, input_min, input_max, num_bits }, attrs = new Dictionary<string, object>() { ["signed_input"] = signed_input, ["range_given"] = range_given, ["narrow_range"] = narrow_range, ["axis"] = axis } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return quantize_and_dequantize_v3_eager_fallback(input, input_min, input_max, num_bits, signed_input: signed_input, range_given: range_given, narrow_range: narrow_range, axis: axis, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["input_min"] = input_min;
        keywords["input_max"] = input_max;
        keywords["num_bits"] = num_bits;
        keywords["signed_input"] = signed_input;
        keywords["range_given"] = range_given;
        keywords["narrow_range"] = narrow_range;
        keywords["axis"] = axis;
        var _op = tf.OpDefLib._apply_op_helper("QuantizeAndDequantizeV3", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "signed_input", _op._get_attr_bool("signed_input"), "range_given", _op._get_attr_bool("range_given"), "T", _op._get_attr_type("T"), "narrow_range", _op._get_attr_bool("narrow_range"), "axis", _op._get_attr_int("axis") };
            _execute.record_gradient("QuantizeAndDequantizeV3", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor quantize_and_dequantize_v3_eager_fallback(Tensor input, Tensor input_min, Tensor input_max, Tensor num_bits, bool signed_input, bool range_given, bool narrow_range, int axis, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, input_min, input_max, num_bits };
        object[] _attrs = new object[] { "signed_input", signed_input, "range_given", range_given, "T", input.dtype, "narrow_range", narrow_range, "axis", axis };
        var _result = _execute.execute("QuantizeAndDequantizeV3", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizeAndDequantizeV3", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Quantizes then dequantizes a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// This is almost identical to QuantizeAndDequantizeV2, except that it returns a
    /// gradient of 1 for inputs that are within the quantization range, or 0 otherwise.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="input_min"></param>
    /// <param name="input_max"></param>
    /// <param name="signed_input">
    /// 
    /// Whether the quantization is signed or unsigned. (actually this parameter should
    /// have been called <b>`signed_output`</b>)
    /// 
    /// </param>
    /// <param name="num_bits">
    /// 
    /// The bitwidth of the quantization.
    /// 
    /// </param>
    /// <param name="range_given">
    /// 
    /// Whether the range is given or should be determined from the `input` tensor.
    /// 
    /// </param>
    /// <param name="round_mode">
    /// 
    /// The 'round_mode' attribute controls which rounding tie-breaking algorithm is
    /// used when rounding float values to their quantized equivalents. The following
    /// rounding modes are currently supported:
    /// 
    /// *   HALF_TO_EVEN: this is the default round_mode.
    /// *   HALF_UP: round towards positive. In this mode 7.5 rounds up to 8 and -7.5
    ///     rounds up to -7.
    /// 
    /// 
    /// </param>
    /// <param name="narrow_range">
    /// 
    /// If True, then the absolute value of the quantized minimum value is the same as
    /// the quantized maximum value, instead of 1 greater.
    /// i.e. for 8 bit quantization, the minimum value is -127 instead of -128.
    /// 
    /// </param>
    /// <param name="axis">
    /// 
    /// If specified, this axis is treated as a channel or slice axis, and a separate
    /// quantization range is used for each channel or slice along this axis.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor quantize_and_dequantize_v4(Tensor input, Tensor input_min, Tensor input_max, bool signed_input = true, int num_bits = 8, bool range_given = false, string round_mode = "HALF_TO_EVEN", bool narrow_range = false, int axis = -1, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizeAndDequantizeV4", name) { args = new object[] { input, input_min, input_max }, attrs = new Dictionary<string, object>() { ["signed_input"] = signed_input, ["num_bits"] = num_bits, ["range_given"] = range_given, ["round_mode"] = round_mode, ["narrow_range"] = narrow_range, ["axis"] = axis } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return quantize_and_dequantize_v4_eager_fallback(input, input_min, input_max, signed_input: signed_input, num_bits: num_bits, range_given: range_given, round_mode: round_mode, narrow_range: narrow_range, axis: axis, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        if (round_mode is null)
        {
            round_mode = "HALF_TO_EVEN";
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["input_min"] = input_min;
        keywords["input_max"] = input_max;
        keywords["signed_input"] = signed_input;
        keywords["num_bits"] = num_bits;
        keywords["range_given"] = range_given;
        keywords["round_mode"] = round_mode;
        keywords["narrow_range"] = narrow_range;
        keywords["axis"] = axis;
        var _op = tf.OpDefLib._apply_op_helper("QuantizeAndDequantizeV4", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "signed_input", _op._get_attr_bool("signed_input"), "num_bits", _op._get_attr_int("num_bits"), "range_given", _op._get_attr_bool("range_given"), "T", _op._get_attr_type("T"), "round_mode", _op.get_attr("round_mode"), "narrow_range", _op._get_attr_bool("narrow_range"), "axis", _op._get_attr_int("axis") };
            _execute.record_gradient("QuantizeAndDequantizeV4", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor quantize_and_dequantize_v4_eager_fallback(Tensor input, Tensor input_min, Tensor input_max, bool signed_input, int num_bits, bool range_given, string round_mode, bool narrow_range, int axis, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, input_min, input_max };
        object[] _attrs = new object[] { "signed_input", signed_input, "num_bits", num_bits, "range_given", range_given, "T", input.dtype, "round_mode", round_mode, "narrow_range", narrow_range, "axis", axis };
        var _result = _execute.execute("QuantizeAndDequantizeV4", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizeAndDequantizeV4", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Quantize the 'input' tensor of type float to 'output' tensor of type 'T'.
    /// </summary>
    /// <remarks>
    /// 
    /// [min_range, max_range] are scalar floats that specify the range for
    /// the 'input' data. The 'mode' attribute controls exactly which calculations are
    /// used to convert the float values to their quantized equivalents.  The
    /// 'round_mode' attribute controls which rounding tie-breaking algorithm is used
    /// when rounding float values to their quantized equivalents.
    /// 
    /// In 'MIN_COMBINED' mode, each value of the tensor will undergo the following:
    /// 
    /// ```
    /// out[i] = (in[i] - min_range) * range(T) / (max_range - min_range)
    /// if T == qint8: out[i] -= (range(T) + 1) / 2.0
    /// ```
    /// 
    /// here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`
    /// 
    /// *MIN_COMBINED Mode Example*
    /// 
    /// Assume the input is type float and has a possible range of [0.0, 6.0] and the
    /// output type is quint8 ([0, 255]). The min_range and max_range values should be
    /// specified as 0.0 and 6.0. Quantizing from float to quint8 will multiply each
    /// value of the input by 255/6 and cast to quint8.
    /// 
    /// If the output type was qint8 ([-128, 127]), the operation will additionally
    /// subtract each value by 128 prior to casting, so that the range of values aligns
    /// with the range of qint8.
    /// 
    /// If the mode is 'MIN_FIRST', then this approach is used:
    /// 
    /// ```
    /// num_discrete_values = 1 << (# of bits in T)
    /// range_adjust = num_discrete_values / (num_discrete_values - 1)
    /// range = (range_max - range_min) * range_adjust
    /// range_scale = num_discrete_values / range
    /// quantized = round(input * range_scale) - round(range_min * range_scale) +
    ///   numeric_limits<T>::min()
    /// quantized = max(quantized, numeric_limits<T>::min())
    /// quantized = min(quantized, numeric_limits<T>::max())
    /// ```
    /// 
    /// The biggest difference between this and MIN_COMBINED is that the minimum range
    /// is rounded first, before it's subtracted from the rounded value. With
    /// MIN_COMBINED, a small bias is introduced where repeated iterations of quantizing
    /// and dequantizing will introduce a larger and larger error.
    /// 
    /// *SCALED mode Example*
    /// 
    /// `SCALED` mode matches the quantization approach used in
    /// `QuantizeAndDequantize{V2|V3}`.
    /// 
    /// If the mode is `SCALED`, the quantization is performed by multiplying each
    /// input value by a scaling_factor.
    /// The scaling_factor is determined from `min_range` and `max_range` to be as large
    /// as possible such that the range from `min_range` to `max_range` is representable
    /// within values of type T.
    /// 
    /// ```c++
    /// 
    ///   const int min_T = std::numeric_limits<T>::min();
    ///   const int max_T = std::numeric_limits<T>::max();
    ///   const float max_float = std::numeric_limits<float>::max();
    /// 
    ///   const float scale_factor_from_min_side =
    ///       (min_T * min_range > 0) ? min_T / min_range : max_float;
    ///   const float scale_factor_from_max_side =
    ///       (max_T * max_range > 0) ? max_T / max_range : max_float;
    /// 
    ///   const float scale_factor = std::min(scale_factor_from_min_side,
    ///                                       scale_factor_from_max_side);
    /// ```
    /// 
    /// We next use the scale_factor to adjust min_range and max_range as follows:
    /// 
    /// ```c++
    ///       min_range = min_T / scale_factor;
    ///       max_range = max_T / scale_factor;
    /// ```
    /// 
    /// 
    /// e.g. if T = qint8, and initially min_range = -10, and max_range = 9, we would
    /// compare -128/-10.0 = 12.8 to 127/9.0 = 14.11, and set scaling_factor = 12.8
    /// In this case, min_range would remain -10, but max_range would be adjusted to
    /// 127 / 12.8 = 9.921875
    /// 
    /// So we will quantize input values in the range (-10, 9.921875) to (-128, 127).
    /// 
    /// The input tensor can now be quantized by clipping values to the range
    /// `min_range` to `max_range`, then multiplying by scale_factor as follows:
    /// 
    /// ```c++
    /// result = round(min(max_range, max(min_range, input)) * scale_factor)
    /// ```
    /// 
    /// The adjusted `min_range` and `max_range` are returned as outputs 2 and 3 of
    /// this operation. These outputs should be used as the range for any further
    /// calculations.
    /// 
    /// 
    /// *narrow_range (bool) attribute*
    /// 
    /// If true, we do not use the minimum quantized value.
    /// i.e. for int8 the quantized output, it would be restricted to the range
    /// -127..127 instead of the full -128..127 range.
    /// This is provided for compatibility with certain inference backends.
    /// (Only applies to SCALED mode)
    /// 
    /// 
    /// *axis (int) attribute*
    /// 
    /// An optional `axis` attribute can specify a dimension index of the input tensor,
    /// such that quantization ranges will be calculated and applied separately for each
    /// slice of the tensor along that dimension. This is useful for per-channel
    /// quantization.
    /// 
    /// If axis is specified, min_range and max_range
    /// 
    /// if `axis`=None, per-tensor quantization is performed as normal.
    /// 
    /// 
    /// *ensure_minimum_range (float) attribute*
    /// 
    /// Ensures the minimum quantization range is at least this value.
    /// The legacy default value for this is 0.01, but it is strongly suggested to
    /// set it to 0 for new uses.
    /// 
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="min_range"></param>
    /// <param name="max_range"></param>
    /// <param name="T"></param>
    /// <param name="mode"></param>
    /// <param name="round_mode"></param>
    /// <param name="narrow_range"></param>
    /// <param name="axis"></param>
    /// <param name="ensure_minimum_range"></param>
    /// <returns></returns>
    public static Tensor[] quantize_v2(Tensor input, Tensor min_range, Tensor max_range, TF_DataType T, string mode = "MIN_COMBINED", string round_mode = "HALF_AWAY_FROM_ZERO", bool narrow_range = false, int axis = -1, float ensure_minimum_range = 0.01f, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizeV2", name) { args = new object[] { input, min_range, max_range }, attrs = new Dictionary<string, object>() { ["T"] = T, ["mode"] = mode, ["round_mode"] = round_mode, ["narrow_range"] = narrow_range, ["axis"] = axis, ["ensure_minimum_range"] = ensure_minimum_range } });
                return _fast_path_result;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return quantize_v2_eager_fallback(input, min_range, max_range, T: T, mode: mode, round_mode: round_mode, narrow_range: narrow_range, axis: axis, ensure_minimum_range: ensure_minimum_range, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        if (mode is null)
        {
            mode = "MIN_COMBINED";
        }
        if (round_mode is null)
        {
            round_mode = "HALF_AWAY_FROM_ZERO";
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["min_range"] = min_range;
        keywords["max_range"] = max_range;
        keywords["T"] = T;
        keywords["mode"] = mode;
        keywords["round_mode"] = round_mode;
        keywords["narrow_range"] = narrow_range;
        keywords["axis"] = axis;
        keywords["ensure_minimum_range"] = ensure_minimum_range;
        var _op = tf.OpDefLib._apply_op_helper("QuantizeV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "mode", _op.get_attr("mode"), "round_mode", _op.get_attr("round_mode"), "narrow_range", _op._get_attr_bool("narrow_range"), "axis", _op._get_attr_int("axis"), "ensure_minimum_range", _op.get_attr("ensure_minimum_range") };
            _execute.record_gradient("QuantizeV2", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantize_v2_eager_fallback(Tensor input, Tensor min_range, Tensor max_range, TF_DataType T, string mode, string round_mode, bool narrow_range, int axis, float ensure_minimum_range, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, min_range, max_range };
        object[] _attrs = new object[] { "T", T, "mode", mode, "round_mode", round_mode, "narrow_range", narrow_range, "axis", axis, "ensure_minimum_range", ensure_minimum_range };
        var _result = _execute.execute("QuantizeV2", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizeV2", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Concatenates quantized tensors along one dimension.
    /// </summary>
    /// <param name="concat_dim"></param>
    /// <param name="values"></param>
    /// <param name="input_mins"></param>
    /// <param name="input_maxes"></param>
    /// <returns></returns>
    public static Tensor[] quantized_concat(Tensor concat_dim, Tensors values, Tensors input_mins, Tensors input_maxes, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedConcat", name) { args = new object[] { concat_dim, values, input_mins, input_maxes }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return quantized_concat_eager_fallback(concat_dim, values, input_mins, input_maxes, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["concat_dim"] = concat_dim;
        keywords["values"] = values;
        keywords["input_mins"] = input_mins;
        keywords["input_maxes"] = input_maxes;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedConcat", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "N", _op._get_attr_int("N"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("QuantizedConcat", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_concat_eager_fallback(Tensor concat_dim, Tensors values, Tensors input_mins, Tensors input_maxes, string name, Context ctx)
    {
        List<Tensor> _inputs_flat_list = new();
        _inputs_flat_list.Add(concat_dim);
        _inputs_flat_list.AddRange(values);
        _inputs_flat_list.AddRange(input_mins);
        _inputs_flat_list.AddRange(input_maxes);
        var _inputs_flat = _inputs_flat_list.ToArray();
        object[] _attrs = new object[] { "N", values.Length, "T", values.dtype };
        var _result = _execute.execute("QuantizedConcat", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedConcat", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Quantized Instance normalization.
    /// </summary>
    /// <param name="x"></param>
    /// <param name="x_min"></param>
    /// <param name="x_max"></param>
    /// <param name="output_range_given">
    /// 
    /// If True, `given_y_min` and `given_y_min`
    /// and `given_y_max` are used as the output range. Otherwise,
    /// the implementation computes the output range.
    /// 
    /// </param>
    /// <param name="given_y_min">
    /// 
    /// Output in `y_min` if `output_range_given` is True.
    /// 
    /// </param>
    /// <param name="given_y_max">
    /// 
    /// Output in `y_max` if `output_range_given` is True.
    /// 
    /// </param>
    /// <param name="variance_epsilon">
    /// 
    /// A small float number to avoid dividing by 0.
    /// 
    /// </param>
    /// <param name="min_separation">
    /// 
    /// Minimum value of `y_max - y_min`
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] quantized_instance_norm(Tensor x, Tensor x_min, Tensor x_max, bool output_range_given = false, float given_y_min = 0f, float given_y_max = 0f, float variance_epsilon = 1E-05f, float min_separation = 0.001f, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedInstanceNorm", name) { args = new object[] { x, x_min, x_max }, attrs = new Dictionary<string, object>() { ["output_range_given"] = output_range_given, ["given_y_min"] = given_y_min, ["given_y_max"] = given_y_max, ["variance_epsilon"] = variance_epsilon, ["min_separation"] = min_separation } });
                return _fast_path_result;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return quantized_instance_norm_eager_fallback(x, x_min, x_max, output_range_given: output_range_given, given_y_min: given_y_min, given_y_max: given_y_max, variance_epsilon: variance_epsilon, min_separation: min_separation, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["x_min"] = x_min;
        keywords["x_max"] = x_max;
        keywords["output_range_given"] = output_range_given;
        keywords["given_y_min"] = given_y_min;
        keywords["given_y_max"] = given_y_max;
        keywords["variance_epsilon"] = variance_epsilon;
        keywords["min_separation"] = min_separation;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedInstanceNorm", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "output_range_given", _op._get_attr_bool("output_range_given"), "given_y_min", _op.get_attr("given_y_min"), "given_y_max", _op.get_attr("given_y_max"), "variance_epsilon", _op.get_attr("variance_epsilon"), "min_separation", _op.get_attr("min_separation") };
            _execute.record_gradient("QuantizedInstanceNorm", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_instance_norm_eager_fallback(Tensor x, Tensor x_min, Tensor x_max, bool output_range_given, float given_y_min, float given_y_max, float variance_epsilon, float min_separation, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, x_min, x_max };
        object[] _attrs = new object[] { "T", x.dtype, "output_range_given", output_range_given, "given_y_min", given_y_min, "given_y_max", given_y_max, "variance_epsilon", variance_epsilon, "min_separation", min_separation };
        var _result = _execute.execute("QuantizedInstanceNorm", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedInstanceNorm", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Reshapes a quantized tensor as per the Reshape op.
    /// </summary>
    /// <remarks>
    /// 
    /// ```
    /// 
    /// </remarks>
    /// <param name="tensor"></param>
    /// <param name="shape"></param>
    /// <param name="input_min"></param>
    /// <param name="input_max"></param>
    /// <returns></returns>
    public static Tensor[] quantized_reshape(Tensor tensor, Tensor shape, Tensor input_min, Tensor input_max, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedReshape", name) { args = new object[] { tensor, shape, input_min, input_max }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return quantized_reshape_eager_fallback(tensor, shape, input_min, input_max, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["tensor"] = tensor;
        keywords["shape"] = shape;
        keywords["input_min"] = input_min;
        keywords["input_max"] = input_max;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedReshape", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tshape", _op._get_attr_type("Tshape") };
            _execute.record_gradient("QuantizedReshape", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_reshape_eager_fallback(Tensor tensor, Tensor shape, Tensor input_min, Tensor input_max, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { tensor, shape, input_min, input_max };
        object[] _attrs = new object[] { "T", tensor.dtype, "Tshape", shape.dtype };
        var _result = _execute.execute("QuantizedReshape", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedReshape", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Returns the rank of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation returns an integer representing the rank of `input`.
    /// 
    /// For example:
    /// 
    /// ```
    /// # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
    /// # shape of tensor 't' is [2, 2, 3]
    /// rank(t) ==> 3
    /// ```
    /// 
    /// **Note**: The rank of a tensor is not the same as the rank of a matrix. The rank
    /// of a tensor is the number of indices required to uniquely select each element
    /// of the tensor. Rank is also known as "order", "degree", or "ndims."
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <returns></returns>
    public static Tensor rank(Tensor input, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Rank", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return rank_eager_fallback(input, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        var _op = tf.OpDefLib._apply_op_helper("Rank", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Rank", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor rank_eager_fallback(Tensor input, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "T", input.dtype };
        var _result = _execute.execute("Rank", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Rank", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Return the same ref tensor as the input ref tensor.
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public static Tensor ref_identity(Tensor input, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            throw new RuntimeError("ref_identity op does not support eager execution. Arg input is a ref.");
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        var _op = tf.OpDefLib._apply_op_helper("RefIdentity", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("RefIdentity", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor ref_identity_eager_fallback(Tensor input, string name, Context ctx)
    {
        throw new RuntimeError($"ref_identity op does not support eager execution. Arg 'input' is a ref.");
    }
    /// <summary>
    /// Reshapes a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Given `tensor`, this operation returns a tensor that has the same values
    /// as `tensor` with shape `shape`.
    /// 
    /// If one component of 1-D tensor `shape` is the special value -1, the size of that
    /// dimension is computed so that the total size remains constant.  In particular, a
    /// `shape` of `[-1]` flattens into 1-D.  At most one component of `shape` may be
    /// unknown.
    /// 
    /// The `shape` must be 1-D and the operation returns a tensor with shape
    /// `shape` filled with the values of `tensor`. In this case, the number of elements
    /// implied by `shape` must be the same as the number of elements in `tensor`.
    /// 
    /// It is an error if `shape` is not 1-D.
    /// 
    /// For example:
    /// 
    /// ```
    /// # tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
    /// # tensor 't' has shape [9]
    /// reshape(t, [3, 3]) ==> [[1, 2, 3],
    ///                         [4, 5, 6],
    ///                         [7, 8, 9]]
    /// 
    /// # tensor 't' is [[[1, 1], [2, 2]],
    /// #                [[3, 3], [4, 4]]]
    /// # tensor 't' has shape [2, 2, 2]
    /// reshape(t, [2, 4]) ==> [[1, 1, 2, 2],
    ///                         [3, 3, 4, 4]]
    /// 
    /// # tensor 't' is [[[1, 1, 1],
    /// #                 [2, 2, 2]],
    /// #                [[3, 3, 3],
    /// #                 [4, 4, 4]],
    /// #                [[5, 5, 5],
    /// #                 [6, 6, 6]]]
    /// # tensor 't' has shape [3, 2, 3]
    /// # pass '[-1]' to flatten 't'
    /// reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]
    /// 
    /// # -1 can also be used to infer the shape
    /// 
    /// # -1 is inferred to be 9:
    /// reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
    ///                          [4, 4, 4, 5, 5, 5, 6, 6, 6]]
    /// # -1 is inferred to be 2:
    /// reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
    ///                          [4, 4, 4, 5, 5, 5, 6, 6, 6]]
    /// # -1 is inferred to be 3:
    /// reshape(t, [ 2, -1, 3]) ==> [[[1, 1, 1],
    ///                               [2, 2, 2],
    ///                               [3, 3, 3]],
    ///                              [[4, 4, 4],
    ///                               [5, 5, 5],
    ///                               [6, 6, 6]]]
    /// 
    /// # tensor 't' is [7]
    /// # shape `[]` reshapes to a scalar
    /// reshape(t, []) ==> 7
    /// ```
    /// 
    /// </remarks>
    /// <param name="tensor"></param>
    /// <param name="shape"></param>
    /// <returns></returns>
    public static Tensor reshape(Tensor tensor, Tensor shape, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Reshape", name) { args = new object[] { tensor, shape }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return reshape_eager_fallback(tensor, shape, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["tensor"] = tensor;
        keywords["shape"] = shape;
        var _op = tf.OpDefLib._apply_op_helper("Reshape", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tshape", _op._get_attr_type("Tshape") };
            _execute.record_gradient("Reshape", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor reshape_eager_fallback(Tensor tensor, Tensor shape, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { tensor, shape };
        object[] _attrs = new object[] { "T", tensor.dtype, "Tshape", shape.dtype };
        var _result = _execute.execute("Reshape", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Reshape", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Assign `value` to the sliced l-value reference of `ref`.
    /// </summary>
    /// <remarks>
    /// 
    /// The values of `value` are assigned to the positions in the variable
    /// `ref` that are selected by the slice parameters. The slice parameters
    /// `begin, `end`, `strides`, etc. work exactly as in `StridedSlice`.
    /// 
    /// NOTE this op currently does not support broadcasting and so `value`'s
    /// shape must be exactly the shape produced by the slice of `ref`.
    /// 
    /// </remarks>
    /// <param name="ref_"></param>
    /// <param name="begin"></param>
    /// <param name="end"></param>
    /// <param name="strides"></param>
    /// <param name="value"></param>
    /// <param name="begin_mask"></param>
    /// <param name="end_mask"></param>
    /// <param name="ellipsis_mask"></param>
    /// <param name="new_axis_mask"></param>
    /// <param name="shrink_axis_mask"></param>
    /// <returns></returns>
    public static Operation resource_strided_slice_assign(Tensor ref_, Tensor begin, Tensor end, Tensor strides, Tensor value, int begin_mask = 0, int end_mask = 0, int ellipsis_mask = 0, int new_axis_mask = 0, int shrink_axis_mask = 0, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ResourceStridedSliceAssign", name) { args = new object[] { ref_, begin, end, strides, value }, attrs = new Dictionary<string, object>() { ["begin_mask"] = begin_mask, ["end_mask"] = end_mask, ["ellipsis_mask"] = ellipsis_mask, ["new_axis_mask"] = new_axis_mask, ["shrink_axis_mask"] = shrink_axis_mask } });
                return null;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return resource_strided_slice_assign_eager_fallback(ref_, begin, end, strides, value, begin_mask: begin_mask, end_mask: end_mask, ellipsis_mask: ellipsis_mask, new_axis_mask: new_axis_mask, shrink_axis_mask: shrink_axis_mask, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["ref"] = ref_;
        keywords["begin"] = begin;
        keywords["end"] = end;
        keywords["strides"] = strides;
        keywords["value"] = value;
        keywords["begin_mask"] = begin_mask;
        keywords["end_mask"] = end_mask;
        keywords["ellipsis_mask"] = ellipsis_mask;
        keywords["new_axis_mask"] = new_axis_mask;
        keywords["shrink_axis_mask"] = shrink_axis_mask;
        var _op = tf.OpDefLib._apply_op_helper("ResourceStridedSliceAssign", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Index", _op._get_attr_type("Index"), "begin_mask", _op._get_attr_int("begin_mask"), "end_mask", _op._get_attr_int("end_mask"), "ellipsis_mask", _op._get_attr_int("ellipsis_mask"), "new_axis_mask", _op._get_attr_int("new_axis_mask"), "shrink_axis_mask", _op._get_attr_int("shrink_axis_mask") };
            _execute.record_gradient("ResourceStridedSliceAssign", _op.inputs, _attrs, _result);
        }
        return _op;
    }

    public static Operation resource_strided_slice_assign_eager_fallback(Tensor ref_, Tensor begin, Tensor end, Tensor strides, Tensor value, int begin_mask, int end_mask, int ellipsis_mask, int new_axis_mask, int shrink_axis_mask, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { ref_, begin, end, strides, value };
        object[] _attrs = new object[] { "T", value.dtype, "Index", begin.dtype, "begin_mask", begin_mask, "end_mask", end_mask, "ellipsis_mask", ellipsis_mask, "new_axis_mask", new_axis_mask, "shrink_axis_mask", shrink_axis_mask };
        var _result = _execute.execute("ResourceStridedSliceAssign", 0, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ResourceStridedSliceAssign", _inputs_flat, _attrs, _result);
        }
        return null;
    }
    /// <summary>
    /// Reverses specific dimensions of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Given a `tensor`, and a `bool` tensor `dims` representing the dimensions
    /// of `tensor`, this operation reverses each dimension i of `tensor` where
    /// `dims[i]` is `True`.
    /// 
    /// `tensor` can have up to 8 dimensions. The number of dimensions
    /// of `tensor` must equal the number of elements in `dims`. In other words:
    /// 
    /// `rank(tensor) = size(dims)`
    /// 
    /// For example:
    /// 
    /// ```
    /// # tensor 't' is [[[[ 0,  1,  2,  3],
    /// #                  [ 4,  5,  6,  7],
    /// #                  [ 8,  9, 10, 11]],
    /// #                 [[12, 13, 14, 15],
    /// #                  [16, 17, 18, 19],
    /// #                  [20, 21, 22, 23]]]]
    /// # tensor 't' shape is [1, 2, 3, 4]
    /// 
    /// # 'dims' is [False, False, False, True]
    /// reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
    ///                         [ 7,  6,  5,  4],
    ///                         [ 11, 10, 9, 8]],
    ///                        [[15, 14, 13, 12],
    ///                         [19, 18, 17, 16],
    ///                         [23, 22, 21, 20]]]]
    /// 
    /// # 'dims' is [False, True, False, False]
    /// reverse(t, dims) ==> [[[[12, 13, 14, 15],
    ///                         [16, 17, 18, 19],
    ///                         [20, 21, 22, 23]
    ///                        [[ 0,  1,  2,  3],
    ///                         [ 4,  5,  6,  7],
    ///                         [ 8,  9, 10, 11]]]]
    /// 
    /// # 'dims' is [False, False, True, False]
    /// reverse(t, dims) ==> [[[[8, 9, 10, 11],
    ///                         [4, 5, 6, 7],
    ///                         [0, 1, 2, 3]]
    ///                        [[20, 21, 22, 23],
    ///                         [16, 17, 18, 19],
    ///                         [12, 13, 14, 15]]]]
    /// ```
    /// 
    /// </remarks>
    /// <param name="tensor"></param>
    /// <param name="dims"></param>
    /// <returns></returns>
    public static Tensor reverse(Tensor tensor, Tensor dims, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Reverse", name) { args = new object[] { tensor, dims }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return reverse_eager_fallback(tensor, dims, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["tensor"] = tensor;
        keywords["dims"] = dims;
        var _op = tf.OpDefLib._apply_op_helper("Reverse", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Reverse", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor reverse_eager_fallback(Tensor tensor, Tensor dims, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { tensor, dims };
        object[] _attrs = new object[] { "T", tensor.dtype };
        var _result = _execute.execute("Reverse", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Reverse", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Reverses variable length slices.
    /// </summary>
    /// <remarks>
    /// 
    /// This op first slices `input` along the dimension `batch_dim`, and for each
    /// slice `i`, reverses the first `seq_lengths[i]` elements along
    /// the dimension `seq_dim`.
    /// 
    /// The elements of `seq_lengths` must obey `seq_lengths[i] <= input.dims[seq_dim]`,
    /// and `seq_lengths` must be a vector of length `input.dims[batch_dim]`.
    /// 
    /// The output slice `i` along dimension `batch_dim` is then given by input
    /// slice `i`, with the first `seq_lengths[i]` slices along dimension
    /// `seq_dim` reversed.
    /// 
    /// For example:
    /// 
    /// ```
    /// # Given this:
    /// batch_dim = 0
    /// seq_dim = 1
    /// input.dims = (4, 8, ...)
    /// seq_lengths = [7, 2, 3, 5]
    /// 
    /// # then slices of input are reversed on seq_dim, but only up to seq_lengths:
    /// output[0, 0:7, :, ...] = input[0, 7:0:-1, :, ...]
    /// output[1, 0:2, :, ...] = input[1, 2:0:-1, :, ...]
    /// output[2, 0:3, :, ...] = input[2, 3:0:-1, :, ...]
    /// output[3, 0:5, :, ...] = input[3, 5:0:-1, :, ...]
    /// 
    /// # while entries past seq_lens are copied through:
    /// output[0, 7:, :, ...] = input[0, 7:, :, ...]
    /// output[1, 2:, :, ...] = input[1, 2:, :, ...]
    /// output[2, 3:, :, ...] = input[2, 3:, :, ...]
    /// output[3, 2:, :, ...] = input[3, 2:, :, ...]
    /// ```
    /// 
    /// In contrast, if:
    /// 
    /// ```
    /// # Given this:
    /// batch_dim = 2
    /// seq_dim = 0
    /// input.dims = (8, ?, 4, ...)
    /// seq_lengths = [7, 2, 3, 5]
    /// 
    /// # then slices of input are reversed on seq_dim, but only up to seq_lengths:
    /// output[0:7, :, 0, :, ...] = input[7:0:-1, :, 0, :, ...]
    /// output[0:2, :, 1, :, ...] = input[2:0:-1, :, 1, :, ...]
    /// output[0:3, :, 2, :, ...] = input[3:0:-1, :, 2, :, ...]
    /// output[0:5, :, 3, :, ...] = input[5:0:-1, :, 3, :, ...]
    /// 
    /// # while entries past seq_lens are copied through:
    /// output[7:, :, 0, :, ...] = input[7:, :, 0, :, ...]
    /// output[2:, :, 1, :, ...] = input[2:, :, 1, :, ...]
    /// output[3:, :, 2, :, ...] = input[3:, :, 2, :, ...]
    /// output[2:, :, 3, :, ...] = input[2:, :, 3, :, ...]
    /// ```
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="seq_lengths"></param>
    /// <param name="seq_dim">
    /// 
    /// The dimension which is partially reversed.
    /// 
    /// </param>
    /// <param name="batch_dim">
    /// 
    /// The dimension along which reversal is performed.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor reverse_sequence(Tensor input, Tensor seq_lengths, int seq_dim = 0, int batch_dim = 0, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ReverseSequence", name) { args = new object[] { input, seq_lengths }, attrs = new Dictionary<string, object>() { ["seq_dim"] = seq_dim, ["batch_dim"] = batch_dim } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return reverse_sequence_eager_fallback(input, seq_lengths, seq_dim: seq_dim, batch_dim: batch_dim, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["seq_lengths"] = seq_lengths;
        keywords["seq_dim"] = seq_dim;
        keywords["batch_dim"] = batch_dim;
        var _op = tf.OpDefLib._apply_op_helper("ReverseSequence", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "seq_dim", _op._get_attr_int("seq_dim"), "batch_dim", _op._get_attr_int("batch_dim"), "T", _op._get_attr_type("T"), "Tlen", _op._get_attr_type("Tlen") };
            _execute.record_gradient("ReverseSequence", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor reverse_sequence_eager_fallback(Tensor input, Tensor seq_lengths, int seq_dim, int batch_dim, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, seq_lengths };
        object[] _attrs = new object[] { "seq_dim", seq_dim, "batch_dim", batch_dim, "T", input.dtype, "Tlen", seq_lengths.dtype };
        var _result = _execute.execute("ReverseSequence", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ReverseSequence", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Reverses specific dimensions of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Given a `tensor`, and a `int32` tensor `axis` representing the set of
    /// dimensions of `tensor` to reverse. This operation reverses each dimension
    /// `i` for which there exists `j` s.t. `axis[j] == i`.
    /// 
    /// `tensor` can have up to 8 dimensions. The number of dimensions specified
    /// in `axis` may be 0 or more entries. If an index is specified more than
    /// once, a InvalidArgument error is raised.
    /// 
    /// For example:
    /// 
    /// ```
    /// # tensor 't' is [[[[ 0,  1,  2,  3],
    /// #                  [ 4,  5,  6,  7],
    /// #                  [ 8,  9, 10, 11]],
    /// #                 [[12, 13, 14, 15],
    /// #                  [16, 17, 18, 19],
    /// #                  [20, 21, 22, 23]]]]
    /// # tensor 't' shape is [1, 2, 3, 4]
    /// 
    /// # 'dims' is [3] or 'dims' is [-1]
    /// reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
    ///                         [ 7,  6,  5,  4],
    ///                         [ 11, 10, 9, 8]],
    ///                        [[15, 14, 13, 12],
    ///                         [19, 18, 17, 16],
    ///                         [23, 22, 21, 20]]]]
    /// 
    /// # 'dims' is '[1]' (or 'dims' is '[-3]')
    /// reverse(t, dims) ==> [[[[12, 13, 14, 15],
    ///                         [16, 17, 18, 19],
    ///                         [20, 21, 22, 23]
    ///                        [[ 0,  1,  2,  3],
    ///                         [ 4,  5,  6,  7],
    ///                         [ 8,  9, 10, 11]]]]
    /// 
    /// # 'dims' is '[2]' (or 'dims' is '[-2]')
    /// reverse(t, dims) ==> [[[[8, 9, 10, 11],
    ///                         [4, 5, 6, 7],
    ///                         [0, 1, 2, 3]]
    ///                        [[20, 21, 22, 23],
    ///                         [16, 17, 18, 19],
    ///                         [12, 13, 14, 15]]]]
    /// ```
    /// 
    /// </remarks>
    /// <param name="tensor"></param>
    /// <param name="axis"></param>
    /// <returns></returns>
    public static Tensor reverse_v2(Tensor tensor, Tensor axis, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ReverseV2", name) { args = new object[] { tensor, axis }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return reverse_v2_eager_fallback(tensor, axis, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["tensor"] = tensor;
        keywords["axis"] = axis;
        var _op = tf.OpDefLib._apply_op_helper("ReverseV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tidx", _op._get_attr_type("Tidx"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("ReverseV2", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor reverse_v2_eager_fallback(Tensor tensor, Tensor axis, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { tensor, axis };
        object[] _attrs = new object[] { "Tidx", axis.dtype, "T", tensor.dtype };
        var _result = _execute.execute("ReverseV2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ReverseV2", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Scatters `updates` into a tensor of shape `shape` according to `indices`.
    /// </summary>
    /// <remarks>
    /// 
    /// Scatter sparse `updates` according to individual values at the specified
    /// `indices`. This op returns an output tensor with the `shape` you specify. This
    /// op is the inverse of the `tf.gather_nd` operator which extracts values or slices
    /// from a given tensor.
    /// 
    /// This operation is similar to `tf.tensor_scatter_nd_add`, except that the tensor
    /// is zero-initialized. Calling `tf.scatter_nd(indices, updates, shape)`
    /// is identical to calling
    /// `tf.tensor_scatter_nd_add(tf.zeros(shape, updates.dtype), indices, updates)`
    /// 
    /// If `indices` contains duplicates, the associated `updates` are accumulated
    /// (summed) into the output tensor.
    /// 
    /// **WARNING**: For floating-point data types, the output may be nondeterministic.
    /// This is because the order in which the updates are applied is nondeterministic
    /// and when floating-point numbers are added in different orders the resulting
    /// numerical approximation error can be slightly different. However, the output
    /// will be deterministic if op determinism is enabled via
    /// `tf.config.experimental.enable_op_determinism`.
    /// 
    /// `indices` is an integer tensor containing indices into the output tensor. The
    /// last dimension of `indices` can be at most the rank of `shape`:
    /// 
    ///     indices.shape[-1] <= shape.rank
    /// 
    /// The last dimension of `indices` corresponds to indices of elements
    /// (if `indices.shape[-1] = shape.rank`) or slices
    /// (if `indices.shape[-1] < shape.rank`) along dimension `indices.shape[-1]` of
    /// `shape`.
    /// 
    /// `updates` is a tensor with shape:
    /// 
    ///     indices.shape[:-1] + shape[indices.shape[-1]:]
    /// 
    /// The simplest form of the scatter op is to insert individual elements in
    /// a tensor by index. Consider an example where you want to insert 4 scattered
    /// elements in a rank-1 tensor with 8 elements.
    /// 
    /// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
    /// <img style="width:100%" src="https://www.tensorflow.org/images/ScatterNd1.png" alt>
    /// </div>
    /// 
    /// In Python, this scatter operation would look like this:
    /// 
    /// ```python
    ///     indices = tf.constant([[4], [3], [1], [7]])
    ///     updates = tf.constant([9, 10, 11, 12])
    ///     shape = tf.constant([8])
    ///     scatter = tf.scatter_nd(indices, updates, shape)
    ///     print(scatter)
    /// ```
    /// 
    /// The resulting tensor would look like this:
    /// 
    ///     [0, 11, 0, 10, 9, 0, 0, 12]
    /// 
    /// You can also insert entire slices of a higher rank tensor all at once. For
    /// example, you can insert two slices in the first dimension of a rank-3 tensor
    /// with two matrices of new values.
    /// 
    /// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
    /// <img style="width:100%" src="https://www.tensorflow.org/images/ScatterNd2.png" alt>
    /// </div>
    /// 
    /// In Python, this scatter operation would look like this:
    /// 
    /// ```python
    ///     indices = tf.constant([[1], [3]])
    ///     updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
    ///                             [7, 7, 7, 7], [8, 8, 8, 8]],
    ///                            [[5, 5, 5, 5], [6, 6, 6, 6],
    ///                             [7, 7, 7, 7], [8, 8, 8, 8]]])
    ///     shape = tf.constant([4, 4, 4])
    ///     scatter = tf.scatter_nd(indices, updates, shape)
    ///     print(scatter)
    /// ```
    /// 
    /// The resulting tensor would look like this:
    /// 
    ///     [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    ///      [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
    ///      [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    ///      [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]]
    /// 
    /// Note that on CPU, if an out of bound index is found, an error is returned.
    /// On GPU, if an out of bound index is found, the index is ignored.
    /// 
    /// </remarks>
    /// <param name="indices"></param>
    /// <param name="updates"></param>
    /// <param name="shape"></param>
    /// <returns></returns>
    public static Tensor scatter_nd(Tensor indices, Tensor updates, Tensor shape, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ScatterNd", name) { args = new object[] { indices, updates, shape }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return scatter_nd_eager_fallback(indices, updates, shape, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["indices"] = indices;
        keywords["updates"] = updates;
        keywords["shape"] = shape;
        var _op = tf.OpDefLib._apply_op_helper("ScatterNd", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tindices", _op._get_attr_type("Tindices") };
            _execute.record_gradient("ScatterNd", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor scatter_nd_eager_fallback(Tensor indices, Tensor updates, Tensor shape, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { indices, updates, shape };
        object[] _attrs = new object[] { "T", updates.dtype, "Tindices", indices.dtype };
        var _result = _execute.execute("ScatterNd", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ScatterNd", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Applies sparse addition to `input` using individual values or slices
    /// </summary>
    /// <remarks>
    /// 
    /// from `updates` according to indices `indices`.  The updates are non-aliasing:
    /// `input` is only modified in-place if no other operations will use it.
    /// Otherwise, a copy of `input` is made.  This operation has a gradient with
    /// respect to both `input` and `updates`.
    /// 
    /// `input` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.
    /// 
    /// `indices` must be integer tensor, containing indices into `input`.
    /// It must be shape \([d_0, ..., d_{Q-2}, K]\) where `0 < K <= P`.
    /// 
    /// The innermost dimension of `indices` (with length `K`) corresponds to
    /// indices into elements (if `K = P`) or `(P-K)`-dimensional slices
    /// (if `K < P`) along the `K`th dimension of `input`.
    /// 
    /// `updates` is `Tensor` of rank `Q-1+P-K` with shape:
    /// 
    /// $$[d_0, ..., d_{Q-2}, input.shape[K], ..., input.shape[P-1]].$$
    /// 
    /// For example, say we want to add 4 scattered elements to a rank-1 tensor to 8
    /// elements. In Python, that addition would look like this:
    /// 
    ///     input = tf.constant([1, 2, 3, 4, 5, 6, 7, 8])
    ///     indices = tf.constant([[4], [3], [1], [7]])
    ///     updates = tf.constant([9, 10, 11, 12])
    ///     output = tf.scatter_nd_non_aliasing_add(input, indices, updates)
    ///     with tf.Session() as sess:
    ///       print(sess.run(output))
    /// 
    /// The resulting value `output` would look like this:
    /// 
    ///     [1, 13, 3, 14, 14, 6, 7, 20]
    /// 
    /// See `tf.scatter_nd` for more details about how to make updates to slices.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="indices"></param>
    /// <param name="updates"></param>
    /// <returns></returns>
    public static Tensor scatter_nd_non_aliasing_add(Tensor input, Tensor indices, Tensor updates, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ScatterNdNonAliasingAdd", name) { args = new object[] { input, indices, updates }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return scatter_nd_non_aliasing_add_eager_fallback(input, indices, updates, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["indices"] = indices;
        keywords["updates"] = updates;
        var _op = tf.OpDefLib._apply_op_helper("ScatterNdNonAliasingAdd", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tindices", _op._get_attr_type("Tindices") };
            _execute.record_gradient("ScatterNdNonAliasingAdd", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor scatter_nd_non_aliasing_add_eager_fallback(Tensor input, Tensor indices, Tensor updates, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, indices, updates };
        object[] _attrs = new object[] { "T", input.dtype, "Tindices", indices.dtype };
        var _result = _execute.execute("ScatterNdNonAliasingAdd", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ScatterNdNonAliasingAdd", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns the shape of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation returns a 1-D integer tensor representing the shape of `input`.
    /// 
    /// For example:
    /// 
    /// ```
    /// # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
    /// shape(t) ==> [2, 2, 3]
    /// ```
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="out_type"></param>
    /// <returns></returns>
    public static Tensor shape(Tensor input, TF_DataType out_type = TF_DataType.TF_INT32, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Shape", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { ["out_type"] = out_type } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return shape_eager_fallback(input, out_type: out_type, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["out_type"] = out_type;
        var _op = tf.OpDefLib._apply_op_helper("Shape", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "out_type", _op._get_attr_type("out_type") };
            _execute.record_gradient("Shape", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor shape_eager_fallback(Tensor input, TF_DataType out_type, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "T", input.dtype, "out_type", out_type };
        var _result = _execute.execute("Shape", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Shape", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns shape of tensors.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation returns N 1-D integer tensors representing shape of `input[i]s`.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="out_type"></param>
    /// <returns></returns>
    public static Tensor[] shape_n(Tensors input, TF_DataType out_type = TF_DataType.TF_INT32, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ShapeN", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { ["out_type"] = out_type } });
                return _fast_path_result;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return shape_n_eager_fallback(input, out_type: out_type, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["out_type"] = out_type;
        var _op = tf.OpDefLib._apply_op_helper("ShapeN", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "N", _op._get_attr_int("N"), "T", _op._get_attr_type("T"), "out_type", _op._get_attr_type("out_type") };
            _execute.record_gradient("ShapeN", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] shape_n_eager_fallback(Tensors input, TF_DataType out_type, string name, Context ctx)
    {
        List<Tensor> _inputs_flat_list = new();
        _inputs_flat_list.AddRange(input);
        var _inputs_flat = _inputs_flat_list.ToArray();
        object[] _attrs = new object[] { "N", input.Length, "T", input.dtype, "out_type", out_type };
        var _result = _execute.execute("ShapeN", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ShapeN", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Returns the size of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation returns an integer representing the number of elements in
    /// `input`.
    /// 
    /// For example:
    /// 
    /// ```
    /// # 't' is [[[1, 1,, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]]
    /// size(t) ==> 12
    /// ```
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="out_type"></param>
    /// <returns></returns>
    public static Tensor size(Tensor input, TF_DataType out_type = TF_DataType.TF_INT32, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Size", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { ["out_type"] = out_type } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return size_eager_fallback(input, out_type: out_type, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["out_type"] = out_type;
        var _op = tf.OpDefLib._apply_op_helper("Size", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "out_type", _op._get_attr_type("out_type") };
            _execute.record_gradient("Size", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor size_eager_fallback(Tensor input, TF_DataType out_type, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "T", input.dtype, "out_type", out_type };
        var _result = _execute.execute("Size", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Size", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Return a slice from 'input'.
    /// </summary>
    /// <remarks>
    /// 
    /// The output tensor is a tensor with dimensions described by 'size'
    /// whose values are extracted from 'input' starting at the offsets in
    /// 'begin'.
    /// 
    /// *Requirements*:
    ///   0 <= begin[i] <= begin[i] + size[i] <= Di  for i in [0, n)
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="begin"></param>
    /// <param name="size"></param>
    /// <returns></returns>
    public static Tensor slice(Tensor input, Tensor begin, Tensor size, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Slice", name) { args = new object[] { input, begin, size }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return slice_eager_fallback(input, begin, size, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["begin"] = begin;
        keywords["size"] = size;
        var _op = tf.OpDefLib._apply_op_helper("Slice", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Index", _op._get_attr_type("Index") };
            _execute.record_gradient("Slice", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor slice_eager_fallback(Tensor input, Tensor begin, Tensor size, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, begin, size };
        object[] _attrs = new object[] { "T", input.dtype, "Index", begin.dtype };
        var _result = _execute.execute("Slice", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Slice", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns a copy of the input tensor.
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public static Tensor snapshot(Tensor input, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Snapshot", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return snapshot_eager_fallback(input, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        var _op = tf.OpDefLib._apply_op_helper("Snapshot", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Snapshot", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor snapshot_eager_fallback(Tensor input, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "T", input.dtype };
        var _result = _execute.execute("Snapshot", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Snapshot", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// SpaceToBatch for 4-D tensors of type T.
    /// </summary>
    /// <remarks>
    /// 
    /// This is a legacy version of the more general SpaceToBatchND.
    /// 
    /// Zero-pads and then rearranges (permutes) blocks of spatial data into batch.
    /// More specifically, this op outputs a copy of the input tensor where values from
    /// the `height` and `width` dimensions are moved to the `batch` dimension. After
    /// the zero-padding, both `height` and `width` of the input must be divisible by the
    /// block size.
    /// 
    /// The attr `block_size` must be greater than one. It indicates the block size.
    /// 
    ///   * Non-overlapping blocks of size `block_size x block size` in the height and
    ///     width dimensions are rearranged into the batch dimension at each location.
    ///   * The batch of the output tensor is `batch * block_size * block_size`.
    ///   * Both height_pad and width_pad must be divisible by block_size.
    /// 
    /// The shape of the output will be:
    /// 
    ///     [batch*block_size*block_size, height_pad/block_size, width_pad/block_size,
    ///      depth]
    /// 
    /// Some examples:
    /// 
    /// (1) For the following input of shape `[1, 2, 2, 1]` and block_size of 2:
    /// 
    /// ```
    /// x = [[[[1], [2]], [[3], [4]]]]
    /// ```
    /// 
    /// The output tensor has shape `[4, 1, 1, 1]` and value:
    /// 
    /// ```
    /// [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
    /// ```
    /// 
    /// (2) For the following input of shape `[1, 2, 2, 3]` and block_size of 2:
    /// 
    /// ```
    /// x = [[[[1, 2, 3], [4, 5, 6]],
    ///       [[7, 8, 9], [10, 11, 12]]]]
    /// ```
    /// 
    /// The output tensor has shape `[4, 1, 1, 3]` and value:
    /// 
    /// ```
    /// [[[[1, 2, 3]]], [[[4, 5, 6]]], [[[7, 8, 9]]], [[[10, 11, 12]]]]
    /// ```
    /// 
    /// (3) For the following input of shape `[1, 4, 4, 1]` and block_size of 2:
    /// 
    /// ```
    /// x = [[[[1],   [2],  [3],  [4]],
    ///       [[5],   [6],  [7],  [8]],
    ///       [[9],  [10], [11],  [12]],
    ///       [[13], [14], [15],  [16]]]]
    /// ```
    /// 
    /// The output tensor has shape `[4, 2, 2, 1]` and value:
    /// 
    /// ```
    /// x = [[[[1], [3]], [[9], [11]]],
    ///      [[[2], [4]], [[10], [12]]],
    ///      [[[5], [7]], [[13], [15]]],
    ///      [[[6], [8]], [[14], [16]]]]
    /// ```
    /// 
    /// (4) For the following input of shape `[2, 2, 4, 1]` and block_size of 2:
    /// 
    /// ```
    /// x = [[[[1],   [2],  [3],  [4]],
    ///       [[5],   [6],  [7],  [8]]],
    ///      [[[9],  [10], [11],  [12]],
    ///       [[13], [14], [15],  [16]]]]
    /// ```
    /// 
    /// The output tensor has shape `[8, 1, 2, 1]` and value:
    /// 
    /// ```
    /// x = [[[[1], [3]]], [[[9], [11]]], [[[2], [4]]], [[[10], [12]]],
    ///      [[[5], [7]]], [[[13], [15]]], [[[6], [8]]], [[[14], [16]]]]
    /// ```
    /// 
    /// Among others, this operation is useful for reducing atrous convolution into
    /// regular convolution.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="paddings"></param>
    /// <param name="block_size"></param>
    /// <returns></returns>
    public static Tensor space_to_batch(Tensor input, Tensor paddings, int block_size = 0, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SpaceToBatch", name) { args = new object[] { input, paddings }, attrs = new Dictionary<string, object>() { ["block_size"] = block_size } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return space_to_batch_eager_fallback(input, paddings, block_size: block_size, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["paddings"] = paddings;
        keywords["block_size"] = block_size;
        var _op = tf.OpDefLib._apply_op_helper("SpaceToBatch", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tpaddings", _op._get_attr_type("Tpaddings"), "block_size", _op._get_attr_int("block_size") };
            _execute.record_gradient("SpaceToBatch", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor space_to_batch_eager_fallback(Tensor input, Tensor paddings, int block_size, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, paddings };
        object[] _attrs = new object[] { "T", input.dtype, "Tpaddings", paddings.dtype, "block_size", block_size };
        var _result = _execute.execute("SpaceToBatch", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SpaceToBatch", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// SpaceToBatch for N-D tensors of type T.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation divides "spatial" dimensions `[1, ..., M]` of the input into a
    /// grid of blocks of shape `block_shape`, and interleaves these blocks with the
    /// "batch" dimension (0) such that in the output, the spatial dimensions
    /// `[1, ..., M]` correspond to the position within the grid, and the batch
    /// dimension combines both the position within a spatial block and the original
    /// batch position.  Prior to division into blocks, the spatial dimensions of the
    /// input are optionally zero padded according to `paddings`. See below for a
    /// precise description.
    /// 
    /// This operation is equivalent to the following steps:
    /// 
    /// 1. Zero-pad the start and end of dimensions `[1, ..., M]` of the
    ///    input according to `paddings` to produce `padded` of shape `padded_shape`.
    /// 
    /// 2. Reshape `padded` to `reshaped_padded` of shape:
    /// 
    ///      [batch] +
    ///      [padded_shape[1] / block_shape[0],
    ///        block_shape[0],
    ///       ...,
    ///       padded_shape[M] / block_shape[M-1],
    ///       block_shape[M-1]] +
    ///      remaining_shape
    /// 
    /// 3. Permute dimensions of `reshaped_padded` to produce
    ///    `permuted_reshaped_padded` of shape:
    /// 
    ///      block_shape +
    ///      [batch] +
    ///      [padded_shape[1] / block_shape[0],
    ///       ...,
    ///       padded_shape[M] / block_shape[M-1]] +
    ///      remaining_shape
    /// 
    /// 4. Reshape `permuted_reshaped_padded` to flatten `block_shape` into the batch
    ///    dimension, producing an output tensor of shape:
    /// 
    ///      [batch * prod(block_shape)] +
    ///      [padded_shape[1] / block_shape[0],
    ///       ...,
    ///       padded_shape[M] / block_shape[M-1]] +
    ///      remaining_shape
    /// 
    /// Some examples:
    /// 
    /// (1) For the following input of shape `[1, 2, 2, 1]`, `block_shape = [2, 2]`, and
    ///     `paddings = [[0, 0], [0, 0]]`:
    /// 
    /// ```
    /// x = [[[[1], [2]], [[3], [4]]]]
    /// ```
    /// 
    /// The output tensor has shape `[4, 1, 1, 1]` and value:
    /// 
    /// ```
    /// [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
    /// ```
    /// 
    /// (2) For the following input of shape `[1, 2, 2, 3]`, `block_shape = [2, 2]`, and
    ///     `paddings = [[0, 0], [0, 0]]`:
    /// 
    /// ```
    /// x = [[[[1, 2, 3], [4, 5, 6]],
    ///       [[7, 8, 9], [10, 11, 12]]]]
    /// ```
    /// 
    /// The output tensor has shape `[4, 1, 1, 3]` and value:
    /// 
    /// ```
    /// [[[[1, 2, 3]]], [[[4, 5, 6]]], [[[7, 8, 9]]], [[[10, 11, 12]]]]
    /// ```
    /// 
    /// (3) For the following input of shape `[1, 4, 4, 1]`, `block_shape = [2, 2]`, and
    ///     `paddings = [[0, 0], [0, 0]]`:
    /// 
    /// ```
    /// x = [[[[1],   [2],  [3],  [4]],
    ///       [[5],   [6],  [7],  [8]],
    ///       [[9],  [10], [11],  [12]],
    ///       [[13], [14], [15],  [16]]]]
    /// ```
    /// 
    /// The output tensor has shape `[4, 2, 2, 1]` and value:
    /// 
    /// ```
    /// x = [[[[1], [3]], [[9], [11]]],
    ///      [[[2], [4]], [[10], [12]]],
    ///      [[[5], [7]], [[13], [15]]],
    ///      [[[6], [8]], [[14], [16]]]]
    /// ```
    /// 
    /// (4) For the following input of shape `[2, 2, 4, 1]`, block_shape = `[2, 2]`, and
    ///     paddings = `[[0, 0], [2, 0]]`:
    /// 
    /// ```
    /// x = [[[[1],   [2],  [3],  [4]],
    ///       [[5],   [6],  [7],  [8]]],
    ///      [[[9],  [10], [11],  [12]],
    ///       [[13], [14], [15],  [16]]]]
    /// ```
    /// 
    /// The output tensor has shape `[8, 1, 3, 1]` and value:
    /// 
    /// ```
    /// x = [[[[0], [1], [3]]], [[[0], [9], [11]]],
    ///      [[[0], [2], [4]]], [[[0], [10], [12]]],
    ///      [[[0], [5], [7]]], [[[0], [13], [15]]],
    ///      [[[0], [6], [8]]], [[[0], [14], [16]]]]
    /// ```
    /// 
    /// Among others, this operation is useful for reducing atrous convolution into
    /// regular convolution.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="block_shape"></param>
    /// <param name="paddings"></param>
    /// <returns></returns>
    public static Tensor space_to_batch_nd(Tensor input, Tensor block_shape, Tensor paddings, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SpaceToBatchND", name) { args = new object[] { input, block_shape, paddings }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return space_to_batch_nd_eager_fallback(input, block_shape, paddings, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["block_shape"] = block_shape;
        keywords["paddings"] = paddings;
        var _op = tf.OpDefLib._apply_op_helper("SpaceToBatchND", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tblock_shape", _op._get_attr_type("Tblock_shape"), "Tpaddings", _op._get_attr_type("Tpaddings") };
            _execute.record_gradient("SpaceToBatchND", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor space_to_batch_nd_eager_fallback(Tensor input, Tensor block_shape, Tensor paddings, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, block_shape, paddings };
        object[] _attrs = new object[] { "T", input.dtype, "Tblock_shape", block_shape.dtype, "Tpaddings", paddings.dtype };
        var _result = _execute.execute("SpaceToBatchND", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SpaceToBatchND", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// SpaceToDepth for tensors of type T.
    /// </summary>
    /// <remarks>
    /// 
    /// Rearranges blocks of spatial data, into depth. More specifically,
    /// this op outputs a copy of the input tensor where values from the `height`
    /// and `width` dimensions are moved to the `depth` dimension.
    /// The attr `block_size` indicates the input block size.
    /// 
    ///   * Non-overlapping blocks of size `block_size x block size` are rearranged
    ///     into depth at each location.
    ///   * The depth of the output tensor is `block_size * block_size * input_depth`.
    ///   * The Y, X coordinates within each block of the input become the high order
    ///     component of the output channel index.
    ///   * The input tensor's height and width must be divisible by block_size.
    /// 
    /// The `data_format` attr specifies the layout of the input and output tensors
    /// with the following options:
    ///   "NHWC": `[ batch, height, width, channels ]`
    ///   "NCHW": `[ batch, channels, height, width ]`
    ///   "NCHW_VECT_C":
    ///       `qint8 [ batch, channels / 4, height, width, 4 ]`
    /// 
    /// It is useful to consider the operation as transforming a 6-D Tensor.
    /// e.g. for data_format = NHWC,
    ///      Each element in the input tensor can be specified via 6 coordinates,
    ///      ordered by decreasing memory layout significance as:
    ///      n,oY,bY,oX,bX,iC  (where n=batch index, oX, oY means X or Y coordinates
    ///                         within the output image, bX, bY means coordinates
    ///                         within the input block, iC means input channels).
    ///      The output would be a transpose to the following layout:
    ///      n,oY,oX,bY,bX,iC
    /// 
    /// This operation is useful for resizing the activations between convolutions
    /// (but keeping all data), e.g. instead of pooling. It is also useful for training
    /// purely convolutional models.
    /// 
    /// For example, given an input of shape `[1, 2, 2, 1]`, data_format = "NHWC" and
    /// block_size = 2:
    /// 
    /// ```
    /// x = [[[[1], [2]],
    ///       [[3], [4]]]]
    /// ```
    /// 
    /// This operation will output a tensor of shape `[1, 1, 1, 4]`:
    /// 
    /// ```
    /// [[[[1, 2, 3, 4]]]]
    /// ```
    /// 
    /// Here, the input has a batch of 1 and each batch element has shape `[2, 2, 1]`,
    /// the corresponding output will have a single element (i.e. width and height are
    /// both 1) and will have a depth of 4 channels (1 * block_size * block_size).
    /// The output element shape is `[1, 1, 4]`.
    /// 
    /// For an input tensor with larger depth, here of shape `[1, 2, 2, 3]`, e.g.
    /// 
    /// ```
    /// x = [[[[1, 2, 3], [4, 5, 6]],
    ///       [[7, 8, 9], [10, 11, 12]]]]
    /// ```
    /// 
    /// This operation, for block_size of 2, will return the following tensor of shape
    /// `[1, 1, 1, 12]`
    /// 
    /// ```
    /// [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
    /// ```
    /// 
    /// Similarly, for the following input of shape `[1 4 4 1]`, and a block size of 2:
    /// 
    /// ```
    /// x = [[[[1],   [2],  [5],  [6]],
    ///       [[3],   [4],  [7],  [8]],
    ///       [[9],  [10], [13],  [14]],
    ///       [[11], [12], [15],  [16]]]]
    /// ```
    /// 
    /// the operator will return the following tensor of shape `[1 2 2 4]`:
    /// 
    /// ```
    /// x = [[[[1, 2, 3, 4],
    ///        [5, 6, 7, 8]],
    ///       [[9, 10, 11, 12],
    ///        [13, 14, 15, 16]]]]
    /// ```
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="block_size">
    /// 
    /// The size of the spatial block.
    /// 
    /// </param>
    /// <param name="data_format"></param>
    /// <returns></returns>
    public static Tensor space_to_depth(Tensor input, int block_size = 0, string data_format = "NHWC", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SpaceToDepth", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { ["block_size"] = block_size, ["data_format"] = data_format } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return space_to_depth_eager_fallback(input, block_size: block_size, data_format: data_format, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        if (data_format is null)
        {
            data_format = "NHWC";
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["block_size"] = block_size;
        keywords["data_format"] = data_format;
        var _op = tf.OpDefLib._apply_op_helper("SpaceToDepth", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "block_size", _op._get_attr_int("block_size"), "data_format", _op.get_attr("data_format") };
            _execute.record_gradient("SpaceToDepth", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor space_to_depth_eager_fallback(Tensor input, int block_size, string data_format, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "T", input.dtype, "block_size", block_size, "data_format", data_format };
        var _result = _execute.execute("SpaceToDepth", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SpaceToDepth", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Splits a tensor into `num_split` tensors along one dimension.
    /// </summary>
    /// <param name="split_dim"></param>
    /// <param name="value"></param>
    /// <param name="num_split">
    /// 
    /// The number of ways to split.  Must evenly divide
    /// `value.shape[split_dim]`.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] split(Tensor split_dim, Tensor value, int num_split = 0, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Split", name) { args = new object[] { split_dim, value }, attrs = new Dictionary<string, object>() { ["num_split"] = num_split } });
                return _fast_path_result;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return split_eager_fallback(split_dim, value, num_split: num_split, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["split_dim"] = split_dim;
        keywords["value"] = value;
        keywords["num_split"] = num_split;
        var _op = tf.OpDefLib._apply_op_helper("Split", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "num_split", _op._get_attr_int("num_split"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("Split", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] split_eager_fallback(Tensor split_dim, Tensor value, int num_split, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { split_dim, value };
        object[] _attrs = new object[] { "num_split", num_split, "T", value.dtype };
        var _result = _execute.execute("Split", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Split", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Splits a tensor into `num_split` tensors along one dimension.
    /// </summary>
    /// <param name="value"></param>
    /// <param name="size_splits"></param>
    /// <param name="split_dim"></param>
    /// <param name="num_split"></param>
    /// <returns></returns>
    public static Tensor[] split_v(Tensor value, Tensor size_splits, Tensor split_dim, int num_split = 0, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SplitV", name) { args = new object[] { value, size_splits, split_dim }, attrs = new Dictionary<string, object>() { ["num_split"] = num_split } });
                return _fast_path_result;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return split_v_eager_fallback(value, size_splits, split_dim, num_split: num_split, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["value"] = value;
        keywords["size_splits"] = size_splits;
        keywords["split_dim"] = split_dim;
        keywords["num_split"] = num_split;
        var _op = tf.OpDefLib._apply_op_helper("SplitV", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "num_split", _op._get_attr_int("num_split"), "T", _op._get_attr_type("T"), "Tlen", _op._get_attr_type("Tlen") };
            _execute.record_gradient("SplitV", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] split_v_eager_fallback(Tensor value, Tensor size_splits, Tensor split_dim, int num_split, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { value, size_splits, split_dim };
        object[] _attrs = new object[] { "num_split", num_split, "T", value.dtype, "Tlen", size_splits.dtype };
        var _result = _execute.execute("SplitV", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SplitV", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Removes dimensions of size 1 from the shape of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Given a tensor `input`, this operation returns a tensor of the same type with
    /// all dimensions of size 1 removed. If you don't want to remove all size 1
    /// dimensions, you can remove specific size 1 dimensions by specifying
    /// `squeeze_dims`.
    /// 
    /// For example:
    /// 
    /// ```
    /// # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
    /// shape(squeeze(t)) ==> [2, 3]
    /// ```
    /// 
    /// Or, to remove specific size 1 dimensions:
    /// 
    /// ```
    /// # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
    /// shape(squeeze(t, [2, 4])) ==> [1, 2, 3, 1]
    /// ```
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="squeeze_dims">
    /// 
    /// If specified, only squeezes the dimensions listed. The dimension
    /// index starts at 0. It is an error to squeeze a dimension that is not 1. Must
    /// be in the range `[-rank(input), rank(input))`.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor squeeze(Tensor input, int[] squeeze_dims = null, string? name = null)
    {
        var _ctx = tf.Context;
        if (squeeze_dims is null)
        {
            squeeze_dims = new int[] { };
        }
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Squeeze", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { ["squeeze_dims"] = squeeze_dims } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return squeeze_eager_fallback(input, squeeze_dims: squeeze_dims, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["squeeze_dims"] = squeeze_dims;
        var _op = tf.OpDefLib._apply_op_helper("Squeeze", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "squeeze_dims", _op.get_attr("squeeze_dims") };
            _execute.record_gradient("Squeeze", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor squeeze_eager_fallback(Tensor input, int[] squeeze_dims, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "T", input.dtype, "squeeze_dims", squeeze_dims };
        var _result = _execute.execute("Squeeze", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Squeeze", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Stops gradient computation.
    /// </summary>
    /// <remarks>
    /// 
    /// When executed in a graph, this op outputs its input tensor as-is.
    /// 
    /// When building ops to compute gradients, this op prevents the contribution of
    /// its inputs to be taken into account.  Normally, the gradient generator adds ops
    /// to a graph to compute the derivatives of a specified 'loss' by recursively
    /// finding out inputs that contributed to its computation.  If you insert this op
    /// in the graph it inputs are masked from the gradient generator.  They are not
    /// taken into account for computing gradients.
    /// 
    /// This is useful any time you want to compute a value with TensorFlow but need
    /// to pretend that the value was a constant. For example, the softmax function
    /// for a vector x can be written as
    /// 
    /// ```python
    /// 
    ///   def softmax(x):
    ///     numerator = tf.exp(x)
    ///     denominator = tf.reduce_sum(numerator)
    ///     return numerator / denominator
    /// ```
    /// 
    /// This however is susceptible to overflow if the values in x are large. An
    /// alternative more stable way is to subtract the maximum of x from each of the
    /// values.
    /// 
    /// ```python
    /// 
    ///   def stable_softmax(x):
    ///     z = x - tf.reduce_max(x)
    ///     numerator = tf.exp(z)
    ///     denominator = tf.reduce_sum(numerator)
    ///     return numerator / denominator
    /// ```
    /// 
    /// However, when we backprop through the softmax to x, we dont want to backprop
    /// through the `tf.reduce_max(x)` (if the max values are not unique then the
    /// gradient could flow to the wrong input) calculation and treat that as a
    /// constant. Therefore, we should write this out as
    /// 
    /// ```python
    /// 
    ///   def stable_softmax(x):
    ///     z = x - tf.stop_gradient(tf.reduce_max(x))
    ///     numerator = tf.exp(z)
    ///     denominator = tf.reduce_sum(numerator)
    ///     return numerator / denominator
    /// ```
    /// 
    /// Some other examples include:
    /// 
    /// *  The *EM* algorithm where the *M-step* should not involve backpropagation
    ///    through the output of the *E-step*.
    /// *  Contrastive divergence training of Boltzmann machines where, when
    ///    differentiating the energy function, the training must not backpropagate
    ///    through the graph that generated the samples from the model.
    /// *  Adversarial training, where no backprop should happen through the adversarial
    ///    example generation process.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <returns></returns>
    public static Tensor stop_gradient(Tensor input, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "StopGradient", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return stop_gradient_eager_fallback(input, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        var _op = tf.OpDefLib._apply_op_helper("StopGradient", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("StopGradient", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor stop_gradient_eager_fallback(Tensor input, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "T", input.dtype };
        var _result = _execute.execute("StopGradient", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("StopGradient", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Return a strided slice from `input`.
    /// </summary>
    /// <remarks>
    /// 
    /// Note, most python users will want to use the Python `Tensor.__getitem__`
    /// or `Variable.__getitem__` rather than this op directly.
    /// 
    /// The goal of this op is to produce a new tensor with a subset of
    /// the elements from the `n` dimensional `input` tensor. The subset is chosen using
    /// a sequence of `m` sparse range specifications encoded into the arguments
    /// of this function. Note, in some cases
    /// `m` could be equal to `n`, but this need not be the case. Each
    /// range specification entry can be one of the following:
    /// 
    /// - An ellipsis (...). Ellipses are used to imply zero or more
    ///   dimensions of full-dimension selection and are produced using
    ///   `ellipsis_mask`. For example, `foo[...]` is the identity slice.
    /// 
    /// - A new axis. This is used to insert a new shape=1 dimension and is
    ///   produced using `new_axis_mask`. For example, `foo[:, ...]` where
    ///   `foo` is shape `(3, 4)` produces a `(1, 3, 4)` tensor.
    /// 
    /// 
    /// - A range `begin:end:stride`. This is used to specify how much to choose from
    ///   a given dimension. `stride` can be any integer but 0.  `begin` is an integer
    ///   which represents the index of the first value to select while `end` represents
    ///   the index of the last value to select. The number of values selected in each
    ///   dimension is `end - begin` if `stride > 0` and `begin - end` if `stride < 0`.
    ///   `begin` and `end` can be negative where `-1` is the last element, `-2` is
    ///   the second to last. `begin_mask` controls whether to replace the explicitly
    ///   given `begin` with an implicit effective value of `0` if `stride > 0` and
    ///   `-1` if `stride < 0`. `end_mask` is analogous but produces the number
    ///   required to create the largest open interval. For example, given a shape
    ///   `(3,)` tensor `foo[:]`, the effective `begin` and `end` are `0` and `3`. Do
    ///   not assume this is equivalent to `foo[0:-1]` which has an effective `begin`
    ///   and `end` of `0` and `2`. Another example is `foo[-2::-1]` which reverses the
    ///   first dimension of a tensor while dropping the last two (in the original
    ///   order elements). For example `foo = [1,2,3,4]; foo[-2::-1]` is `[4,3]`.
    /// 
    /// - A single index. This is used to keep only elements that have a given
    ///   index. For example (`foo[2, :]` on a shape `(5,6)` tensor produces a
    ///   shape `(6,)` tensor. This is encoded in `begin` and `end` and
    ///   `shrink_axis_mask`.
    /// 
    /// Each conceptual range specification is encoded in the op's argument. This
    /// encoding is best understand by considering a non-trivial example. In
    /// particular,
    /// `foo[1, 2:4, None, ..., :-3:-1, :]` will be encoded as
    /// 
    /// ```
    /// begin = [1, 2, x, x, 0, x] # x denotes don't care (usually 0)
    /// end = [2, 4, x, x, -3, x]
    /// strides = [1, 1, x, x, -1, 1]
    /// begin_mask = 1<<4 | 1<<5 = 48
    /// end_mask = 1<<5 = 32
    /// ellipsis_mask = 1<<3 = 8
    /// new_axis_mask = 1<<2 = 4
    /// shrink_axis_mask = 1<<0 = 1
    /// ```
    /// 
    /// In this case if `foo.shape` is (5, 5, 5, 5, 5, 5) the final shape of
    /// the slice becomes (2, 1, 5, 5, 2, 5).
    /// Let us walk step by step through each argument specification.
    /// 
    /// 1.  The first argument in the example slice is turned into `begin = 1` and
    /// `end = begin + 1 = 2`. To disambiguate from the original spec `2:4` we
    /// also set the appropriate bit in `shrink_axis_mask`.
    /// 
    /// 2. `2:4` is contributes 2, 4, 1 to begin, end, and stride. All masks have
    /// zero bits contributed.
    /// 
    /// 3. None is a synonym for `tf.newaxis`. This means insert a dimension of size 1
    /// dimension in the final shape. Dummy values are contributed to begin,
    /// end and stride, while the new_axis_mask bit is set.
    /// 
    /// 4. `...` grab the full ranges from as many dimensions as needed to
    /// fully specify a slice for every dimension of the input shape.
    /// 
    /// 5. `:-3:-1` shows the use of negative indices. A negative index `i` associated
    /// with a dimension that has shape `s` is converted to a positive index
    /// `s + i`. So `-1` becomes `s-1` (i.e. the last element). This conversion
    /// is done internally so begin, end and strides receive x, -3, and -1.
    /// The appropriate begin_mask bit is set to indicate the start range is the
    /// full range (ignoring the x).
    /// 
    /// 6. `:` indicates that the entire contents of the corresponding dimension
    /// is selected. This is equivalent to `::` or `0::1`. begin, end, and strides
    /// receive 0, 0, and 1, respectively. The appropriate bits in `begin_mask` and
    /// `end_mask` are also set.
    /// 
    /// *Requirements*:
    ///   `0 != strides[i] for i in [0, m)`
    ///   `ellipsis_mask must be a power of two (only one ellipsis)`
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="begin"></param>
    /// <param name="end"></param>
    /// <param name="strides"></param>
    /// <param name="begin_mask">
    /// 
    /// a bitmask where a bit i being 1 means to ignore the begin
    /// value and instead use the largest interval possible. At runtime
    /// begin[i] will be replaced with `[0, n-1)` if `stride[i] > 0` or
    /// `[-1, n-1]` if `stride[i] < 0`
    /// 
    /// </param>
    /// <param name="end_mask">
    /// 
    /// analogous to `begin_mask`
    /// 
    /// </param>
    /// <param name="ellipsis_mask">
    /// 
    /// a bitmask where bit `i` being 1 means the `i`th
    /// position is actually an ellipsis. One bit at most can be 1.
    /// If `ellipsis_mask == 0`, then an implicit ellipsis mask of `1 << (m+1)`
    /// is provided. This means that `foo[3:5] == foo[3:5, ...]`. An ellipsis
    /// implicitly creates as many range specifications as necessary to fully
    /// specify the sliced range for every dimension. For example for a 4-dimensional
    /// tensor `foo` the slice `foo[2, ..., 5:8]` implies `foo[2, :, :, 5:8]`.
    /// 
    /// </param>
    /// <param name="new_axis_mask">
    /// 
    /// a bitmask where bit `i` being 1 means the `i`th
    /// specification creates a new shape 1 dimension. For example
    /// `foo[:4, tf.newaxis, :2]` would produce a shape `(4, 1, 2)` tensor.
    /// 
    /// </param>
    /// <param name="shrink_axis_mask">
    /// 
    /// a bitmask where bit `i` implies that the `i`th
    /// specification should shrink the dimensionality. begin and end
    /// must imply a slice of size 1 in the dimension. For example in
    /// python one might do `foo[:, 3, :]` which would result in
    /// `shrink_axis_mask` being 2.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor strided_slice(Tensor input, Tensor begin, Tensor end, Tensor strides, int begin_mask = 0, int end_mask = 0, int ellipsis_mask = 0, int new_axis_mask = 0, int shrink_axis_mask = 0, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "StridedSlice", name) { args = new object[] { input, begin, end, strides }, attrs = new Dictionary<string, object>() { ["begin_mask"] = begin_mask, ["end_mask"] = end_mask, ["ellipsis_mask"] = ellipsis_mask, ["new_axis_mask"] = new_axis_mask, ["shrink_axis_mask"] = shrink_axis_mask } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return strided_slice_eager_fallback(input, begin, end, strides, begin_mask: begin_mask, end_mask: end_mask, ellipsis_mask: ellipsis_mask, new_axis_mask: new_axis_mask, shrink_axis_mask: shrink_axis_mask, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["begin"] = begin;
        keywords["end"] = end;
        keywords["strides"] = strides;
        keywords["begin_mask"] = begin_mask;
        keywords["end_mask"] = end_mask;
        keywords["ellipsis_mask"] = ellipsis_mask;
        keywords["new_axis_mask"] = new_axis_mask;
        keywords["shrink_axis_mask"] = shrink_axis_mask;
        var _op = tf.OpDefLib._apply_op_helper("StridedSlice", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Index", _op._get_attr_type("Index"), "begin_mask", _op._get_attr_int("begin_mask"), "end_mask", _op._get_attr_int("end_mask"), "ellipsis_mask", _op._get_attr_int("ellipsis_mask"), "new_axis_mask", _op._get_attr_int("new_axis_mask"), "shrink_axis_mask", _op._get_attr_int("shrink_axis_mask") };
            _execute.record_gradient("StridedSlice", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor strided_slice_eager_fallback(Tensor input, Tensor begin, Tensor end, Tensor strides, int begin_mask, int end_mask, int ellipsis_mask, int new_axis_mask, int shrink_axis_mask, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, begin, end, strides };
        object[] _attrs = new object[] { "T", input.dtype, "Index", begin.dtype, "begin_mask", begin_mask, "end_mask", end_mask, "ellipsis_mask", ellipsis_mask, "new_axis_mask", new_axis_mask, "shrink_axis_mask", shrink_axis_mask };
        var _result = _execute.execute("StridedSlice", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("StridedSlice", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Assign `value` to the sliced l-value reference of `ref`.
    /// </summary>
    /// <remarks>
    /// 
    /// The values of `value` are assigned to the positions in the variable
    /// `ref` that are selected by the slice parameters. The slice parameters
    /// `begin`, `end`, `strides`, etc. work exactly as in `StridedSlice`.
    /// 
    /// NOTE this op currently does not support broadcasting and so `value`'s
    /// shape must be exactly the shape produced by the slice of `ref`.
    /// 
    /// </remarks>
    /// <param name="ref_"></param>
    /// <param name="begin"></param>
    /// <param name="end"></param>
    /// <param name="strides"></param>
    /// <param name="value"></param>
    /// <param name="begin_mask"></param>
    /// <param name="end_mask"></param>
    /// <param name="ellipsis_mask"></param>
    /// <param name="new_axis_mask"></param>
    /// <param name="shrink_axis_mask"></param>
    /// <returns></returns>
    public static Tensor strided_slice_assign(Tensor ref_, Tensor begin, Tensor end, Tensor strides, Tensor value, int begin_mask = 0, int end_mask = 0, int ellipsis_mask = 0, int new_axis_mask = 0, int shrink_axis_mask = 0, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            throw new RuntimeError("strided_slice_assign op does not support eager execution. Arg ref is a ref.");
        }
        Dictionary<string, object> keywords = new();
        keywords["ref"] = ref_;
        keywords["begin"] = begin;
        keywords["end"] = end;
        keywords["strides"] = strides;
        keywords["value"] = value;
        keywords["begin_mask"] = begin_mask;
        keywords["end_mask"] = end_mask;
        keywords["ellipsis_mask"] = ellipsis_mask;
        keywords["new_axis_mask"] = new_axis_mask;
        keywords["shrink_axis_mask"] = shrink_axis_mask;
        var _op = tf.OpDefLib._apply_op_helper("StridedSliceAssign", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Index", _op._get_attr_type("Index"), "begin_mask", _op._get_attr_int("begin_mask"), "end_mask", _op._get_attr_int("end_mask"), "ellipsis_mask", _op._get_attr_int("ellipsis_mask"), "new_axis_mask", _op._get_attr_int("new_axis_mask"), "shrink_axis_mask", _op._get_attr_int("shrink_axis_mask") };
            _execute.record_gradient("StridedSliceAssign", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor strided_slice_assign_eager_fallback(Tensor ref_, Tensor begin, Tensor end, Tensor strides, Tensor value, int begin_mask, int end_mask, int ellipsis_mask, int new_axis_mask, int shrink_axis_mask, string name, Context ctx)
    {
        throw new RuntimeError($"strided_slice_assign op does not support eager execution. Arg 'ref' is a ref.");
    }
    /// <summary>
    /// Returns the gradient of `StridedSlice`.
    /// </summary>
    /// <remarks>
    /// 
    /// Since `StridedSlice` cuts out pieces of its `input` which is size
    /// `shape`, its gradient will have the same shape (which is passed here
    /// as `shape`). The gradient will be zero in any element that the slice
    /// does not select.
    /// 
    /// Arguments are the same as StridedSliceGrad with the exception that
    /// `dy` is the input gradient to be propagated and `shape` is the
    /// shape of `StridedSlice`'s `input`.
    /// 
    /// </remarks>
    /// <param name="shape"></param>
    /// <param name="begin"></param>
    /// <param name="end"></param>
    /// <param name="strides"></param>
    /// <param name="dy"></param>
    /// <param name="begin_mask"></param>
    /// <param name="end_mask"></param>
    /// <param name="ellipsis_mask"></param>
    /// <param name="new_axis_mask"></param>
    /// <param name="shrink_axis_mask"></param>
    /// <returns></returns>
    public static Tensor strided_slice_grad(Tensor shape, Tensor begin, Tensor end, Tensor strides, Tensor dy, int begin_mask = 0, int end_mask = 0, int ellipsis_mask = 0, int new_axis_mask = 0, int shrink_axis_mask = 0, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "StridedSliceGrad", name) { args = new object[] { shape, begin, end, strides, dy }, attrs = new Dictionary<string, object>() { ["begin_mask"] = begin_mask, ["end_mask"] = end_mask, ["ellipsis_mask"] = ellipsis_mask, ["new_axis_mask"] = new_axis_mask, ["shrink_axis_mask"] = shrink_axis_mask } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return strided_slice_grad_eager_fallback(shape, begin, end, strides, dy, begin_mask: begin_mask, end_mask: end_mask, ellipsis_mask: ellipsis_mask, new_axis_mask: new_axis_mask, shrink_axis_mask: shrink_axis_mask, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["shape"] = shape;
        keywords["begin"] = begin;
        keywords["end"] = end;
        keywords["strides"] = strides;
        keywords["dy"] = dy;
        keywords["begin_mask"] = begin_mask;
        keywords["end_mask"] = end_mask;
        keywords["ellipsis_mask"] = ellipsis_mask;
        keywords["new_axis_mask"] = new_axis_mask;
        keywords["shrink_axis_mask"] = shrink_axis_mask;
        var _op = tf.OpDefLib._apply_op_helper("StridedSliceGrad", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Index", _op._get_attr_type("Index"), "begin_mask", _op._get_attr_int("begin_mask"), "end_mask", _op._get_attr_int("end_mask"), "ellipsis_mask", _op._get_attr_int("ellipsis_mask"), "new_axis_mask", _op._get_attr_int("new_axis_mask"), "shrink_axis_mask", _op._get_attr_int("shrink_axis_mask") };
            _execute.record_gradient("StridedSliceGrad", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor strided_slice_grad_eager_fallback(Tensor shape, Tensor begin, Tensor end, Tensor strides, Tensor dy, int begin_mask, int end_mask, int ellipsis_mask, int new_axis_mask, int shrink_axis_mask, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { shape, begin, end, strides, dy };
        object[] _attrs = new object[] { "T", dy.dtype, "Index", shape.dtype, "begin_mask", begin_mask, "end_mask", end_mask, "ellipsis_mask", ellipsis_mask, "new_axis_mask", new_axis_mask, "shrink_axis_mask", shrink_axis_mask };
        var _result = _execute.execute("StridedSliceGrad", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("StridedSliceGrad", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Adds sparse `updates` to an existing tensor according to `indices`.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation creates a new tensor by adding sparse `updates` to the passed
    /// in `tensor`.
    /// This operation is very similar to `tf.compat.v1.scatter_nd_add`, except that the
    /// updates are added onto an existing tensor (as opposed to a variable). If the
    /// memory for the existing tensor cannot be re-used, a copy is made and updated.
    /// 
    /// `indices` is an integer tensor containing indices into a new tensor of shape
    /// `tensor.shape`.  The last dimension of `indices` can be at most the rank of
    /// `tensor.shape`:
    /// 
    /// ```
    /// indices.shape[-1] <= tensor.shape.rank
    /// ```
    /// 
    /// The last dimension of `indices` corresponds to indices into elements
    /// (if `indices.shape[-1] = tensor.shape.rank`) or slices
    /// (if `indices.shape[-1] < tensor.shape.rank`) along dimension
    /// `indices.shape[-1]` of `tensor.shape`.  `updates` is a tensor with shape
    /// 
    /// ```
    /// indices.shape[:-1] + tensor.shape[indices.shape[-1]:]
    /// ```
    /// 
    /// The simplest form of `tensor_scatter_nd_add` is to add individual elements to a
    /// tensor by index. For example, say we want to add 4 elements in a rank-1
    /// tensor with 8 elements.
    /// 
    /// In Python, this scatter add operation would look like this:
    /// 
    /// >>> indices = tf.constant([[4], [3], [1], [7]])
    /// >>> updates = tf.constant([9, 10, 11, 12])
    /// >>> tensor = tf.ones([8], dtype=tf.int32)
    /// >>> updated = tf.tensor_scatter_nd_add(tensor, indices, updates)
    /// >>> updated
    /// <tf.Tensor: shape=(8,), dtype=int32,
    /// numpy=array([ 1, 12,  1, 11, 10,  1,  1, 13], dtype=int32)>
    /// 
    /// We can also, insert entire slices of a higher rank tensor all at once. For
    /// example, if we wanted to insert two slices in the first dimension of a
    /// rank-3 tensor with two matrices of new values.
    /// 
    /// In Python, this scatter add operation would look like this:
    /// 
    /// >>> indices = tf.constant([[0], [2]])
    /// >>> updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
    /// ...                         [7, 7, 7, 7], [8, 8, 8, 8]],
    /// ...                        [[5, 5, 5, 5], [6, 6, 6, 6],
    /// ...                         [7, 7, 7, 7], [8, 8, 8, 8]]])
    /// >>> tensor = tf.ones([4, 4, 4],dtype=tf.int32)
    /// >>> updated = tf.tensor_scatter_nd_add(tensor, indices, updates)
    /// >>> updated
    /// <tf.Tensor: shape=(4, 4, 4), dtype=int32,
    /// numpy=array([[[6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9]],
    ///              [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
    ///              [[6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9]],
    ///              [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]], dtype=int32)>
    /// 
    /// Note: on CPU, if an out of bound index is found, an error is returned.
    /// On GPU, if an out of bound index is found, the index is ignored.
    /// 
    /// </remarks>
    /// <param name="tensor"></param>
    /// <param name="indices"></param>
    /// <param name="updates"></param>
    /// <returns></returns>
    public static Tensor tensor_scatter_add(Tensor tensor, Tensor indices, Tensor updates, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TensorScatterAdd", name) { args = new object[] { tensor, indices, updates }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return tensor_scatter_add_eager_fallback(tensor, indices, updates, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["tensor"] = tensor;
        keywords["indices"] = indices;
        keywords["updates"] = updates;
        var _op = tf.OpDefLib._apply_op_helper("TensorScatterAdd", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tindices", _op._get_attr_type("Tindices") };
            _execute.record_gradient("TensorScatterAdd", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor tensor_scatter_add_eager_fallback(Tensor tensor, Tensor indices, Tensor updates, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { tensor, indices, updates };
        object[] _attrs = new object[] { "T", tensor.dtype, "Tindices", indices.dtype };
        var _result = _execute.execute("TensorScatterAdd", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TensorScatterAdd", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Apply a sparse update to a tensor taking the element-wise maximum.
    /// </summary>
    /// <remarks>
    /// 
    /// Returns a new tensor copied from `tensor` whose values are element-wise maximum between
    /// tensor and updates according to the indices.
    /// 
    /// >>> tensor = [0, 0, 0, 0, 0, 0, 0, 0] 
    /// >>> indices = [[1], [4], [5]]
    /// >>> updates = [1, -1, 1]
    /// >>> tf.tensor_scatter_nd_max(tensor, indices, updates).numpy()
    /// array([0, 1, 0, 0, 0, 1, 0, 0], dtype=int32)
    /// 
    /// Refer to `tf.tensor_scatter_nd_update` for more details.
    /// 
    /// </remarks>
    /// <param name="tensor"></param>
    /// <param name="indices"></param>
    /// <param name="updates"></param>
    /// <returns></returns>
    public static Tensor tensor_scatter_max(Tensor tensor, Tensor indices, Tensor updates, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TensorScatterMax", name) { args = new object[] { tensor, indices, updates }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return tensor_scatter_max_eager_fallback(tensor, indices, updates, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["tensor"] = tensor;
        keywords["indices"] = indices;
        keywords["updates"] = updates;
        var _op = tf.OpDefLib._apply_op_helper("TensorScatterMax", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tindices", _op._get_attr_type("Tindices") };
            _execute.record_gradient("TensorScatterMax", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor tensor_scatter_max_eager_fallback(Tensor tensor, Tensor indices, Tensor updates, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { tensor, indices, updates };
        object[] _attrs = new object[] { "T", tensor.dtype, "Tindices", indices.dtype };
        var _result = _execute.execute("TensorScatterMax", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TensorScatterMax", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="tensor"></param>
    /// <param name="indices"></param>
    /// <param name="updates"></param>
    /// <returns></returns>
    public static Tensor tensor_scatter_min(Tensor tensor, Tensor indices, Tensor updates, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TensorScatterMin", name) { args = new object[] { tensor, indices, updates }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return tensor_scatter_min_eager_fallback(tensor, indices, updates, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["tensor"] = tensor;
        keywords["indices"] = indices;
        keywords["updates"] = updates;
        var _op = tf.OpDefLib._apply_op_helper("TensorScatterMin", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tindices", _op._get_attr_type("Tindices") };
            _execute.record_gradient("TensorScatterMin", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor tensor_scatter_min_eager_fallback(Tensor tensor, Tensor indices, Tensor updates, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { tensor, indices, updates };
        object[] _attrs = new object[] { "T", tensor.dtype, "Tindices", indices.dtype };
        var _result = _execute.execute("TensorScatterMin", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TensorScatterMin", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Subtracts sparse `updates` from an existing tensor according to `indices`.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation creates a new tensor by subtracting sparse `updates` from the
    /// passed in `tensor`.
    /// This operation is very similar to `tf.scatter_nd_sub`, except that the updates
    /// are subtracted from an existing tensor (as opposed to a variable). If the memory
    /// for the existing tensor cannot be re-used, a copy is made and updated.
    /// 
    /// `indices` is an integer tensor containing indices into a new tensor of shape
    /// `shape`.  The last dimension of `indices` can be at most the rank of `shape`:
    /// 
    ///     indices.shape[-1] <= shape.rank
    /// 
    /// The last dimension of `indices` corresponds to indices into elements
    /// (if `indices.shape[-1] = shape.rank`) or slices
    /// (if `indices.shape[-1] < shape.rank`) along dimension `indices.shape[-1]` of
    /// `shape`.  `updates` is a tensor with shape
    /// 
    ///     indices.shape[:-1] + shape[indices.shape[-1]:]
    /// 
    /// The simplest form of tensor_scatter_sub is to subtract individual elements
    /// from a tensor by index. For example, say we want to insert 4 scattered elements
    /// in a rank-1 tensor with 8 elements.
    /// 
    /// In Python, this scatter subtract operation would look like this:
    /// 
    /// ```python
    ///     indices = tf.constant([[4], [3], [1], [7]])
    ///     updates = tf.constant([9, 10, 11, 12])
    ///     tensor = tf.ones([8], dtype=tf.int32)
    ///     updated = tf.tensor_scatter_nd_sub(tensor, indices, updates)
    ///     print(updated)
    /// ```
    /// 
    /// The resulting tensor would look like this:
    /// 
    ///     [1, -10, 1, -9, -8, 1, 1, -11]
    /// 
    /// We can also, insert entire slices of a higher rank tensor all at once. For
    /// example, if we wanted to insert two slices in the first dimension of a
    /// rank-3 tensor with two matrices of new values.
    /// 
    /// In Python, this scatter add operation would look like this:
    /// 
    /// ```python
    ///     indices = tf.constant([[0], [2]])
    ///     updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
    ///                             [7, 7, 7, 7], [8, 8, 8, 8]],
    ///                            [[5, 5, 5, 5], [6, 6, 6, 6],
    ///                             [7, 7, 7, 7], [8, 8, 8, 8]]])
    ///     tensor = tf.ones([4, 4, 4],dtype=tf.int32)
    ///     updated = tf.tensor_scatter_nd_sub(tensor, indices, updates)
    ///     print(updated)
    /// ```
    /// 
    /// The resulting tensor would look like this:
    /// 
    ///     [[[-4, -4, -4, -4], [-5, -5, -5, -5], [-6, -6, -6, -6], [-7, -7, -7, -7]],
    ///      [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
    ///      [[-4, -4, -4, -4], [-5, -5, -5, -5], [-6, -6, -6, -6], [-7, -7, -7, -7]],
    ///      [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]]
    /// 
    /// Note that on CPU, if an out of bound index is found, an error is returned.
    /// On GPU, if an out of bound index is found, the index is ignored.
    /// 
    /// </remarks>
    /// <param name="tensor"></param>
    /// <param name="indices"></param>
    /// <param name="updates"></param>
    /// <returns></returns>
    public static Tensor tensor_scatter_sub(Tensor tensor, Tensor indices, Tensor updates, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TensorScatterSub", name) { args = new object[] { tensor, indices, updates }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return tensor_scatter_sub_eager_fallback(tensor, indices, updates, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["tensor"] = tensor;
        keywords["indices"] = indices;
        keywords["updates"] = updates;
        var _op = tf.OpDefLib._apply_op_helper("TensorScatterSub", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tindices", _op._get_attr_type("Tindices") };
            _execute.record_gradient("TensorScatterSub", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor tensor_scatter_sub_eager_fallback(Tensor tensor, Tensor indices, Tensor updates, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { tensor, indices, updates };
        object[] _attrs = new object[] { "T", tensor.dtype, "Tindices", indices.dtype };
        var _result = _execute.execute("TensorScatterSub", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TensorScatterSub", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Scatter `updates` into an existing tensor according to `indices`.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation creates a new tensor by applying sparse `updates` to the passed
    /// in `tensor`.
    /// This operation is very similar to `tf.scatter_nd`, except that the updates are
    /// scattered onto an existing tensor (as opposed to a zero-tensor). If the memory
    /// for the existing tensor cannot be re-used, a copy is made and updated.
    /// 
    /// If `indices` contains duplicates, then we pick the last update for the index.
    /// 
    /// If an out of bound index is found on CPU, an error is returned.
    /// 
    /// **WARNING**: There are some GPU specific semantics for this operation.
    /// - If an out of bound index is found, the index is ignored.
    /// - The order in which updates are applied is nondeterministic, so the output
    /// will be nondeterministic if `indices` contains duplicates.
    /// 
    /// `indices` is an integer tensor containing indices into a new tensor of shape
    /// `shape`.
    /// 
    /// * `indices` must have at least 2 axes: `(num_updates, index_depth)`.
    /// * The last axis of `indices` is how deep to index into `tensor` so  this index
    ///   depth must be less than the rank of `tensor`: `indices.shape[-1] <= tensor.ndim`
    /// 
    /// if `indices.shape[-1] = tensor.rank` this Op indexes and updates scalar elements.
    /// if `indices.shape[-1] < tensor.rank` it indexes and updates slices of the input
    /// `tensor`.
    /// 
    /// Each `update` has a rank of `tensor.rank - indices.shape[-1]`.
    /// The overall shape of `updates` is:
    /// 
    /// ```
    /// indices.shape[:-1] + tensor.shape[indices.shape[-1]:]
    /// ```
    /// 
    /// For usage examples see the python [tf.tensor_scatter_nd_update](
    /// https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_update) function
    /// 
    /// 
    /// </remarks>
    /// <param name="tensor"></param>
    /// <param name="indices"></param>
    /// <param name="updates"></param>
    /// <returns></returns>
    public static Tensor tensor_scatter_update(Tensor tensor, Tensor indices, Tensor updates, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TensorScatterUpdate", name) { args = new object[] { tensor, indices, updates }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return tensor_scatter_update_eager_fallback(tensor, indices, updates, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["tensor"] = tensor;
        keywords["indices"] = indices;
        keywords["updates"] = updates;
        var _op = tf.OpDefLib._apply_op_helper("TensorScatterUpdate", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tindices", _op._get_attr_type("Tindices") };
            _execute.record_gradient("TensorScatterUpdate", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor tensor_scatter_update_eager_fallback(Tensor tensor, Tensor indices, Tensor updates, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { tensor, indices, updates };
        object[] _attrs = new object[] { "T", tensor.dtype, "Tindices", indices.dtype };
        var _result = _execute.execute("TensorScatterUpdate", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TensorScatterUpdate", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Assign `value` to the sliced l-value reference of `input`.
    /// </summary>
    /// <remarks>
    /// 
    /// The values of `value` are assigned to the positions in the tensor `input` that
    /// are selected by the slice parameters. The slice parameters `begin` `end`
    /// `strides` etc. work exactly as in `StridedSlice`.
    /// 
    /// NOTE this op currently does not support broadcasting and so `value`'s shape
    /// must be exactly the shape produced by the slice of `input`.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="begin"></param>
    /// <param name="end"></param>
    /// <param name="strides"></param>
    /// <param name="value"></param>
    /// <param name="begin_mask"></param>
    /// <param name="end_mask"></param>
    /// <param name="ellipsis_mask"></param>
    /// <param name="new_axis_mask"></param>
    /// <param name="shrink_axis_mask"></param>
    /// <returns></returns>
    public static Tensor tensor_strided_slice_update(Tensor input, Tensor begin, Tensor end, Tensor strides, Tensor value, int begin_mask = 0, int end_mask = 0, int ellipsis_mask = 0, int new_axis_mask = 0, int shrink_axis_mask = 0, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TensorStridedSliceUpdate", name) { args = new object[] { input, begin, end, strides, value }, attrs = new Dictionary<string, object>() { ["begin_mask"] = begin_mask, ["end_mask"] = end_mask, ["ellipsis_mask"] = ellipsis_mask, ["new_axis_mask"] = new_axis_mask, ["shrink_axis_mask"] = shrink_axis_mask } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return tensor_strided_slice_update_eager_fallback(input, begin, end, strides, value, begin_mask: begin_mask, end_mask: end_mask, ellipsis_mask: ellipsis_mask, new_axis_mask: new_axis_mask, shrink_axis_mask: shrink_axis_mask, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["begin"] = begin;
        keywords["end"] = end;
        keywords["strides"] = strides;
        keywords["value"] = value;
        keywords["begin_mask"] = begin_mask;
        keywords["end_mask"] = end_mask;
        keywords["ellipsis_mask"] = ellipsis_mask;
        keywords["new_axis_mask"] = new_axis_mask;
        keywords["shrink_axis_mask"] = shrink_axis_mask;
        var _op = tf.OpDefLib._apply_op_helper("TensorStridedSliceUpdate", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Index", _op._get_attr_type("Index"), "begin_mask", _op._get_attr_int("begin_mask"), "end_mask", _op._get_attr_int("end_mask"), "ellipsis_mask", _op._get_attr_int("ellipsis_mask"), "new_axis_mask", _op._get_attr_int("new_axis_mask"), "shrink_axis_mask", _op._get_attr_int("shrink_axis_mask") };
            _execute.record_gradient("TensorStridedSliceUpdate", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor tensor_strided_slice_update_eager_fallback(Tensor input, Tensor begin, Tensor end, Tensor strides, Tensor value, int begin_mask, int end_mask, int ellipsis_mask, int new_axis_mask, int shrink_axis_mask, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, begin, end, strides, value };
        object[] _attrs = new object[] { "T", input.dtype, "Index", begin.dtype, "begin_mask", begin_mask, "end_mask", end_mask, "ellipsis_mask", ellipsis_mask, "new_axis_mask", new_axis_mask, "shrink_axis_mask", shrink_axis_mask };
        var _result = _execute.execute("TensorStridedSliceUpdate", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TensorStridedSliceUpdate", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Constructs a tensor by tiling a given tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation creates a new tensor by replicating `input` `multiples` times.
    /// The output tensor's i'th dimension has `input.dims(i) * multiples[i]` elements,
    /// and the values of `input` are replicated `multiples[i]` times along the 'i'th
    /// dimension. For example, tiling `[a b c d]` by `[2]` produces
    /// `[a b c d a b c d]`.
    /// 
    /// >>> a = tf.constant([[1,2,3],[4,5,6]], tf.int32)
    /// >>> b = tf.constant([1,2], tf.int32)
    /// >>> tf.tile(a, b)
    /// <tf.Tensor: shape=(2, 6), dtype=int32, numpy=
    /// array([[1, 2, 3, 1, 2, 3],
    ///        [4, 5, 6, 4, 5, 6]], dtype=int32)>
    /// >>> c = tf.constant([2,1], tf.int32)
    /// >>> tf.tile(a, c)
    /// <tf.Tensor: shape=(4, 3), dtype=int32, numpy=
    /// array([[1, 2, 3],
    ///        [4, 5, 6],
    ///        [1, 2, 3],
    ///        [4, 5, 6]], dtype=int32)>
    /// >>> d = tf.constant([2,2], tf.int32)
    /// >>> tf.tile(a, d)
    /// <tf.Tensor: shape=(4, 6), dtype=int32, numpy=
    /// array([[1, 2, 3, 1, 2, 3],
    ///        [4, 5, 6, 4, 5, 6],
    ///        [1, 2, 3, 1, 2, 3],
    ///        [4, 5, 6, 4, 5, 6]], dtype=int32)>
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="multiples"></param>
    /// <returns></returns>
    public static Tensor tile(Tensor input, Tensor multiples, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Tile", name) { args = new object[] { input, multiples }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return tile_eager_fallback(input, multiples, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["multiples"] = multiples;
        var _op = tf.OpDefLib._apply_op_helper("Tile", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tmultiples", _op._get_attr_type("Tmultiples") };
            _execute.record_gradient("Tile", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor tile_eager_fallback(Tensor input, Tensor multiples, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, multiples };
        object[] _attrs = new object[] { "T", input.dtype, "Tmultiples", multiples.dtype };
        var _result = _execute.execute("Tile", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Tile", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns the gradient of `Tile`.
    /// </summary>
    /// <remarks>
    /// 
    /// Since `Tile` takes an input and repeats the input `multiples` times
    /// along each dimension, `TileGrad` takes in `multiples` and aggregates
    /// each repeated tile of `input` into `output`.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="multiples"></param>
    /// <returns></returns>
    public static Tensor tile_grad(Tensor input, Tensor multiples, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TileGrad", name) { args = new object[] { input, multiples }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return tile_grad_eager_fallback(input, multiples, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["multiples"] = multiples;
        var _op = tf.OpDefLib._apply_op_helper("TileGrad", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("TileGrad", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor tile_grad_eager_fallback(Tensor input, Tensor multiples, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, multiples };
        object[] _attrs = new object[] { "T", input.dtype };
        var _result = _execute.execute("TileGrad", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TileGrad", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Shuffle dimensions of x according to a permutation.
    /// </summary>
    /// <remarks>
    /// 
    /// The output `y` has the same rank as `x`. The shapes of `x` and `y` satisfy:
    ///   `y.shape[i] == x.shape[perm[i]] for i in [0, 1, ..., rank(x) - 1]`
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="perm"></param>
    /// <returns></returns>
    public static Tensor transpose(Tensor x, Tensor perm, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Transpose", name) { args = new object[] { x, perm }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return transpose_eager_fallback(x, perm, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["perm"] = perm;
        var _op = tf.OpDefLib._apply_op_helper("Transpose", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tperm", _op._get_attr_type("Tperm") };
            _execute.record_gradient("Transpose", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor transpose_eager_fallback(Tensor x, Tensor perm, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, perm };
        object[] _attrs = new object[] { "T", x.dtype, "Tperm", perm.dtype };
        var _result = _execute.execute("Transpose", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Transpose", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Finds unique elements in a 1-D tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation returns a tensor `y` containing all of the unique elements of `x`
    /// sorted in the same order that they occur in `x`; `x` does not need to be sorted.
    /// This operation also returns a tensor `idx` the same size as `x` that contains
    /// the index of each value of `x` in the unique output `y`. In other words:
    /// 
    /// `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`
    /// 
    /// Examples:
    /// 
    /// ```
    /// # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
    /// y, idx = unique(x)
    /// y ==> [1, 2, 4, 7, 8]
    /// idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
    /// ```
    /// 
    /// ```
    /// # tensor 'x' is [4, 5, 1, 2, 3, 3, 4, 5]
    /// y, idx = unique(x)
    /// y ==> [4, 5, 1, 2, 3]
    /// idx ==> [0, 1, 2, 3, 4, 4, 0, 1]
    /// ```
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="out_idx"></param>
    /// <returns></returns>
    public static Tensor[] unique(Tensor x, TF_DataType out_idx = TF_DataType.TF_INT32, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Unique", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { ["out_idx"] = out_idx } });
                return _fast_path_result;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return unique_eager_fallback(x, out_idx: out_idx, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["out_idx"] = out_idx;
        var _op = tf.OpDefLib._apply_op_helper("Unique", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "out_idx", _op._get_attr_type("out_idx") };
            _execute.record_gradient("Unique", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] unique_eager_fallback(Tensor x, TF_DataType out_idx, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype, "out_idx", out_idx };
        var _result = _execute.execute("Unique", 2, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Unique", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Finds unique elements along an axis of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation either returns a tensor `y` containing unique elements
    /// along the `axis` of a tensor. The returned unique elements is sorted
    /// in the same order as they occur along `axis` in `x`.
    /// This operation also returns a tensor `idx` that is the same size as
    /// the number of the elements in `x` along the `axis` dimension. It
    /// contains the index in the unique output `y`.
    /// In other words, for an `1-D` tensor `x` with `axis = None:
    /// 
    /// `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`
    /// 
    /// For example:
    /// 
    /// ```
    /// # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
    /// y, idx = unique(x)
    /// y ==> [1, 2, 4, 7, 8]
    /// idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
    /// ```
    /// 
    /// For an `2-D` tensor `x` with `axis = 0`:
    /// 
    /// ```
    /// # tensor 'x' is [[1, 0, 0],
    /// #                [1, 0, 0],
    /// #                [2, 0, 0]]
    /// y, idx = unique(x, axis=0)
    /// y ==> [[1, 0, 0],
    ///        [2, 0, 0]]
    /// idx ==> [0, 0, 1]
    /// ```
    /// 
    /// For an `2-D` tensor `x` with `axis = 1`:
    /// 
    /// ```
    /// # tensor 'x' is [[1, 0, 0],
    /// #                [1, 0, 0],
    /// #                [2, 0, 0]]
    /// y, idx = unique(x, axis=1)
    /// y ==> [[1, 0],
    ///        [1, 0],
    ///        [2, 0]]
    /// idx ==> [0, 1, 1]
    /// ```
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="axis"></param>
    /// <param name="out_idx"></param>
    /// <returns></returns>
    public static Tensor[] unique_v2(Tensor x, Tensor axis, TF_DataType out_idx = TF_DataType.TF_INT32, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "UniqueV2", name) { args = new object[] { x, axis }, attrs = new Dictionary<string, object>() { ["out_idx"] = out_idx } });
                return _fast_path_result;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return unique_v2_eager_fallback(x, axis, out_idx: out_idx, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["axis"] = axis;
        keywords["out_idx"] = out_idx;
        var _op = tf.OpDefLib._apply_op_helper("UniqueV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Taxis", _op._get_attr_type("Taxis"), "out_idx", _op._get_attr_type("out_idx") };
            _execute.record_gradient("UniqueV2", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] unique_v2_eager_fallback(Tensor x, Tensor axis, TF_DataType out_idx, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, axis };
        object[] _attrs = new object[] { "T", x.dtype, "Taxis", axis.dtype, "out_idx", out_idx };
        var _result = _execute.execute("UniqueV2", 2, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("UniqueV2", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Finds unique elements in a 1-D tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation returns a tensor `y` containing all of the unique elements of `x`
    /// sorted in the same order that they occur in `x`. This operation also returns a
    /// tensor `idx` the same size as `x` that contains the index of each value of `x`
    /// in the unique output `y`. Finally, it returns a third tensor `count` that
    /// contains the count of each element of `y` in `x`. In other words:
    /// 
    /// `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`
    /// 
    /// For example:
    /// 
    /// ```
    /// # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
    /// y, idx, count = unique_with_counts(x)
    /// y ==> [1, 2, 4, 7, 8]
    /// idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
    /// count ==> [2, 1, 3, 1, 2]
    /// ```
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="out_idx"></param>
    /// <returns></returns>
    public static Tensor[] unique_with_counts(Tensor x, TF_DataType out_idx = TF_DataType.TF_INT32, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "UniqueWithCounts", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { ["out_idx"] = out_idx } });
                return _fast_path_result;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return unique_with_counts_eager_fallback(x, out_idx: out_idx, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["out_idx"] = out_idx;
        var _op = tf.OpDefLib._apply_op_helper("UniqueWithCounts", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "out_idx", _op._get_attr_type("out_idx") };
            _execute.record_gradient("UniqueWithCounts", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] unique_with_counts_eager_fallback(Tensor x, TF_DataType out_idx, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype, "out_idx", out_idx };
        var _result = _execute.execute("UniqueWithCounts", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("UniqueWithCounts", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Finds unique elements along an axis of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation either returns a tensor `y` containing unique elements
    /// along the `axis` of a tensor. The returned unique elements is sorted
    /// in the same order as they occur along `axis` in `x`.
    /// This operation also returns a tensor `idx` and a tensor `count`
    /// that are the same size as the number of the elements in `x` along the
    /// `axis` dimension. The `idx` contains the index in the unique output `y`
    /// and the `count` contains the count in the unique output `y`.
    /// In other words, for an `1-D` tensor `x` with `axis = None:
    /// 
    /// `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`
    /// 
    /// For example:
    /// 
    /// ```
    /// x = tf.constant([1, 1, 2, 4, 4, 4, 7, 8, 8])
    /// y, idx, count = UniqueWithCountsV2(x, axis = [0])
    /// y ==> [1, 2, 4, 7, 8]
    /// idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
    /// count ==> [2, 1, 3, 1, 2]
    /// ```
    /// 
    /// For a `2-D` tensor `x` with `axis = 0`:
    /// 
    /// ```
    /// x = tf.constant([[1, 0, 0],
    ///                 [1, 0, 0],
    ///                 [2, 0, 0]])
    /// y, idx, count = UniqueWithCountsV2(x, axis=[0])
    /// y ==> [[1, 0, 0],
    ///        [2, 0, 0]]
    /// idx ==> [0, 0, 1]
    /// count ==> [2, 1]
    /// ```
    /// 
    /// For a `2-D` tensor `x` with `axis = 1`:
    /// 
    /// ```
    /// x = tf.constant([[1, 0, 0],
    ///                 [1, 0, 0],
    ///                 [2, 0, 0]])
    /// y, idx, count = UniqueWithCountsV2(x, axis=[1])
    /// y ==> [[1, 0],
    ///        [1, 0],
    ///        [2, 0]]
    /// idx ==> [0, 1, 1]
    /// count ==> [1, 2]
    /// ```
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="axis"></param>
    /// <param name="out_idx"></param>
    /// <returns></returns>
    public static Tensor[] unique_with_counts_v2(Tensor x, Tensor axis, TF_DataType out_idx = TF_DataType.TF_INT32, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "UniqueWithCountsV2", name) { args = new object[] { x, axis }, attrs = new Dictionary<string, object>() { ["out_idx"] = out_idx } });
                return _fast_path_result;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return unique_with_counts_v2_eager_fallback(x, axis, out_idx: out_idx, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["axis"] = axis;
        keywords["out_idx"] = out_idx;
        var _op = tf.OpDefLib._apply_op_helper("UniqueWithCountsV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Taxis", _op._get_attr_type("Taxis"), "out_idx", _op._get_attr_type("out_idx") };
            _execute.record_gradient("UniqueWithCountsV2", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] unique_with_counts_v2_eager_fallback(Tensor x, Tensor axis, TF_DataType out_idx, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, axis };
        object[] _attrs = new object[] { "T", x.dtype, "Taxis", axis.dtype, "out_idx", out_idx };
        var _result = _execute.execute("UniqueWithCountsV2", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("UniqueWithCountsV2", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Unpacks a given dimension of a rank-`R` tensor into `num` rank-`(R-1)` tensors.
    /// </summary>
    /// <remarks>
    /// 
    /// Unpacks `num` tensors from `value` by chipping it along the `axis` dimension.
    /// For example, given a tensor of shape `(A, B, C, D)`;
    /// 
    /// If `axis == 0` then the i'th tensor in `output` is the slice `value[i, :, :, :]`
    ///   and each tensor in `output` will have shape `(B, C, D)`. (Note that the
    ///   dimension unpacked along is gone, unlike `split`).
    /// 
    /// If `axis == 1` then the i'th tensor in `output` is the slice `value[:, i, :, :]`
    ///   and each tensor in `output` will have shape `(A, C, D)`.
    /// Etc.
    /// 
    /// This is the opposite of `pack`.
    /// 
    /// </remarks>
    /// <param name="value"></param>
    /// <param name="num"></param>
    /// <param name="axis">
    /// 
    /// Dimension along which to unpack.  Negative values wrap around, so the
    /// valid range is `[-R, R)`.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] unpack(Tensor value, int num = 0, int axis = 0, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Unpack", name) { args = new object[] { value }, attrs = new Dictionary<string, object>() { ["num"] = num, ["axis"] = axis } });
                return _fast_path_result;
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return unpack_eager_fallback(value, num: num, axis: axis, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["value"] = value;
        keywords["num"] = num;
        keywords["axis"] = axis;
        var _op = tf.OpDefLib._apply_op_helper("Unpack", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "num", _op._get_attr_int("num"), "T", _op._get_attr_type("T"), "axis", _op._get_attr_int("axis") };
            _execute.record_gradient("Unpack", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] unpack_eager_fallback(Tensor value, int num, int axis, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { value };
        object[] _attrs = new object[] { "num", num, "T", value.dtype, "axis", axis };
        var _result = _execute.execute("Unpack", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Unpack", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Converts an array of flat indices into a tuple of coordinate arrays.
    /// </summary>
    /// <remarks>
    /// 
    /// 
    /// Example:
    /// 
    /// ```
    /// y = tf.unravel_index(indices=[2, 5, 7], dims=[3, 3])
    /// # 'dims' represent a hypothetical (3, 3) tensor of indices:
    /// # [[0, 1, *2*],
    /// #  [3, 4, *5*],
    /// #  [6, *7*, 8]]
    /// # For each entry from 'indices', this operation returns
    /// # its coordinates (marked with '*'), such as
    /// # 2 ==> (0, 2)
    /// # 5 ==> (1, 2)
    /// # 7 ==> (2, 1)
    /// y ==> [[0, 1, 2], [2, 2, 1]]
    /// ```
    /// 
    /// @compatibility(numpy)
    /// Equivalent to np.unravel_index
    /// @end_compatibility
    /// 
    /// </remarks>
    /// <param name="indices"></param>
    /// <param name="dims"></param>
    /// <returns></returns>
    public static Tensor unravel_index(Tensor indices, Tensor dims, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "UnravelIndex", name) { args = new object[] { indices, dims }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return unravel_index_eager_fallback(indices, dims, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["indices"] = indices;
        keywords["dims"] = dims;
        var _op = tf.OpDefLib._apply_op_helper("UnravelIndex", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tidx", _op._get_attr_type("Tidx") };
            _execute.record_gradient("UnravelIndex", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor unravel_index_eager_fallback(Tensor indices, Tensor dims, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { indices, dims };
        object[] _attrs = new object[] { "Tidx", indices.dtype };
        var _result = _execute.execute("UnravelIndex", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("UnravelIndex", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Applies upper_bound(sorted_search_values, values) along each row.
    /// </summary>
    /// <remarks>
    /// 
    /// Each set of rows with the same index in (sorted_inputs, values) is treated
    /// independently.  The resulting row is the equivalent of calling
    /// `np.searchsorted(sorted_inputs, values, side='right')`.
    /// 
    /// The result is not a global index to the entire
    /// `Tensor`, but rather just the index in the last dimension.
    /// 
    /// A 2-D example:
    ///   sorted_sequence = [[0, 3, 9, 9, 10],
    ///                      [1, 2, 3, 4, 5]]
    ///   values = [[2, 4, 9],
    ///             [0, 2, 6]]
    /// 
    ///   result = UpperBound(sorted_sequence, values)
    /// 
    ///   result == [[1, 2, 4],
    ///              [0, 2, 5]]
    /// 
    /// </remarks>
    /// <param name="sorted_inputs"></param>
    /// <param name="values"></param>
    /// <param name="out_type"></param>
    /// <returns></returns>
    public static Tensor upper_bound(Tensor sorted_inputs, Tensor values, TF_DataType out_type = TF_DataType.TF_INT32, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "UpperBound", name) { args = new object[] { sorted_inputs, values }, attrs = new Dictionary<string, object>() { ["out_type"] = out_type } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return upper_bound_eager_fallback(sorted_inputs, values, out_type: out_type, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["sorted_inputs"] = sorted_inputs;
        keywords["values"] = values;
        keywords["out_type"] = out_type;
        var _op = tf.OpDefLib._apply_op_helper("UpperBound", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "out_type", _op._get_attr_type("out_type") };
            _execute.record_gradient("UpperBound", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor upper_bound_eager_fallback(Tensor sorted_inputs, Tensor values, TF_DataType out_type, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { sorted_inputs, values };
        object[] _attrs = new object[] { "T", sorted_inputs.dtype, "out_type", out_type };
        var _result = _execute.execute("UpperBound", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("UpperBound", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns locations of nonzero / true values in a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation returns the coordinates of true elements in `input`. The
    /// coordinates are returned in a 2-D tensor where the first dimension (rows)
    /// represents the number of true elements, and the second dimension (columns)
    /// represents the coordinates of the true elements. Keep in mind, the shape of
    /// the output tensor can vary depending on how many true values there are in
    /// `input`. Indices are output in row-major order.
    /// 
    /// For example:
    /// 
    /// ```
    /// # 'input' tensor is [[True, False]
    /// #                    [True, False]]
    /// # 'input' has two true values, so output has two coordinates.
    /// # 'input' has rank of 2, so coordinates have two indices.
    /// where(input) ==> [[0, 0],
    ///                   [1, 0]]
    /// 
    /// # `input` tensor is [[[True, False]
    /// #                     [True, False]]
    /// #                    [[False, True]
    /// #                     [False, True]]
    /// #                    [[False, False]
    /// #                     [False, True]]]
    /// # 'input' has 5 true values, so output has 5 coordinates.
    /// # 'input' has rank of 3, so coordinates have three indices.
    /// where(input) ==> [[0, 0, 0],
    ///                   [0, 1, 0],
    ///                   [1, 0, 1],
    ///                   [1, 1, 1],
    ///                   [2, 1, 1]]
    /// 
    /// # `input` tensor is [[[1.5,  0.0]
    /// #                     [-0.5, 0.0]]
    /// #                    [[0.0,  0.25]
    /// #                     [0.0,  0.75]]
    /// #                    [[0.0,  0.0]
    /// #                     [0.0,  0.01]]]
    /// # 'input' has 5 nonzero values, so output has 5 coordinates.
    /// # 'input' has rank of 3, so coordinates have three indices.
    /// where(input) ==> [[0, 0, 0],
    ///                   [0, 1, 0],
    ///                   [1, 0, 1],
    ///                   [1, 1, 1],
    ///                   [2, 1, 1]]
    /// 
    /// # `input` tensor is [[[1.5 + 0.0j, 0.0  + 0.0j]
    /// #                     [0.0 + 0.5j, 0.0  + 0.0j]]
    /// #                    [[0.0 + 0.0j, 0.25 + 1.5j]
    /// #                     [0.0 + 0.0j, 0.75 + 0.0j]]
    /// #                    [[0.0 + 0.0j, 0.0  + 0.0j]
    /// #                     [0.0 + 0.0j, 0.01 + 0.0j]]]
    /// # 'input' has 5 nonzero magnitude values, so output has 5 coordinates.
    /// # 'input' has rank of 3, so coordinates have three indices.
    /// where(input) ==> [[0, 0, 0],
    ///                   [0, 1, 0],
    ///                   [1, 0, 1],
    ///                   [1, 1, 1],
    ///                   [2, 1, 1]]
    /// ```
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <returns></returns>
    public static Tensor where(Tensor input, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Where", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return where_eager_fallback(input, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        var _op = tf.OpDefLib._apply_op_helper("Where", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Where", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor where_eager_fallback(Tensor input, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "T", input.dtype };
        var _result = _execute.execute("Where", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Where", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns a tensor of zeros with the same shape and type as x.
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor zeros_like(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ZerosLike", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return zeros_like_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("ZerosLike", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("ZerosLike", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor zeros_like_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("ZerosLike", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ZerosLike", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
}
