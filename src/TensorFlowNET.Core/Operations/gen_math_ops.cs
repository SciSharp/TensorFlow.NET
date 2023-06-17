/*Wrappers around TensorFlow ops. This file is MACHINE GENERATED! Do not edit.*/

using Tensorflow.Eager;
using Tensorflow.Contexts;
using Tensorflow.Exceptions;
using static Tensorflow.Binding;

namespace Tensorflow;

public static class gen_math_ops
{
    /// <summary>
    /// Computes the absolute value of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Given a tensor `x`, this operation returns a tensor containing the absolute
    /// value of each element in `x`. For example, if x is an input element and y is
    /// an output element, this operation computes \(y = |x|\).
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor abs(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Abs", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return abs_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Abs", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Abs", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor abs_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Abs", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Abs", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns the element-wise sum of a list of tensors.
    /// </summary>
    /// <remarks>
    /// 
    /// `tf.accumulate_n_v2` performs the same operation as `tf.add_n`, but does not
    /// wait for all of its inputs to be ready before beginning to sum. This can
    /// save memory if inputs are ready at different times, since minimum temporary
    /// storage is proportional to the output size rather than the inputs size.
    /// 
    /// Unlike the original `accumulate_n`, `accumulate_n_v2` is differentiable.
    /// 
    /// Returns a `Tensor` of same shape and type as the elements of `inputs`.
    /// 
    /// </remarks>
    /// <param name="inputs"></param>
    /// <param name="shape">
    /// 
    /// Shape of elements of `inputs`.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor accumulate_nv2(Tensors inputs, Shape shape, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "AccumulateNV2", name) { args = new object[] { inputs }, attrs = new Dictionary<string, object>() { ["shape"] = shape } });
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
                return accumulate_nv2_eager_fallback(inputs, shape: shape, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["inputs"] = inputs;
        keywords["shape"] = shape;
        var _op = tf.OpDefLib._apply_op_helper("AccumulateNV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "N", _op._get_attr_int("N"), "T", _op._get_attr_type("T"), "shape", _op.get_attr("shape") };
            _execute.record_gradient("AccumulateNV2", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor accumulate_nv2_eager_fallback(Tensors inputs, Shape shape, string name, Context ctx)
    {
        List<Tensor> _inputs_flat_list = new();
        _inputs_flat_list.AddRange(inputs);
        var _inputs_flat = _inputs_flat_list.ToArray();
        object[] _attrs = new object[] { "N", inputs.Length, "T", inputs.dtype, "shape", shape };
        var _result = _execute.execute("AccumulateNV2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("AccumulateNV2", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes acos of x element-wise.
    /// </summary>
    /// <remarks>
    /// 
    /// 
    ///   Provided an input tensor, the `tf.math.acos` operation returns the inverse cosine of each element of the tensor. If `y = tf.math.cos(x)` then, `x = tf.math.acos(y)`.
    /// 
    ///   Input range is `[-1, 1]` and the output has a range of `[0, pi]`.
    /// 
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor acos(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Acos", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return acos_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Acos", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Acos", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor acos_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Acos", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Acos", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes inverse hyperbolic cosine of x element-wise.
    /// </summary>
    /// <remarks>
    /// 
    /// Given an input tensor, the function computes inverse hyperbolic cosine of every element.
    /// Input range is `[1, inf]`. It returns `nan` if the input lies outside the range.
    /// 
    /// ```python
    /// x = tf.constant([-2, -0.5, 1, 1.2, 200, 10000, float("inf")])
    /// tf.math.acosh(x) ==> [nan nan 0. 0.62236255 5.9914584 9.903487 inf]
    /// ```
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor acosh(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Acosh", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return acosh_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Acosh", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Acosh", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor acosh_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Acosh", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Acosh", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns x + y element-wise.
    /// </summary>
    /// <remarks>
    /// 
    /// *NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
    /// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    /// 
    /// Given two input tensors, the `tf.add` operation computes the sum for every element in the tensor.
    /// 
    /// Both input and output have a range `(-inf, inf)`.
    /// 
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <returns></returns>
    public static Tensor add(Tensor x, Tensor y, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Add", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { } });
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
                return add_eager_fallback(x, y, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        var _op = tf.OpDefLib._apply_op_helper("Add", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Add", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor add_eager_fallback(Tensor x, Tensor y, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Add", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Add", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Add all input tensors element wise.
    /// </summary>
    /// <remarks>
    /// 
    ///   Inputs must be of same size and shape.
    /// 
    ///   ```python
    ///   x = [9, 7, 10]
    ///   tf.math.add_n(x) ==> 26
    ///   ```
    /// 
    /// </remarks>
    /// <param name="inputs"></param>
    /// <returns></returns>
    public static Tensor add_n(Tensors inputs, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "AddN", name) { args = new object[] { inputs }, attrs = new Dictionary<string, object>() { } });
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
                return add_n_eager_fallback(inputs, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["inputs"] = inputs;
        var _op = tf.OpDefLib._apply_op_helper("AddN", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "N", _op._get_attr_int("N"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("AddN", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor add_n_eager_fallback(Tensors inputs, string name, Context ctx)
    {
        List<Tensor> _inputs_flat_list = new();
        _inputs_flat_list.AddRange(inputs);
        var _inputs_flat = _inputs_flat_list.ToArray();
        object[] _attrs = new object[] { "N", inputs.Length, "T", inputs.dtype };
        var _result = _execute.execute("AddN", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("AddN", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns x + y element-wise.
    /// </summary>
    /// <remarks>
    /// 
    /// *NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
    /// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <returns></returns>
    public static Tensor add_v2(Tensor x, Tensor y, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "AddV2", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { } });
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
                return add_v2_eager_fallback(x, y, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        var _op = tf.OpDefLib._apply_op_helper("AddV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("AddV2", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor add_v2_eager_fallback(Tensor x, Tensor y, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("AddV2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("AddV2", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the "logical and" of elements across dimensions of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Reduces `input` along the dimensions given in `reduction_indices`. Unless
    /// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
    /// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
    /// retained with length 1.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="reduction_indices"></param>
    /// <param name="keep_dims">
    /// 
    /// If true, retain reduced dimensions with length 1.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor all(Tensor input, Tensor reduction_indices, bool keep_dims = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "All", name) { args = new object[] { input, reduction_indices }, attrs = new Dictionary<string, object>() { ["keep_dims"] = keep_dims } });
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
                return all_eager_fallback(input, reduction_indices, keep_dims: keep_dims, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["reduction_indices"] = reduction_indices;
        keywords["keep_dims"] = keep_dims;
        var _op = tf.OpDefLib._apply_op_helper("All", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "keep_dims", _op._get_attr_bool("keep_dims"), "Tidx", _op._get_attr_type("Tidx") };
            _execute.record_gradient("All", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor all_eager_fallback(Tensor input, Tensor reduction_indices, bool keep_dims, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, reduction_indices };
        object[] _attrs = new object[] { "keep_dims", keep_dims, "Tidx", reduction_indices.dtype };
        var _result = _execute.execute("All", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("All", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns the argument of a complex number.
    /// </summary>
    /// <remarks>
    /// 
    /// Given a tensor `input` of complex numbers, this operation returns a tensor of
    /// type `float` that is the argument of each element in `input`. All elements in
    /// `input` must be complex numbers of the form \(a + bj\), where *a*
    /// is the real part and *b* is the imaginary part.
    /// 
    /// The argument returned by this operation is of the form \(atan2(b, a)\).
    /// 
    /// For example:
    /// 
    /// ```
    /// # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
    /// tf.angle(input) ==> [2.0132, 1.056]
    /// ```
    /// 
    /// @compatibility(numpy)
    /// Equivalent to np.angle.
    /// @end_compatibility
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="Tout"></param>
    /// <returns></returns>
    public static Tensor angle(Tensor input, TF_DataType Tout = TF_DataType.TF_FLOAT, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Angle", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { ["Tout"] = Tout } });
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
                return angle_eager_fallback(input, Tout: Tout, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["Tout"] = Tout;
        var _op = tf.OpDefLib._apply_op_helper("Angle", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tout", _op._get_attr_type("Tout") };
            _execute.record_gradient("Angle", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor angle_eager_fallback(Tensor input, TF_DataType Tout, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "T", input.dtype, "Tout", Tout };
        var _result = _execute.execute("Angle", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Angle", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the "logical or" of elements across dimensions of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Reduces `input` along the dimensions given in `reduction_indices`. Unless
    /// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
    /// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
    /// retained with length 1.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="reduction_indices"></param>
    /// <param name="keep_dims">
    /// 
    /// If true, retain reduced dimensions with length 1.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor any(Tensor input, Tensor reduction_indices, bool keep_dims = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Any", name) { args = new object[] { input, reduction_indices }, attrs = new Dictionary<string, object>() { ["keep_dims"] = keep_dims } });
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
                return any_eager_fallback(input, reduction_indices, keep_dims: keep_dims, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["reduction_indices"] = reduction_indices;
        keywords["keep_dims"] = keep_dims;
        var _op = tf.OpDefLib._apply_op_helper("Any", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "keep_dims", _op._get_attr_bool("keep_dims"), "Tidx", _op._get_attr_type("Tidx") };
            _execute.record_gradient("Any", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor any_eager_fallback(Tensor input, Tensor reduction_indices, bool keep_dims, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, reduction_indices };
        object[] _attrs = new object[] { "keep_dims", keep_dims, "Tidx", reduction_indices.dtype };
        var _result = _execute.execute("Any", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Any", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns the truth value of abs(x-y) < tolerance element-wise.
    /// </summary>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <param name="tolerance"></param>
    /// <returns></returns>
    public static Tensor approximate_equal(Tensor x, Tensor y, float tolerance = 1E-05f, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ApproximateEqual", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { ["tolerance"] = tolerance } });
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
                return approximate_equal_eager_fallback(x, y, tolerance: tolerance, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        keywords["tolerance"] = tolerance;
        var _op = tf.OpDefLib._apply_op_helper("ApproximateEqual", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "tolerance", _op.get_attr("tolerance") };
            _execute.record_gradient("ApproximateEqual", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor approximate_equal_eager_fallback(Tensor x, Tensor y, float tolerance, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { "T", x.dtype, "tolerance", tolerance };
        var _result = _execute.execute("ApproximateEqual", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ApproximateEqual", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns the index with the largest value across dimensions of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Note that in case of ties the identity of the return value is not guaranteed.
    /// 
    /// Usage:
    ///   ```python
    ///   import tensorflow as tf
    ///   a = [1, 10, 26.9, 2.8, 166.32, 62.3]
    ///   b = tf.math.argmax(input = a)
    ///   c = tf.keras.backend.eval(b)
    ///   # c = 4
    ///   # here a[4] = 166.32 which is the largest element of a across axis 0
    ///   ```
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="dimension"></param>
    /// <param name="output_type"></param>
    /// <returns></returns>
    public static Tensor arg_max(Tensor input, Tensor dimension, TF_DataType output_type = TF_DataType.TF_INT64, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ArgMax", name) { args = new object[] { input, dimension }, attrs = new Dictionary<string, object>() { ["output_type"] = output_type } });
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
                return arg_max_eager_fallback(input, dimension, output_type: output_type, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["dimension"] = dimension;
        keywords["output_type"] = output_type;
        var _op = tf.OpDefLib._apply_op_helper("ArgMax", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tidx", _op._get_attr_type("Tidx"), "output_type", _op._get_attr_type("output_type") };
            _execute.record_gradient("ArgMax", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor arg_max_eager_fallback(Tensor input, Tensor dimension, TF_DataType output_type, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, dimension };
        object[] _attrs = new object[] { "T", input.dtype, "Tidx", dimension.dtype, "output_type", output_type };
        var _result = _execute.execute("ArgMax", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ArgMax", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns the index with the smallest value across dimensions of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Note that in case of ties the identity of the return value is not guaranteed.
    /// 
    /// Usage:
    ///   ```python
    ///   import tensorflow as tf
    ///   a = [1, 10, 26.9, 2.8, 166.32, 62.3]
    ///   b = tf.math.argmin(input = a)
    ///   c = tf.keras.backend.eval(b)
    ///   # c = 0
    ///   # here a[0] = 1 which is the smallest element of a across axis 0
    ///   ```
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="dimension"></param>
    /// <param name="output_type"></param>
    /// <returns></returns>
    public static Tensor arg_min(Tensor input, Tensor dimension, TF_DataType output_type = TF_DataType.TF_INT64, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ArgMin", name) { args = new object[] { input, dimension }, attrs = new Dictionary<string, object>() { ["output_type"] = output_type } });
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
                return arg_min_eager_fallback(input, dimension, output_type: output_type, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["dimension"] = dimension;
        keywords["output_type"] = output_type;
        var _op = tf.OpDefLib._apply_op_helper("ArgMin", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tidx", _op._get_attr_type("Tidx"), "output_type", _op._get_attr_type("output_type") };
            _execute.record_gradient("ArgMin", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor arg_min_eager_fallback(Tensor input, Tensor dimension, TF_DataType output_type, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, dimension };
        object[] _attrs = new object[] { "T", input.dtype, "Tidx", dimension.dtype, "output_type", output_type };
        var _result = _execute.execute("ArgMin", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ArgMin", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the trignometric inverse sine of x element-wise.
    /// </summary>
    /// <remarks>
    /// 
    /// The `tf.math.asin` operation returns the inverse of `tf.math.sin`, such that
    /// if `y = tf.math.sin(x)` then, `x = tf.math.asin(y)`.
    /// 
    /// **Note**: The output of `tf.math.asin` will lie within the invertible range
    /// of sine, i.e [-pi/2, pi/2].
    /// 
    /// For example:
    /// 
    /// ```python
    /// # Note: [1.047, 0.785] ~= [(pi/3), (pi/4)]
    /// x = tf.constant([1.047, 0.785])
    /// y = tf.math.sin(x) # [0.8659266, 0.7068252]
    /// 
    /// tf.math.asin(y) # [1.047, 0.785] = x
    /// ```
    /// 
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor asin(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Asin", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return asin_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Asin", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Asin", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor asin_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Asin", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Asin", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes inverse hyperbolic sine of x element-wise.
    /// </summary>
    /// <remarks>
    /// 
    ///   Given an input tensor, this function computes inverse hyperbolic sine
    ///   for every element in the tensor. Both input and output has a range of
    ///   `[-inf, inf]`.
    /// 
    ///   ```python
    ///   x = tf.constant([-float("inf"), -2, -0.5, 1, 1.2, 200, 10000, float("inf")])
    ///   tf.math.asinh(x) ==> [-inf -1.4436355 -0.4812118 0.8813736 1.0159732 5.991471 9.903487 inf]
    ///   ```
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor asinh(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Asinh", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return asinh_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Asinh", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Asinh", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor asinh_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Asinh", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Asinh", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the trignometric inverse tangent of x element-wise.
    /// </summary>
    /// <remarks>
    /// 
    /// The `tf.math.atan` operation returns the inverse of `tf.math.tan`, such that
    /// if `y = tf.math.tan(x)` then, `x = tf.math.atan(y)`.
    /// 
    /// **Note**: The output of `tf.math.atan` will lie within the invertible range
    /// of tan, i.e (-pi/2, pi/2).
    /// 
    /// For example:
    /// 
    /// ```python
    /// # Note: [1.047, 0.785] ~= [(pi/3), (pi/4)]
    /// x = tf.constant([1.047, 0.785])
    /// y = tf.math.tan(x) # [1.731261, 0.99920404]
    /// 
    /// tf.math.atan(y) # [1.047, 0.785] = x
    /// ```
    /// 
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor atan(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Atan", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return atan_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Atan", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Atan", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor atan_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Atan", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Atan", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes arctangent of `y/x` element-wise, respecting signs of the arguments.
    /// </summary>
    /// <remarks>
    /// 
    /// This is the angle \( 	heta in [-pi, pi] \) such that
    /// \[ x = r cos(	heta) \]
    /// and
    /// \[ y = r sin(	heta) \]
    /// where \(r = sqrt{x^2 + y^2} \).
    /// 
    /// For example:
    /// 
    /// >>> x = [1., 1.]
    /// >>> y = [1., -1.]
    /// >>> print((tf.math.atan2(y,x) * (180 / np.pi)).numpy())
    /// [ 45. -45.]
    /// 
    /// 
    /// 
    /// </remarks>
    /// <param name="y"></param>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor atan2(Tensor y, Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Atan2", name) { args = new object[] { y, x }, attrs = new Dictionary<string, object>() { } });
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
                return atan2_eager_fallback(y, x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["y"] = y;
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Atan2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Atan2", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor atan2_eager_fallback(Tensor y, Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { y, x };
        object[] _attrs = new object[] { "T", y.dtype };
        var _result = _execute.execute("Atan2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Atan2", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes inverse hyperbolic tangent of x element-wise.
    /// </summary>
    /// <remarks>
    /// 
    ///   Given an input tensor, this function computes inverse hyperbolic tangent
    ///   for every element in the tensor. Input range is `[-1,1]` and output range is
    ///   `[-inf, inf]`. If input is `-1`, output will be `-inf` and if the
    ///   input is `1`, output will be `inf`. Values outside the range will have
    ///   `nan` as output.
    /// 
    ///   ```python
    ///   x = tf.constant([-float("inf"), -1, -0.5, 1, 0, 0.5, 10, float("inf")])
    ///   tf.math.atanh(x) ==> [nan -inf -0.54930615 inf  0. 0.54930615 nan nan]
    ///   ```
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor atanh(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Atanh", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return atanh_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Atanh", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Atanh", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor atanh_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Atanh", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Atanh", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Multiplies slices of two tensors in batches.
    /// </summary>
    /// <remarks>
    /// 
    /// Multiplies all slices of `Tensor` `x` and `y` (each slice can be
    /// viewed as an element of a batch), and arranges the individual results
    /// in a single output tensor of the same batch size. Each of the
    /// individual slices can optionally be adjointed (to adjoint a matrix
    /// means to transpose and conjugate it) before multiplication by setting
    /// the `adj_x` or `adj_y` flag to `True`, which are by default `False`.
    /// 
    /// The input tensors `x` and `y` are 2-D or higher with shape `[..., r_x, c_x]`
    /// and `[..., r_y, c_y]`.
    /// 
    /// The output tensor is 2-D or higher with shape `[..., r_o, c_o]`, where:
    /// 
    ///     r_o = c_x if adj_x else r_x
    ///     c_o = r_y if adj_y else c_y
    /// 
    /// It is computed as:
    /// 
    ///     output[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <param name="adj_x">
    /// 
    /// If `True`, adjoint the slices of `x`. Defaults to `False`.
    /// 
    /// </param>
    /// <param name="adj_y">
    /// 
    /// If `True`, adjoint the slices of `y`. Defaults to `False`.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor batch_mat_mul(Tensor x, Tensor y, bool adj_x = false, bool adj_y = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "BatchMatMul", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { ["adj_x"] = adj_x, ["adj_y"] = adj_y } });
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
                return batch_mat_mul_eager_fallback(x, y, adj_x: adj_x, adj_y: adj_y, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        keywords["adj_x"] = adj_x;
        keywords["adj_y"] = adj_y;
        var _op = tf.OpDefLib._apply_op_helper("BatchMatMul", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "adj_x", _op._get_attr_bool("adj_x"), "adj_y", _op._get_attr_bool("adj_y") };
            _execute.record_gradient("BatchMatMul", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor batch_mat_mul_eager_fallback(Tensor x, Tensor y, bool adj_x, bool adj_y, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { "T", x.dtype, "adj_x", adj_x, "adj_y", adj_y };
        var _result = _execute.execute("BatchMatMul", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("BatchMatMul", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Multiplies slices of two tensors in batches.
    /// </summary>
    /// <remarks>
    /// 
    /// Multiplies all slices of `Tensor` `x` and `y` (each slice can be
    /// viewed as an element of a batch), and arranges the individual results
    /// in a single output tensor of the same batch size. Each of the
    /// individual slices can optionally be adjointed (to adjoint a matrix
    /// means to transpose and conjugate it) before multiplication by setting
    /// the `adj_x` or `adj_y` flag to `True`, which are by default `False`.
    /// 
    /// The input tensors `x` and `y` are 2-D or higher with shape `[..., r_x, c_x]`
    /// and `[..., r_y, c_y]`.
    /// 
    /// The output tensor is 2-D or higher with shape `[..., r_o, c_o]`, where:
    /// 
    ///     r_o = c_x if adj_x else r_x
    ///     c_o = r_y if adj_y else c_y
    /// 
    /// It is computed as:
    /// 
    ///     output[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])
    /// 
    /// *NOTE*: `BatchMatMulV2` supports broadcasting in the batch dimensions. More
    /// about broadcasting
    /// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    /// 
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <param name="adj_x">
    /// 
    /// If `True`, adjoint the slices of `x`. Defaults to `False`.
    /// 
    /// </param>
    /// <param name="adj_y">
    /// 
    /// If `True`, adjoint the slices of `y`. Defaults to `False`.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor batch_mat_mul_v2(Tensor x, Tensor y, bool adj_x = false, bool adj_y = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "BatchMatMulV2", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { ["adj_x"] = adj_x, ["adj_y"] = adj_y } });
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
                return batch_mat_mul_v2_eager_fallback(x, y, adj_x: adj_x, adj_y: adj_y, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        keywords["adj_x"] = adj_x;
        keywords["adj_y"] = adj_y;
        var _op = tf.OpDefLib._apply_op_helper("BatchMatMulV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "adj_x", _op._get_attr_bool("adj_x"), "adj_y", _op._get_attr_bool("adj_y") };
            _execute.record_gradient("BatchMatMulV2", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor batch_mat_mul_v2_eager_fallback(Tensor x, Tensor y, bool adj_x, bool adj_y, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { "T", x.dtype, "adj_x", adj_x, "adj_y", adj_y };
        var _result = _execute.execute("BatchMatMulV2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("BatchMatMulV2", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Multiplies slices of two tensors in batches.
    /// </summary>
    /// <remarks>
    /// 
    /// Multiplies all slices of `Tensor` `x` and `y` (each slice can be
    /// viewed as an element of a batch), and arranges the individual results
    /// in a single output tensor of the same batch size. Each of the
    /// individual slices can optionally be adjointed (to adjoint a matrix
    /// means to transpose and conjugate it) before multiplication by setting
    /// the `adj_x` or `adj_y` flag to `True`, which are by default `False`.
    /// 
    /// The input tensors `x` and `y` are 2-D or higher with shape `[..., r_x, c_x]`
    /// and `[..., r_y, c_y]`.
    /// 
    /// The output tensor is 2-D or higher with shape `[..., r_o, c_o]`, where:
    /// 
    ///     r_o = c_x if adj_x else r_x
    ///     c_o = r_y if adj_y else c_y
    /// 
    /// It is computed as:
    /// 
    ///     output[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])
    /// 
    /// *NOTE*: `BatchMatMulV3` supports broadcasting in the batch dimensions. More
    /// about broadcasting
    /// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    /// 
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <param name="Tout">
    /// 
    /// If not spcified, Tout is the same type to input type.
    /// 
    /// </param>
    /// <param name="adj_x">
    /// 
    /// If `True`, adjoint the slices of `x`. Defaults to `False`.
    /// 
    /// </param>
    /// <param name="adj_y">
    /// 
    /// If `True`, adjoint the slices of `y`. Defaults to `False`.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor batch_mat_mul_v3(Tensor x, Tensor y, TF_DataType Tout, bool adj_x = false, bool adj_y = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "BatchMatMulV3", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { ["Tout"] = Tout, ["adj_x"] = adj_x, ["adj_y"] = adj_y } });
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
                return batch_mat_mul_v3_eager_fallback(x, y, Tout: Tout, adj_x: adj_x, adj_y: adj_y, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        keywords["Tout"] = Tout;
        keywords["adj_x"] = adj_x;
        keywords["adj_y"] = adj_y;
        var _op = tf.OpDefLib._apply_op_helper("BatchMatMulV3", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Ta", _op._get_attr_type("Ta"), "Tb", _op._get_attr_type("Tb"), "Tout", _op._get_attr_type("Tout"), "adj_x", _op._get_attr_bool("adj_x"), "adj_y", _op._get_attr_bool("adj_y") };
            _execute.record_gradient("BatchMatMulV3", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor batch_mat_mul_v3_eager_fallback(Tensor x, Tensor y, TF_DataType Tout, bool adj_x, bool adj_y, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { "Ta", x.dtype, "Tb", y.dtype, "Tout", Tout, "adj_x", adj_x, "adj_y", adj_y };
        var _result = _execute.execute("BatchMatMulV3", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("BatchMatMulV3", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Compute the regularized incomplete beta integral \\(I_x(a, b)\\).
    /// </summary>
    /// <remarks>
    /// 
    /// The regularized incomplete beta integral is defined as:
    /// 
    /// 
    /// \(I_x(a, b) = rac{B(x; a, b)}{B(a, b)}\)
    /// 
    /// where
    /// 
    /// 
    /// \(B(x; a, b) = int_0^x t^{a-1} (1 - t)^{b-1} dt\)
    /// 
    /// 
    /// is the incomplete beta function and \(B(a, b)\) is the *complete*
    /// beta function.
    /// 
    /// </remarks>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor betainc(Tensor a, Tensor b, Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Betainc", name) { args = new object[] { a, b, x }, attrs = new Dictionary<string, object>() { } });
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
                return betainc_eager_fallback(a, b, x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["a"] = a;
        keywords["b"] = b;
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Betainc", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Betainc", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor betainc_eager_fallback(Tensor a, Tensor b, Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { a, b, x };
        object[] _attrs = new object[] { "T", a.dtype };
        var _result = _execute.execute("Betainc", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Betainc", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Counts the number of occurrences of each value in an integer array.
    /// </summary>
    /// <remarks>
    /// 
    /// Outputs a vector with length `size` and the same dtype as `weights`. If
    /// `weights` are empty, then index `i` stores the number of times the value `i` is
    /// counted in `arr`. If `weights` are non-empty, then index `i` stores the sum of
    /// the value in `weights` at each index where the corresponding value in `arr` is
    /// `i`.
    /// 
    /// Values in `arr` outside of the range [0, size) are ignored.
    /// 
    /// </remarks>
    /// <param name="arr"></param>
    /// <param name="size"></param>
    /// <param name="weights"></param>
    /// <returns></returns>
    public static Tensor bincount(Tensor arr, Tensor size, Tensor weights, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Bincount", name) { args = new object[] { arr, size, weights }, attrs = new Dictionary<string, object>() { } });
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
                return bincount_eager_fallback(arr, size, weights, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["arr"] = arr;
        keywords["size"] = size;
        keywords["weights"] = weights;
        var _op = tf.OpDefLib._apply_op_helper("Bincount", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Bincount", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor bincount_eager_fallback(Tensor arr, Tensor size, Tensor weights, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { arr, size, weights };
        object[] _attrs = new object[] { "T", weights.dtype };
        var _result = _execute.execute("Bincount", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Bincount", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Bucketizes 'input' based on 'boundaries'.
    /// </summary>
    /// <remarks>
    /// 
    /// For example, if the inputs are
    ///     boundaries = [0, 10, 100]
    ///     input = [[-5, 10000]
    ///              [150,   10]
    ///              [5,    100]]
    /// 
    /// then the output will be
    ///     output = [[0, 3]
    ///               [3, 2]
    ///               [1, 3]]
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="boundaries">
    /// 
    /// A sorted list of floats gives the boundary of the buckets.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor bucketize(Tensor input, float[] boundaries, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Bucketize", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { ["boundaries"] = boundaries } });
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
                return bucketize_eager_fallback(input, boundaries: boundaries, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["boundaries"] = boundaries;
        var _op = tf.OpDefLib._apply_op_helper("Bucketize", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "boundaries", _op.get_attr("boundaries") };
            _execute.record_gradient("Bucketize", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor bucketize_eager_fallback(Tensor input, float[] boundaries, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "T", input.dtype, "boundaries", boundaries };
        var _result = _execute.execute("Bucketize", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Bucketize", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Cast x of type SrcT to y of DstT.
    /// </summary>
    /// <param name="x"></param>
    /// <param name="DstT"></param>
    /// <param name="Truncate"></param>
    /// <returns></returns>
    public static Tensor cast(Tensor x, TF_DataType DstT, bool Truncate = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Cast", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { ["DstT"] = DstT, ["Truncate"] = Truncate } });
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
                return cast_eager_fallback(x, DstT: DstT, Truncate: Truncate, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["DstT"] = DstT;
        keywords["Truncate"] = Truncate;
        var _op = tf.OpDefLib._apply_op_helper("Cast", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "SrcT", _op._get_attr_type("SrcT"), "DstT", _op._get_attr_type("DstT"), "Truncate", _op._get_attr_bool("Truncate") };
            _execute.record_gradient("Cast", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor cast_eager_fallback(Tensor x, TF_DataType DstT, bool Truncate, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "SrcT", x.dtype, "DstT", DstT, "Truncate", Truncate };
        var _result = _execute.execute("Cast", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Cast", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns element-wise smallest integer not less than x.
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor ceil(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Ceil", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return ceil_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Ceil", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Ceil", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor ceil_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Ceil", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Ceil", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Clips tensor values to a specified min and max.
    /// </summary>
    /// <remarks>
    /// 
    /// Given a tensor `t`, this operation returns a tensor of the same type and
    /// shape as `t` with its values clipped to `clip_value_min` and `clip_value_max`.
    /// Any values less than `clip_value_min` are set to `clip_value_min`. Any values
    /// greater than `clip_value_max` are set to `clip_value_max`.
    /// 
    /// </remarks>
    /// <param name="t"></param>
    /// <param name="clip_value_min"></param>
    /// <param name="clip_value_max"></param>
    /// <returns></returns>
    public static Tensor clip_by_value(Tensor t, Tensor clip_value_min, Tensor clip_value_max, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ClipByValue", name) { args = new object[] { t, clip_value_min, clip_value_max }, attrs = new Dictionary<string, object>() { } });
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
                return clip_by_value_eager_fallback(t, clip_value_min, clip_value_max, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["t"] = t;
        keywords["clip_value_min"] = clip_value_min;
        keywords["clip_value_max"] = clip_value_max;
        var _op = tf.OpDefLib._apply_op_helper("ClipByValue", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("ClipByValue", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor clip_by_value_eager_fallback(Tensor t, Tensor clip_value_min, Tensor clip_value_max, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { t, clip_value_min, clip_value_max };
        object[] _attrs = new object[] { "T", t.dtype };
        var _result = _execute.execute("ClipByValue", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ClipByValue", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Converts two real numbers to a complex number.
    /// </summary>
    /// <remarks>
    /// 
    /// Given a tensor `real` representing the real part of a complex number, and a
    /// tensor `imag` representing the imaginary part of a complex number, this
    /// operation returns complex numbers elementwise of the form \(a + bj\), where
    /// *a* represents the `real` part and *b* represents the `imag` part.
    /// 
    /// The input tensors `real` and `imag` must have the same shape.
    /// 
    /// For example:
    /// 
    /// ```
    /// # tensor 'real' is [2.25, 3.25]
    /// # tensor `imag` is [4.75, 5.75]
    /// tf.complex(real, imag) ==> [[2.25 + 4.75j], [3.25 + 5.75j]]
    /// ```
    /// 
    /// </remarks>
    /// <param name="real"></param>
    /// <param name="imag"></param>
    /// <param name="Tout"></param>
    /// <returns></returns>
    public static Tensor complex(Tensor real, Tensor imag, TF_DataType Tout = TF_DataType.TF_COMPLEX64, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Complex", name) { args = new object[] { real, imag }, attrs = new Dictionary<string, object>() { ["Tout"] = Tout } });
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
                return complex_eager_fallback(real, imag, Tout: Tout, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["real"] = real;
        keywords["imag"] = imag;
        keywords["Tout"] = Tout;
        var _op = tf.OpDefLib._apply_op_helper("Complex", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tout", _op._get_attr_type("Tout") };
            _execute.record_gradient("Complex", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor complex_eager_fallback(Tensor real, Tensor imag, TF_DataType Tout, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { real, imag };
        object[] _attrs = new object[] { "T", real.dtype, "Tout", Tout };
        var _result = _execute.execute("Complex", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Complex", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the complex absolute value of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Given a tensor `x` of complex numbers, this operation returns a tensor of type
    /// `float` or `double` that is the absolute value of each element in `x`. All
    /// elements in `x` must be complex numbers of the form \(a + bj\). The absolute
    /// value is computed as \( sqrt{a^2 + b^2}\).
    /// 
    /// For example:
    /// 
    /// >>> x = tf.complex(3.0, 4.0)
    /// >>> print((tf.raw_ops.ComplexAbs(x=x, Tout=tf.dtypes.float32, name=None)).numpy())
    /// 5.0
    /// 
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="Tout"></param>
    /// <returns></returns>
    public static Tensor complex_abs(Tensor x, TF_DataType Tout = TF_DataType.TF_FLOAT, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ComplexAbs", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { ["Tout"] = Tout } });
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
                return complex_abs_eager_fallback(x, Tout: Tout, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["Tout"] = Tout;
        var _op = tf.OpDefLib._apply_op_helper("ComplexAbs", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tout", _op._get_attr_type("Tout") };
            _execute.record_gradient("ComplexAbs", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor complex_abs_eager_fallback(Tensor x, TF_DataType Tout, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype, "Tout", Tout };
        var _result = _execute.execute("ComplexAbs", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ComplexAbs", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns the complex conjugate of a complex number.
    /// </summary>
    /// <remarks>
    /// 
    /// Given a tensor `input` of complex numbers, this operation returns a tensor of
    /// complex numbers that are the complex conjugate of each element in `input`. The
    /// complex numbers in `input` must be of the form \(a + bj\), where *a* is the
    /// real part and *b* is the imaginary part.
    /// 
    /// The complex conjugate returned by this operation is of the form \(a - bj\).
    /// 
    /// For example:
    /// 
    /// ```
    /// # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
    /// tf.conj(input) ==> [-2.25 - 4.75j, 3.25 - 5.75j]
    /// ```
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <returns></returns>
    public static Tensor conj(Tensor input, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Conj", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { } });
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
                return conj_eager_fallback(input, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        var _op = tf.OpDefLib._apply_op_helper("Conj", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Conj", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor conj_eager_fallback(Tensor input, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "T", input.dtype };
        var _result = _execute.execute("Conj", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Conj", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes cos of x element-wise.
    /// </summary>
    /// <remarks>
    /// 
    ///   Given an input tensor, this function computes cosine of every
    ///   element in the tensor. Input range is `(-inf, inf)` and
    ///   output range is `[-1,1]`. If input lies outside the boundary, `nan`
    ///   is returned.
    /// 
    ///   ```python
    ///   x = tf.constant([-float("inf"), -9, -0.5, 1, 1.2, 200, 10000, float("inf")])
    ///   tf.math.cos(x) ==> [nan -0.91113025 0.87758255 0.5403023 0.36235774 0.48718765 -0.95215535 nan]
    ///   ```
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor cos(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Cos", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return cos_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Cos", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Cos", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor cos_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Cos", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Cos", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes hyperbolic cosine of x element-wise.
    /// </summary>
    /// <remarks>
    /// 
    ///   Given an input tensor, this function computes hyperbolic cosine of every
    ///   element in the tensor. Input range is `[-inf, inf]` and output range
    ///   is `[1, inf]`.
    /// 
    ///   ```python
    ///   x = tf.constant([-float("inf"), -9, -0.5, 1, 1.2, 2, 10, float("inf")])
    ///   tf.math.cosh(x) ==> [inf 4.0515420e+03 1.1276259e+00 1.5430807e+00 1.8106556e+00 3.7621956e+00 1.1013233e+04 inf]
    ///   ```
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor cosh(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Cosh", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return cosh_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Cosh", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Cosh", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor cosh_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Cosh", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Cosh", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Compute the pairwise cross product.
    /// </summary>
    /// <remarks>
    /// 
    /// `a` and `b` must be the same shape; they can either be simple 3-element vectors,
    /// or any shape where the innermost dimension is 3. In the latter case, each pair
    /// of corresponding 3-element vectors is cross-multiplied independently.
    /// 
    /// </remarks>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <returns></returns>
    public static Tensor cross(Tensor a, Tensor b, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Cross", name) { args = new object[] { a, b }, attrs = new Dictionary<string, object>() { } });
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
                return cross_eager_fallback(a, b, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["a"] = a;
        keywords["b"] = b;
        var _op = tf.OpDefLib._apply_op_helper("Cross", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Cross", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor cross_eager_fallback(Tensor a, Tensor b, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { a, b };
        object[] _attrs = new object[] { "T", a.dtype };
        var _result = _execute.execute("Cross", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Cross", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Compute the cumulative product of the tensor `x` along `axis`.
    /// </summary>
    /// <remarks>
    /// 
    /// By default, this op performs an inclusive cumprod, which means that the first
    /// element of the input is identical to the first element of the output:
    /// 
    /// ```python
    /// tf.cumprod([a, b, c])  # => [a, a * b, a * b * c]
    /// ```
    /// 
    /// By setting the `exclusive` kwarg to `True`, an exclusive cumprod is
    /// performed instead:
    /// 
    /// ```python
    /// tf.cumprod([a, b, c], exclusive=True)  # => [1, a, a * b]
    /// ```
    /// 
    /// By setting the `reverse` kwarg to `True`, the cumprod is performed in the
    /// opposite direction:
    /// 
    /// ```python
    /// tf.cumprod([a, b, c], reverse=True)  # => [a * b * c, b * c, c]
    /// ```
    /// 
    /// This is more efficient than using separate `tf.reverse` ops.
    /// 
    /// The `reverse` and `exclusive` kwargs can also be combined:
    /// 
    /// ```python
    /// tf.cumprod([a, b, c], exclusive=True, reverse=True)  # => [b * c, c, 1]
    /// ```
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="axis"></param>
    /// <param name="exclusive">
    /// 
    /// If `True`, perform exclusive cumprod.
    /// 
    /// </param>
    /// <param name="reverse">
    /// 
    /// A `bool` (default: False).
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor cumprod(Tensor x, Tensor axis, bool exclusive = false, bool reverse = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Cumprod", name) { args = new object[] { x, axis }, attrs = new Dictionary<string, object>() { ["exclusive"] = exclusive, ["reverse"] = reverse } });
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
                return cumprod_eager_fallback(x, axis, exclusive: exclusive, reverse: reverse, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["axis"] = axis;
        keywords["exclusive"] = exclusive;
        keywords["reverse"] = reverse;
        var _op = tf.OpDefLib._apply_op_helper("Cumprod", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "exclusive", _op._get_attr_bool("exclusive"), "reverse", _op._get_attr_bool("reverse"), "T", _op._get_attr_type("T"), "Tidx", _op._get_attr_type("Tidx") };
            _execute.record_gradient("Cumprod", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor cumprod_eager_fallback(Tensor x, Tensor axis, bool exclusive, bool reverse, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, axis };
        object[] _attrs = new object[] { "exclusive", exclusive, "reverse", reverse, "T", x.dtype, "Tidx", axis.dtype };
        var _result = _execute.execute("Cumprod", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Cumprod", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Compute the cumulative sum of the tensor `x` along `axis`.
    /// </summary>
    /// <remarks>
    /// 
    /// By default, this op performs an inclusive cumsum, which means that the first
    /// element of the input is identical to the first element of the output:
    /// 
    /// ```python
    /// tf.cumsum([a, b, c])  # => [a, a + b, a + b + c]
    /// ```
    /// 
    /// By setting the `exclusive` kwarg to `True`, an exclusive cumsum is
    /// performed instead:
    /// 
    /// ```python
    /// tf.cumsum([a, b, c], exclusive=True)  # => [0, a, a + b]
    /// ```
    /// 
    /// By setting the `reverse` kwarg to `True`, the cumsum is performed in the
    /// opposite direction:
    /// 
    /// ```python
    /// tf.cumsum([a, b, c], reverse=True)  # => [a + b + c, b + c, c]
    /// ```
    /// 
    /// This is more efficient than using separate `tf.reverse` ops.
    /// 
    /// The `reverse` and `exclusive` kwargs can also be combined:
    /// 
    /// ```python
    /// tf.cumsum([a, b, c], exclusive=True, reverse=True)  # => [b + c, c, 0]
    /// ```
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="axis"></param>
    /// <param name="exclusive">
    /// 
    /// If `True`, perform exclusive cumsum.
    /// 
    /// </param>
    /// <param name="reverse">
    /// 
    /// A `bool` (default: False).
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor cumsum(Tensor x, Tensor axis, bool exclusive = false, bool reverse = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Cumsum", name) { args = new object[] { x, axis }, attrs = new Dictionary<string, object>() { ["exclusive"] = exclusive, ["reverse"] = reverse } });
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
                return cumsum_eager_fallback(x, axis, exclusive: exclusive, reverse: reverse, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["axis"] = axis;
        keywords["exclusive"] = exclusive;
        keywords["reverse"] = reverse;
        var _op = tf.OpDefLib._apply_op_helper("Cumsum", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "exclusive", _op._get_attr_bool("exclusive"), "reverse", _op._get_attr_bool("reverse"), "T", _op._get_attr_type("T"), "Tidx", _op._get_attr_type("Tidx") };
            _execute.record_gradient("Cumsum", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor cumsum_eager_fallback(Tensor x, Tensor axis, bool exclusive, bool reverse, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, axis };
        object[] _attrs = new object[] { "exclusive", exclusive, "reverse", reverse, "T", x.dtype, "Tidx", axis.dtype };
        var _result = _execute.execute("Cumsum", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Cumsum", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Compute the cumulative product of the tensor `x` along `axis`.
    /// </summary>
    /// <remarks>
    /// 
    /// By default, this op performs an inclusive cumulative log-sum-exp,
    /// which means that the first
    /// element of the input is identical to the first element of the output:
    /// ```python
    /// tf.math.cumulative_logsumexp([a, b, c])  # => [a, log(exp(a) + exp(b)), log(exp(a) + exp(b) + exp(c))]
    /// ```
    /// 
    /// By setting the `exclusive` kwarg to `True`, an exclusive cumulative log-sum-exp is
    /// performed instead:
    /// ```python
    /// tf.cumulative_logsumexp([a, b, c], exclusive=True)  # => [-inf, a, log(exp(a) * exp(b))]
    /// ```
    /// Note that the neutral element of the log-sum-exp operation is `-inf`,
    /// however, for performance reasons, the minimal value representable by the
    /// floating point type is used instead.
    /// 
    /// By setting the `reverse` kwarg to `True`, the cumulative log-sum-exp is performed in the
    /// opposite direction.
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="axis"></param>
    /// <param name="exclusive">
    /// 
    /// If `True`, perform exclusive cumulative log-sum-exp.
    /// 
    /// </param>
    /// <param name="reverse">
    /// 
    /// A `bool` (default: False).
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor cumulative_logsumexp(Tensor x, Tensor axis, bool exclusive = false, bool reverse = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "CumulativeLogsumexp", name) { args = new object[] { x, axis }, attrs = new Dictionary<string, object>() { ["exclusive"] = exclusive, ["reverse"] = reverse } });
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
                return cumulative_logsumexp_eager_fallback(x, axis, exclusive: exclusive, reverse: reverse, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["axis"] = axis;
        keywords["exclusive"] = exclusive;
        keywords["reverse"] = reverse;
        var _op = tf.OpDefLib._apply_op_helper("CumulativeLogsumexp", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "exclusive", _op._get_attr_bool("exclusive"), "reverse", _op._get_attr_bool("reverse"), "T", _op._get_attr_type("T"), "Tidx", _op._get_attr_type("Tidx") };
            _execute.record_gradient("CumulativeLogsumexp", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor cumulative_logsumexp_eager_fallback(Tensor x, Tensor axis, bool exclusive, bool reverse, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, axis };
        object[] _attrs = new object[] { "exclusive", exclusive, "reverse", reverse, "T", x.dtype, "Tidx", axis.dtype };
        var _result = _execute.execute("CumulativeLogsumexp", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("CumulativeLogsumexp", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Counts the number of occurrences of each value in an integer array.
    /// </summary>
    /// <remarks>
    /// 
    /// Outputs a vector with length `size` and the same dtype as `weights`. If
    /// `weights` are empty, then index `i` stores the number of times the value `i` is
    /// counted in `arr`. If `weights` are non-empty, then index `i` stores the sum of
    /// the value in `weights` at each index where the corresponding value in `arr` is
    /// `i`.
    /// 
    /// Values in `arr` outside of the range [0, size) are ignored.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="size"></param>
    /// <param name="weights"></param>
    /// <param name="binary_output">
    /// 
    /// bool; Whether the kernel should count the appearance or number of occurrences.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor dense_bincount(Tensor input, Tensor size, Tensor weights, bool binary_output = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "DenseBincount", name) { args = new object[] { input, size, weights }, attrs = new Dictionary<string, object>() { ["binary_output"] = binary_output } });
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
                return dense_bincount_eager_fallback(input, size, weights, binary_output: binary_output, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["size"] = size;
        keywords["weights"] = weights;
        keywords["binary_output"] = binary_output;
        var _op = tf.OpDefLib._apply_op_helper("DenseBincount", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tidx", _op._get_attr_type("Tidx"), "T", _op._get_attr_type("T"), "binary_output", _op._get_attr_bool("binary_output") };
            _execute.record_gradient("DenseBincount", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor dense_bincount_eager_fallback(Tensor input, Tensor size, Tensor weights, bool binary_output, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, size, weights };
        object[] _attrs = new object[] { "Tidx", input.dtype, "T", weights.dtype, "binary_output", binary_output };
        var _result = _execute.execute("DenseBincount", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("DenseBincount", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes Psi, the derivative of Lgamma (the log of the absolute value of
    /// </summary>
    /// <remarks>
    /// 
    /// `Gamma(x)`), element-wise.
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor digamma(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Digamma", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return digamma_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Digamma", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Digamma", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor digamma_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Digamma", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Digamma", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns x / y element-wise.
    /// </summary>
    /// <remarks>
    /// 
    /// *NOTE*: `Div` supports broadcasting. More about broadcasting
    /// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <returns></returns>
    public static Tensor div(Tensor x, Tensor y, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Div", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { } });
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
                return div_eager_fallback(x, y, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        var _op = tf.OpDefLib._apply_op_helper("Div", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Div", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor div_eager_fallback(Tensor x, Tensor y, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Div", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Div", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns 0 if the denominator is zero.
    /// </summary>
    /// <remarks>
    /// 
    /// 
    /// *NOTE*: `DivNoNan` supports broadcasting. More about broadcasting
    /// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <returns></returns>
    public static Tensor div_no_nan(Tensor x, Tensor y, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "DivNoNan", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { } });
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
                return div_no_nan_eager_fallback(x, y, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        var _op = tf.OpDefLib._apply_op_helper("DivNoNan", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("DivNoNan", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor div_no_nan_eager_fallback(Tensor x, Tensor y, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("DivNoNan", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("DivNoNan", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns the truth value of (x == y) element-wise.
    /// </summary>
    /// <remarks>
    /// 
    /// *NOTE*: `Equal` supports broadcasting. More about broadcasting
    /// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    /// 
    /// ```python
    /// x = tf.constant([2, 4])
    /// y = tf.constant(2)
    /// tf.math.equal(x, y) ==> array([True, False])
    /// 
    /// x = tf.constant([2, 4])
    /// y = tf.constant([2, 4])
    /// tf.math.equal(x, y) ==> array([True,  True])
    /// ```
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <param name="incompatible_shape_error"></param>
    /// <returns></returns>
    public static Tensor equal(Tensor x, Tensor y, bool incompatible_shape_error = true, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Equal", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { ["incompatible_shape_error"] = incompatible_shape_error } });
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
                return equal_eager_fallback(x, y, incompatible_shape_error: incompatible_shape_error, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        keywords["incompatible_shape_error"] = incompatible_shape_error;
        var _op = tf.OpDefLib._apply_op_helper("Equal", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "incompatible_shape_error", _op._get_attr_bool("incompatible_shape_error") };
            _execute.record_gradient("Equal", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor equal_eager_fallback(Tensor x, Tensor y, bool incompatible_shape_error, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { "T", x.dtype, "incompatible_shape_error", incompatible_shape_error };
        var _result = _execute.execute("Equal", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Equal", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the [Gauss error function](https://en.wikipedia.org/wiki/Error_function) of `x` element-wise. In statistics, for non-negative values of $x$, the error function has the following interpretation: for a random variable $Y$ that is normally distributed with mean 0 and variance $1/\sqrt{2}$, $erf(x)$ is the probability that $Y$ falls in the range $[−x, x]$.
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor erf(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Erf", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return erf_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Erf", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Erf", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor erf_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Erf", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Erf", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the complementary error function of `x` element-wise.
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor erfc(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Erfc", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return erfc_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Erfc", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Erfc", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor erfc_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Erfc", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Erfc", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor erfinv(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Erfinv", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return erfinv_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Erfinv", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Erfinv", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor erfinv_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Erfinv", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Erfinv", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the euclidean norm of elements across dimensions of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Reduces `input` along the dimensions given in `reduction_indices`. Unless
    /// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
    /// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
    /// retained with length 1.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="reduction_indices"></param>
    /// <param name="keep_dims">
    /// 
    /// If true, retain reduced dimensions with length 1.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor euclidean_norm(Tensor input, Tensor reduction_indices, bool keep_dims = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "EuclideanNorm", name) { args = new object[] { input, reduction_indices }, attrs = new Dictionary<string, object>() { ["keep_dims"] = keep_dims } });
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
                return euclidean_norm_eager_fallback(input, reduction_indices, keep_dims: keep_dims, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["reduction_indices"] = reduction_indices;
        keywords["keep_dims"] = keep_dims;
        var _op = tf.OpDefLib._apply_op_helper("EuclideanNorm", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "keep_dims", _op._get_attr_bool("keep_dims"), "T", _op._get_attr_type("T"), "Tidx", _op._get_attr_type("Tidx") };
            _execute.record_gradient("EuclideanNorm", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor euclidean_norm_eager_fallback(Tensor input, Tensor reduction_indices, bool keep_dims, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, reduction_indices };
        object[] _attrs = new object[] { "keep_dims", keep_dims, "T", input.dtype, "Tidx", reduction_indices.dtype };
        var _result = _execute.execute("EuclideanNorm", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("EuclideanNorm", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes exponential of x element-wise.  \\(y = e^x\\).
    /// </summary>
    /// <remarks>
    /// 
    ///   This function computes the exponential of every element in the input tensor.
    ///   i.e. `exp(x)` or `e^(x)`, where `x` is the input tensor.
    ///   `e` denotes Euler's number and is approximately equal to 2.718281.
    ///   Output is positive for any real input.
    /// 
    ///   ```python
    ///   x = tf.constant(2.0)
    ///   tf.math.exp(x) ==> 7.389056
    /// 
    ///   x = tf.constant([2.0, 8.0])
    ///   tf.math.exp(x) ==> array([7.389056, 2980.958], dtype=float32)
    ///   ```
    /// 
    ///   For complex numbers, the exponential value is calculated as follows:
    /// 
    ///   ```
    ///   e^(x+iy) = e^x * e^iy = e^x * (cos y + i sin y)
    ///   ```
    /// 
    ///   Let's consider complex number 1+1j as an example.
    ///   e^1 * (cos 1 + i sin 1) = 2.7182818284590 * (0.54030230586+0.8414709848j)
    /// 
    ///   ```python
    ///   x = tf.constant(1 + 1j)
    ///   tf.math.exp(x) ==> 1.4686939399158851+2.2873552871788423j
    ///   ```
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor exp(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Exp", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return exp_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Exp", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Exp", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor exp_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Exp", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Exp", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes `exp(x) - 1` element-wise.
    /// </summary>
    /// <remarks>
    /// 
    ///   i.e. `exp(x) - 1` or `e^(x) - 1`, where `x` is the input tensor.
    ///   `e` denotes Euler's number and is approximately equal to 2.718281.
    /// 
    ///   ```python
    ///   x = tf.constant(2.0)
    ///   tf.math.expm1(x) ==> 6.389056
    /// 
    ///   x = tf.constant([2.0, 8.0])
    ///   tf.math.expm1(x) ==> array([6.389056, 2979.958], dtype=float32)
    /// 
    ///   x = tf.constant(1 + 1j)
    ///   tf.math.expm1(x) ==> (0.46869393991588515+2.2873552871788423j)
    ///   ```
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor expm1(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Expm1", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return expm1_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Expm1", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Expm1", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor expm1_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Expm1", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Expm1", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns element-wise largest integer not greater than x.
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor floor(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Floor", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return floor_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Floor", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Floor", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor floor_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Floor", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Floor", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns x // y element-wise.
    /// </summary>
    /// <remarks>
    /// 
    /// *NOTE*: `FloorDiv` supports broadcasting. More about broadcasting
    /// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <returns></returns>
    public static Tensor floor_div(Tensor x, Tensor y, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "FloorDiv", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { } });
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
                return floor_div_eager_fallback(x, y, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        var _op = tf.OpDefLib._apply_op_helper("FloorDiv", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("FloorDiv", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor floor_div_eager_fallback(Tensor x, Tensor y, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("FloorDiv", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("FloorDiv", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns element-wise remainder of division.
    /// </summary>
    /// <remarks>
    /// 
    /// This follows Python semantics in that the
    /// result here is consistent with a flooring divide. E.g.
    /// `floor(x / y) * y + floormod(x, y) = x`, regardless of the signs of x and y.
    /// 
    /// *NOTE*: `FloorMod` supports broadcasting. More about broadcasting
    /// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <returns></returns>
    public static Tensor floor_mod(Tensor x, Tensor y, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "FloorMod", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { } });
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
                return floor_mod_eager_fallback(x, y, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        var _op = tf.OpDefLib._apply_op_helper("FloorMod", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("FloorMod", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor floor_mod_eager_fallback(Tensor x, Tensor y, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("FloorMod", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("FloorMod", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns the truth value of (x > y) element-wise.
    /// </summary>
    /// <remarks>
    /// 
    /// *NOTE*: `Greater` supports broadcasting. More about broadcasting
    /// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    /// 
    /// Example:
    /// 
    /// ```python
    /// x = tf.constant([5, 4, 6])
    /// y = tf.constant([5, 2, 5])
    /// tf.math.greater(x, y) ==> [False, True, True]
    /// 
    /// x = tf.constant([5, 4, 6])
    /// y = tf.constant([5])
    /// tf.math.greater(x, y) ==> [False, False, True]
    /// ```
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <returns></returns>
    public static Tensor greater(Tensor x, Tensor y, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Greater", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { } });
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
                return greater_eager_fallback(x, y, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        var _op = tf.OpDefLib._apply_op_helper("Greater", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Greater", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor greater_eager_fallback(Tensor x, Tensor y, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Greater", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Greater", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns the truth value of (x >= y) element-wise.
    /// </summary>
    /// <remarks>
    /// 
    /// *NOTE*: `GreaterEqual` supports broadcasting. More about broadcasting
    /// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    /// 
    /// Example:
    /// 
    /// ```python
    /// x = tf.constant([5, 4, 6, 7])
    /// y = tf.constant([5, 2, 5, 10])
    /// tf.math.greater_equal(x, y) ==> [True, True, True, False]
    /// 
    /// x = tf.constant([5, 4, 6, 7])
    /// y = tf.constant([5])
    /// tf.math.greater_equal(x, y) ==> [True, False, True, True]
    /// ```
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <returns></returns>
    public static Tensor greater_equal(Tensor x, Tensor y, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "GreaterEqual", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { } });
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
                return greater_equal_eager_fallback(x, y, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        var _op = tf.OpDefLib._apply_op_helper("GreaterEqual", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("GreaterEqual", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor greater_equal_eager_fallback(Tensor x, Tensor y, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("GreaterEqual", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("GreaterEqual", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Return histogram of values.
    /// </summary>
    /// <remarks>
    /// 
    /// Given the tensor `values`, this operation returns a rank 1 histogram counting
    /// the number of entries in `values` that fall into every bin.  The bins are
    /// equal width and determined by the arguments `value_range` and `nbins`.
    /// 
    /// ```python
    /// # Bins will be:  (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
    /// nbins = 5
    /// value_range = [0.0, 5.0]
    /// new_values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]
    /// 
    /// with tf.get_default_session() as sess:
    ///   hist = tf.histogram_fixed_width(new_values, value_range, nbins=5)
    ///   variables.global_variables_initializer().run()
    ///   sess.run(hist) => [2, 1, 1, 0, 2]
    /// ```
    /// 
    /// </remarks>
    /// <param name="values"></param>
    /// <param name="value_range"></param>
    /// <param name="nbins"></param>
    /// <param name="dtype"></param>
    /// <returns></returns>
    public static Tensor histogram_fixed_width(Tensor values, Tensor value_range, Tensor nbins, TF_DataType dtype = TF_DataType.TF_INT32, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "HistogramFixedWidth", name) { args = new object[] { values, value_range, nbins }, attrs = new Dictionary<string, object>() { ["dtype"] = dtype } });
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
                return histogram_fixed_width_eager_fallback(values, value_range, nbins, dtype: dtype, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["values"] = values;
        keywords["value_range"] = value_range;
        keywords["nbins"] = nbins;
        keywords["dtype"] = dtype;
        var _op = tf.OpDefLib._apply_op_helper("HistogramFixedWidth", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "dtype", _op._get_attr_type("dtype") };
            _execute.record_gradient("HistogramFixedWidth", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor histogram_fixed_width_eager_fallback(Tensor values, Tensor value_range, Tensor nbins, TF_DataType dtype, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { values, value_range, nbins };
        object[] _attrs = new object[] { "T", values.dtype, "dtype", dtype };
        var _result = _execute.execute("HistogramFixedWidth", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("HistogramFixedWidth", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Compute the lower regularized incomplete Gamma function `P(a, x)`.
    /// </summary>
    /// <remarks>
    /// 
    /// The lower regularized incomplete Gamma function is defined as:
    /// 
    /// 
    /// \(P(a, x) = gamma(a, x) / Gamma(a) = 1 - Q(a, x)\)
    /// 
    /// where
    /// 
    /// \(gamma(a, x) = \int_{0}^{x} t^{a-1} exp(-t) dt\)
    /// 
    /// is the lower incomplete Gamma function.
    /// 
    /// Note, above `Q(a, x)` (`Igammac`) is the upper regularized complete
    /// Gamma function.
    /// 
    /// </remarks>
    /// <param name="a"></param>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor igamma(Tensor a, Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Igamma", name) { args = new object[] { a, x }, attrs = new Dictionary<string, object>() { } });
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
                return igamma_eager_fallback(a, x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["a"] = a;
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Igamma", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Igamma", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor igamma_eager_fallback(Tensor a, Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { a, x };
        object[] _attrs = new object[] { "T", a.dtype };
        var _result = _execute.execute("Igamma", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Igamma", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the gradient of `igamma(a, x)` wrt `a`.
    /// </summary>
    /// <param name="a"></param>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor igamma_grad_a(Tensor a, Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "IgammaGradA", name) { args = new object[] { a, x }, attrs = new Dictionary<string, object>() { } });
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
                return igamma_grad_a_eager_fallback(a, x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["a"] = a;
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("IgammaGradA", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("IgammaGradA", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor igamma_grad_a_eager_fallback(Tensor a, Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { a, x };
        object[] _attrs = new object[] { "T", a.dtype };
        var _result = _execute.execute("IgammaGradA", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("IgammaGradA", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Compute the upper regularized incomplete Gamma function `Q(a, x)`.
    /// </summary>
    /// <remarks>
    /// 
    /// The upper regularized incomplete Gamma function is defined as:
    /// 
    /// \(Q(a, x) = Gamma(a, x) / Gamma(a) = 1 - P(a, x)\)
    /// 
    /// where
    /// 
    /// \(Gamma(a, x) = int_{x}^{infty} t^{a-1} exp(-t) dt\)
    /// 
    /// is the upper incomplete Gamma function.
    /// 
    /// Note, above `P(a, x)` (`Igamma`) is the lower regularized complete
    /// Gamma function.
    /// 
    /// </remarks>
    /// <param name="a"></param>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor igammac(Tensor a, Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Igammac", name) { args = new object[] { a, x }, attrs = new Dictionary<string, object>() { } });
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
                return igammac_eager_fallback(a, x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["a"] = a;
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Igammac", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Igammac", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor igammac_eager_fallback(Tensor a, Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { a, x };
        object[] _attrs = new object[] { "T", a.dtype };
        var _result = _execute.execute("Igammac", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Igammac", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns the imaginary part of a complex number.
    /// </summary>
    /// <remarks>
    /// 
    /// Given a tensor `input` of complex numbers, this operation returns a tensor of
    /// type `float` that is the imaginary part of each element in `input`. All
    /// elements in `input` must be complex numbers of the form \(a + bj\), where *a*
    /// is the real part and *b* is the imaginary part returned by this operation.
    /// 
    /// For example:
    /// 
    /// ```
    /// # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
    /// tf.imag(input) ==> [4.75, 5.75]
    /// ```
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="Tout"></param>
    /// <returns></returns>
    public static Tensor imag(Tensor input, TF_DataType Tout = TF_DataType.TF_FLOAT, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Imag", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { ["Tout"] = Tout } });
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
                return imag_eager_fallback(input, Tout: Tout, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["Tout"] = Tout;
        var _op = tf.OpDefLib._apply_op_helper("Imag", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tout", _op._get_attr_type("Tout") };
            _execute.record_gradient("Imag", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor imag_eager_fallback(Tensor input, TF_DataType Tout, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "T", input.dtype, "Tout", Tout };
        var _result = _execute.execute("Imag", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Imag", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the reciprocal of x element-wise.
    /// </summary>
    /// <remarks>
    /// 
    /// I.e., \(y = 1 / x\).
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor inv(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Inv", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return inv_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Inv", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Inv", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor inv_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Inv", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Inv", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the gradient for the inverse of `x` wrt its input.
    /// </summary>
    /// <remarks>
    /// 
    /// Specifically, `grad = -dy * y*y`, where `y = 1/x`, and `dy`
    /// is the corresponding input gradient.
    /// 
    /// </remarks>
    /// <param name="y"></param>
    /// <param name="dy"></param>
    /// <returns></returns>
    public static Tensor inv_grad(Tensor y, Tensor dy, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "InvGrad", name) { args = new object[] { y, dy }, attrs = new Dictionary<string, object>() { } });
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
                return inv_grad_eager_fallback(y, dy, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["y"] = y;
        keywords["dy"] = dy;
        var _op = tf.OpDefLib._apply_op_helper("InvGrad", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("InvGrad", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor inv_grad_eager_fallback(Tensor y, Tensor dy, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { y, dy };
        object[] _attrs = new object[] { "T", y.dtype };
        var _result = _execute.execute("InvGrad", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("InvGrad", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns which elements of x are finite.
    /// </summary>
    /// <remarks>
    /// 
    /// @compatibility(numpy)
    /// Equivalent to np.isfinite
    /// @end_compatibility
    /// 
    /// Example:
    /// 
    /// ```python
    /// x = tf.constant([5.0, 4.8, 6.8, np.inf, np.nan])
    /// tf.math.is_finite(x) ==> [True, True, True, False, False]
    /// ```
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor is_finite(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "IsFinite", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return is_finite_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("IsFinite", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("IsFinite", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor is_finite_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("IsFinite", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("IsFinite", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns which elements of x are Inf.
    /// </summary>
    /// <remarks>
    /// 
    /// @compatibility(numpy)
    /// Equivalent to np.isinf
    /// @end_compatibility
    /// 
    /// Example:
    /// 
    /// ```python
    /// x = tf.constant([5.0, np.inf, 6.8, np.inf])
    /// tf.math.is_inf(x) ==> [False, True, False, True]
    /// ```
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor is_inf(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "IsInf", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return is_inf_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("IsInf", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("IsInf", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor is_inf_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("IsInf", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("IsInf", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns which elements of x are NaN.
    /// </summary>
    /// <remarks>
    /// 
    /// @compatibility(numpy)
    /// Equivalent to np.isnan
    /// @end_compatibility
    /// 
    /// Example:
    /// 
    /// ```python
    /// x = tf.constant([5.0, np.nan, 6.8, np.nan, np.inf])
    /// tf.math.is_nan(x) ==> [False, True, False, True, False]
    /// ```
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor is_nan(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "IsNan", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return is_nan_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("IsNan", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("IsNan", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor is_nan_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("IsNan", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("IsNan", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns the truth value of (x < y) element-wise.
    /// </summary>
    /// <remarks>
    /// 
    /// *NOTE*: `Less` supports broadcasting. More about broadcasting
    /// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    /// 
    /// Example:
    /// 
    /// ```python
    /// x = tf.constant([5, 4, 6])
    /// y = tf.constant([5])
    /// tf.math.less(x, y) ==> [False, True, False]
    /// 
    /// x = tf.constant([5, 4, 6])
    /// y = tf.constant([5, 6, 7])
    /// tf.math.less(x, y) ==> [False, True, True]
    /// ```
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <returns></returns>
    public static Tensor less(Tensor x, Tensor y, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Less", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { } });
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
                return less_eager_fallback(x, y, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        var _op = tf.OpDefLib._apply_op_helper("Less", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Less", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor less_eager_fallback(Tensor x, Tensor y, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Less", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Less", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns the truth value of (x <= y) element-wise.
    /// </summary>
    /// <remarks>
    /// 
    /// *NOTE*: `LessEqual` supports broadcasting. More about broadcasting
    /// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    /// 
    /// Example:
    /// 
    /// ```python
    /// x = tf.constant([5, 4, 6])
    /// y = tf.constant([5])
    /// tf.math.less_equal(x, y) ==> [True, True, False]
    /// 
    /// x = tf.constant([5, 4, 6])
    /// y = tf.constant([5, 6, 6])
    /// tf.math.less_equal(x, y) ==> [True, True, True]
    /// ```
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <returns></returns>
    public static Tensor less_equal(Tensor x, Tensor y, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "LessEqual", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { } });
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
                return less_equal_eager_fallback(x, y, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        var _op = tf.OpDefLib._apply_op_helper("LessEqual", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("LessEqual", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor less_equal_eager_fallback(Tensor x, Tensor y, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("LessEqual", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("LessEqual", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the log of the absolute value of `Gamma(x)` element-wise.
    /// </summary>
    /// <remarks>
    /// 
    ///   For positive numbers, this function computes log((input - 1)!) for every element in the tensor.
    ///   `lgamma(5) = log((5-1)!) = log(4!) = log(24) = 3.1780539`
    /// 
    /// Example:
    /// 
    /// ```python
    /// x = tf.constant([0, 0.5, 1, 4.5, -4, -5.6])
    /// tf.math.lgamma(x) ==> [inf, 0.5723649, 0., 2.4537368, inf, -4.6477685]
    /// ```
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor lgamma(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Lgamma", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return lgamma_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Lgamma", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Lgamma", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor lgamma_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Lgamma", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Lgamma", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Generates values in an interval.
    /// </summary>
    /// <remarks>
    /// 
    /// A sequence of `num` evenly-spaced values are generated beginning at `start`.
    /// If `num > 1`, the values in the sequence increase by `stop - start / num - 1`,
    /// so that the last one is exactly `stop`.
    /// 
    /// For example:
    /// 
    /// ```
    /// tf.linspace(10.0, 12.0, 3, name="linspace") => [ 10.0  11.0  12.0]
    /// ```
    /// 
    /// </remarks>
    /// <param name="start"></param>
    /// <param name="stop"></param>
    /// <param name="num"></param>
    /// <returns></returns>
    public static Tensor lin_space(Tensor start, Tensor stop, Tensor num, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "LinSpace", name) { args = new object[] { start, stop, num }, attrs = new Dictionary<string, object>() { } });
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
                return lin_space_eager_fallback(start, stop, num, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["start"] = start;
        keywords["stop"] = stop;
        keywords["num"] = num;
        var _op = tf.OpDefLib._apply_op_helper("LinSpace", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tidx", _op._get_attr_type("Tidx") };
            _execute.record_gradient("LinSpace", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor lin_space_eager_fallback(Tensor start, Tensor stop, Tensor num, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { start, stop, num };
        object[] _attrs = new object[] { "T", start.dtype, "Tidx", num.dtype };
        var _result = _execute.execute("LinSpace", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("LinSpace", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes natural logarithm of x element-wise.
    /// </summary>
    /// <remarks>
    /// 
    /// I.e., \(y = log_e x\).
    /// 
    /// Example:
    /// 
    /// ```python
    /// x = tf.constant([0, 0.5, 1, 5])
    /// tf.math.log(x) ==> [-inf, -0.6931472,  0. ,  1.609438]
    /// ```
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor log(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Log", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return log_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Log", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Log", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor log_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Log", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Log", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes natural logarithm of (1 + x) element-wise.
    /// </summary>
    /// <remarks>
    /// 
    /// I.e., \(y = log_e (1 + x)\).
    /// 
    /// Example:
    /// 
    /// ```python
    /// x = tf.constant([0, 0.5, 1, 5])
    /// tf.math.log1p(x) ==> [0., 0.4054651, 0.6931472, 1.7917595]
    /// ```
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor log1p(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Log1p", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return log1p_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Log1p", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Log1p", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor log1p_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Log1p", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Log1p", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns the truth value of x AND y element-wise.
    /// </summary>
    /// <remarks>
    /// 
    /// *NOTE*: `LogicalAnd` supports broadcasting. More about broadcasting
    /// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <returns></returns>
    public static Tensor logical_and(Tensor x, Tensor y, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "LogicalAnd", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { } });
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
                return logical_and_eager_fallback(x, y, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        var _op = tf.OpDefLib._apply_op_helper("LogicalAnd", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { };
            _execute.record_gradient("LogicalAnd", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor logical_and_eager_fallback(Tensor x, Tensor y, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { };
        var _result = _execute.execute("LogicalAnd", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("LogicalAnd", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns the truth value of `NOT x` element-wise.
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor logical_not(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "LogicalNot", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return logical_not_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("LogicalNot", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { };
            _execute.record_gradient("LogicalNot", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor logical_not_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { };
        var _result = _execute.execute("LogicalNot", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("LogicalNot", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns the truth value of x OR y element-wise.
    /// </summary>
    /// <remarks>
    /// 
    /// *NOTE*: `LogicalOr` supports broadcasting. More about broadcasting
    /// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <returns></returns>
    public static Tensor logical_or(Tensor x, Tensor y, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "LogicalOr", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { } });
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
                return logical_or_eager_fallback(x, y, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        var _op = tf.OpDefLib._apply_op_helper("LogicalOr", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { };
            _execute.record_gradient("LogicalOr", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor logical_or_eager_fallback(Tensor x, Tensor y, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { };
        var _result = _execute.execute("LogicalOr", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("LogicalOr", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Multiply the matrix "a" by the matrix "b".
    /// </summary>
    /// <remarks>
    /// 
    /// The inputs must be two-dimensional matrices and the inner dimension of
    /// "a" (after being transposed if transpose_a is true) must match the
    /// outer dimension of "b" (after being transposed if transposed_b is
    /// true).
    /// 
    /// *Note*: The default kernel implementation for MatMul on GPUs uses
    /// cublas.
    /// 
    /// </remarks>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <param name="transpose_a">
    /// 
    /// If true, "a" is transposed before multiplication.
    /// 
    /// </param>
    /// <param name="transpose_b">
    /// 
    /// If true, "b" is transposed before multiplication.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor mat_mul(Tensor a, Tensor b, bool transpose_a = false, bool transpose_b = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "MatMul", name) { args = new object[] { a, b }, attrs = new Dictionary<string, object>() { ["transpose_a"] = transpose_a, ["transpose_b"] = transpose_b } });
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
                return mat_mul_eager_fallback(a, b, transpose_a: transpose_a, transpose_b: transpose_b, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["a"] = a;
        keywords["b"] = b;
        keywords["transpose_a"] = transpose_a;
        keywords["transpose_b"] = transpose_b;
        var _op = tf.OpDefLib._apply_op_helper("MatMul", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "transpose_a", _op._get_attr_bool("transpose_a"), "transpose_b", _op._get_attr_bool("transpose_b"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("MatMul", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor mat_mul_eager_fallback(Tensor a, Tensor b, bool transpose_a, bool transpose_b, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { a, b };
        object[] _attrs = new object[] { "transpose_a", transpose_a, "transpose_b", transpose_b, "T", a.dtype };
        var _result = _execute.execute("MatMul", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("MatMul", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the maximum of elements across dimensions of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Reduces `input` along the dimensions given in `reduction_indices`. Unless
    /// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
    /// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
    /// retained with length 1.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="reduction_indices"></param>
    /// <param name="keep_dims">
    /// 
    /// If true, retain reduced dimensions with length 1.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor max(Tensor input, Tensor reduction_indices, bool keep_dims = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Max", name) { args = new object[] { input, reduction_indices }, attrs = new Dictionary<string, object>() { ["keep_dims"] = keep_dims } });
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
                return max_eager_fallback(input, reduction_indices, keep_dims: keep_dims, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["reduction_indices"] = reduction_indices;
        keywords["keep_dims"] = keep_dims;
        var _op = tf.OpDefLib._apply_op_helper("Max", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "keep_dims", _op._get_attr_bool("keep_dims"), "T", _op._get_attr_type("T"), "Tidx", _op._get_attr_type("Tidx") };
            _execute.record_gradient("Max", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor max_eager_fallback(Tensor input, Tensor reduction_indices, bool keep_dims, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, reduction_indices };
        object[] _attrs = new object[] { "keep_dims", keep_dims, "T", input.dtype, "Tidx", reduction_indices.dtype };
        var _result = _execute.execute("Max", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Max", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns the max of x and y (i.e. x > y ? x : y) element-wise.
    /// </summary>
    /// <remarks>
    /// 
    /// *NOTE*: `Maximum` supports broadcasting. More about broadcasting
    /// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <returns></returns>
    public static Tensor maximum(Tensor x, Tensor y, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Maximum", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { } });
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
                return maximum_eager_fallback(x, y, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        var _op = tf.OpDefLib._apply_op_helper("Maximum", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Maximum", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor maximum_eager_fallback(Tensor x, Tensor y, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Maximum", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Maximum", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the mean of elements across dimensions of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Reduces `input` along the dimensions given in `reduction_indices`. Unless
    /// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
    /// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
    /// retained with length 1.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="reduction_indices"></param>
    /// <param name="keep_dims">
    /// 
    /// If true, retain reduced dimensions with length 1.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor mean(Tensor input, Tensor reduction_indices, bool keep_dims = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Mean", name) { args = new object[] { input, reduction_indices }, attrs = new Dictionary<string, object>() { ["keep_dims"] = keep_dims } });
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
                return mean_eager_fallback(input, reduction_indices, keep_dims: keep_dims, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["reduction_indices"] = reduction_indices;
        keywords["keep_dims"] = keep_dims;
        var _op = tf.OpDefLib._apply_op_helper("Mean", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "keep_dims", _op._get_attr_bool("keep_dims"), "T", _op._get_attr_type("T"), "Tidx", _op._get_attr_type("Tidx") };
            _execute.record_gradient("Mean", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor mean_eager_fallback(Tensor input, Tensor reduction_indices, bool keep_dims, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, reduction_indices };
        object[] _attrs = new object[] { "keep_dims", keep_dims, "T", input.dtype, "Tidx", reduction_indices.dtype };
        var _result = _execute.execute("Mean", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Mean", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the minimum of elements across dimensions of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Reduces `input` along the dimensions given in `reduction_indices`. Unless
    /// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
    /// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
    /// retained with length 1.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="reduction_indices"></param>
    /// <param name="keep_dims">
    /// 
    /// If true, retain reduced dimensions with length 1.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor min(Tensor input, Tensor reduction_indices, bool keep_dims = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Min", name) { args = new object[] { input, reduction_indices }, attrs = new Dictionary<string, object>() { ["keep_dims"] = keep_dims } });
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
                return min_eager_fallback(input, reduction_indices, keep_dims: keep_dims, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["reduction_indices"] = reduction_indices;
        keywords["keep_dims"] = keep_dims;
        var _op = tf.OpDefLib._apply_op_helper("Min", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "keep_dims", _op._get_attr_bool("keep_dims"), "T", _op._get_attr_type("T"), "Tidx", _op._get_attr_type("Tidx") };
            _execute.record_gradient("Min", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor min_eager_fallback(Tensor input, Tensor reduction_indices, bool keep_dims, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, reduction_indices };
        object[] _attrs = new object[] { "keep_dims", keep_dims, "T", input.dtype, "Tidx", reduction_indices.dtype };
        var _result = _execute.execute("Min", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Min", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns the min of x and y (i.e. x < y ? x : y) element-wise.
    /// </summary>
    /// <remarks>
    /// 
    /// *NOTE*: `Minimum` supports broadcasting. More about broadcasting
    /// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <returns></returns>
    public static Tensor minimum(Tensor x, Tensor y, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Minimum", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { } });
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
                return minimum_eager_fallback(x, y, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        var _op = tf.OpDefLib._apply_op_helper("Minimum", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Minimum", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor minimum_eager_fallback(Tensor x, Tensor y, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Minimum", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Minimum", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns element-wise remainder of division. This emulates C semantics in that
    /// </summary>
    /// <remarks>
    /// 
    /// the result here is consistent with a truncating divide. E.g.
    /// `tf.truncatediv(x, y) * y + truncate_mod(x, y) = x`.
    /// 
    /// *NOTE*: `Mod` supports broadcasting. More about broadcasting
    /// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <returns></returns>
    public static Tensor mod(Tensor x, Tensor y, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Mod", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { } });
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
                return mod_eager_fallback(x, y, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        var _op = tf.OpDefLib._apply_op_helper("Mod", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Mod", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor mod_eager_fallback(Tensor x, Tensor y, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Mod", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Mod", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns x * y element-wise.
    /// </summary>
    /// <remarks>
    /// 
    /// *NOTE*: `Mul` supports broadcasting. More about broadcasting
    /// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <returns></returns>
    public static Tensor mul(Tensor x, Tensor y, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Mul", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { } });
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
                return mul_eager_fallback(x, y, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        var _op = tf.OpDefLib._apply_op_helper("Mul", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Mul", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor mul_eager_fallback(Tensor x, Tensor y, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Mul", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Mul", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns x * y element-wise. Returns zero if y is zero, even if x if infinite or NaN.
    /// </summary>
    /// <remarks>
    /// 
    /// *NOTE*: `MulNoNan` supports broadcasting. More about broadcasting
    /// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <returns></returns>
    public static Tensor mul_no_nan(Tensor x, Tensor y, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "MulNoNan", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { } });
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
                return mul_no_nan_eager_fallback(x, y, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        var _op = tf.OpDefLib._apply_op_helper("MulNoNan", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("MulNoNan", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor mul_no_nan_eager_fallback(Tensor x, Tensor y, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("MulNoNan", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("MulNoNan", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor ndtri(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Ndtri", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return ndtri_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Ndtri", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Ndtri", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor ndtri_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Ndtri", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Ndtri", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes numerical negative value element-wise.
    /// </summary>
    /// <remarks>
    /// 
    /// I.e., \(y = -x\).
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor neg(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Neg", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return neg_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Neg", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Neg", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor neg_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Neg", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Neg", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns the next representable value of `x1` in the direction of `x2`, element-wise.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation returns the same result as the C++ std::nextafter function.
    /// 
    /// It can also return a subnormal number.
    /// 
    /// @compatibility(cpp)
    /// Equivalent to C++ std::nextafter function.
    /// @end_compatibility
    /// 
    /// </remarks>
    /// <param name="x1"></param>
    /// <param name="x2"></param>
    /// <returns></returns>
    public static Tensor next_after(Tensor x1, Tensor x2, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "NextAfter", name) { args = new object[] { x1, x2 }, attrs = new Dictionary<string, object>() { } });
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
                return next_after_eager_fallback(x1, x2, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x1"] = x1;
        keywords["x2"] = x2;
        var _op = tf.OpDefLib._apply_op_helper("NextAfter", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("NextAfter", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor next_after_eager_fallback(Tensor x1, Tensor x2, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x1, x2 };
        object[] _attrs = new object[] { "T", x1.dtype };
        var _result = _execute.execute("NextAfter", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("NextAfter", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns the truth value of (x != y) element-wise.
    /// </summary>
    /// <remarks>
    /// 
    /// *NOTE*: `NotEqual` supports broadcasting. More about broadcasting
    /// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <param name="incompatible_shape_error"></param>
    /// <returns></returns>
    public static Tensor not_equal(Tensor x, Tensor y, bool incompatible_shape_error = true, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "NotEqual", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { ["incompatible_shape_error"] = incompatible_shape_error } });
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
                return not_equal_eager_fallback(x, y, incompatible_shape_error: incompatible_shape_error, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        keywords["incompatible_shape_error"] = incompatible_shape_error;
        var _op = tf.OpDefLib._apply_op_helper("NotEqual", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "incompatible_shape_error", _op._get_attr_bool("incompatible_shape_error") };
            _execute.record_gradient("NotEqual", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor not_equal_eager_fallback(Tensor x, Tensor y, bool incompatible_shape_error, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { "T", x.dtype, "incompatible_shape_error", incompatible_shape_error };
        var _result = _execute.execute("NotEqual", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("NotEqual", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Compute the polygamma function \\(\psi^{(n)}(x)\\).
    /// </summary>
    /// <remarks>
    /// 
    /// The polygamma function is defined as:
    /// 
    /// 
    /// \(psi^{(a)}(x) = rac{d^a}{dx^a} psi(x)\)
    /// 
    /// where \(psi(x)\) is the digamma function.
    /// The polygamma function is defined only for non-negative integer orders \a\.
    /// 
    /// </remarks>
    /// <param name="a"></param>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor polygamma(Tensor a, Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Polygamma", name) { args = new object[] { a, x }, attrs = new Dictionary<string, object>() { } });
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
                return polygamma_eager_fallback(a, x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["a"] = a;
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Polygamma", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Polygamma", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor polygamma_eager_fallback(Tensor a, Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { a, x };
        object[] _attrs = new object[] { "T", a.dtype };
        var _result = _execute.execute("Polygamma", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Polygamma", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the power of one value to another.
    /// </summary>
    /// <remarks>
    /// 
    /// Given a tensor `x` and a tensor `y`, this operation computes \(x^y\) for
    /// corresponding elements in `x` and `y`. For example:
    /// 
    /// ```
    /// # tensor 'x' is [[2, 2]], [3, 3]]
    /// # tensor 'y' is [[8, 16], [2, 3]]
    /// tf.pow(x, y) ==> [[256, 65536], [9, 27]]
    /// ```
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <returns></returns>
    public static Tensor pow(Tensor x, Tensor y, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Pow", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { } });
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
                return pow_eager_fallback(x, y, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        var _op = tf.OpDefLib._apply_op_helper("Pow", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Pow", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor pow_eager_fallback(Tensor x, Tensor y, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Pow", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Pow", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the product of elements across dimensions of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Reduces `input` along the dimensions given in `reduction_indices`. Unless
    /// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
    /// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
    /// retained with length 1.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="reduction_indices"></param>
    /// <param name="keep_dims">
    /// 
    /// If true, retain reduced dimensions with length 1.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor prod(Tensor input, Tensor reduction_indices, bool keep_dims = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Prod", name) { args = new object[] { input, reduction_indices }, attrs = new Dictionary<string, object>() { ["keep_dims"] = keep_dims } });
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
                return prod_eager_fallback(input, reduction_indices, keep_dims: keep_dims, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["reduction_indices"] = reduction_indices;
        keywords["keep_dims"] = keep_dims;
        var _op = tf.OpDefLib._apply_op_helper("Prod", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "keep_dims", _op._get_attr_bool("keep_dims"), "T", _op._get_attr_type("T"), "Tidx", _op._get_attr_type("Tidx") };
            _execute.record_gradient("Prod", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor prod_eager_fallback(Tensor input, Tensor reduction_indices, bool keep_dims, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, reduction_indices };
        object[] _attrs = new object[] { "keep_dims", keep_dims, "T", input.dtype, "Tidx", reduction_indices.dtype };
        var _result = _execute.execute("Prod", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Prod", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Convert the quantized 'input' tensor into a lower-precision 'output', using the
    /// </summary>
    /// <remarks>
    /// 
    /// actual distribution of the values to maximize the usage of the lower bit depth
    /// and adjusting the output min and max ranges accordingly.
    /// 
    /// [input_min, input_max] are scalar floats that specify the range for the float
    /// interpretation of the 'input' data. For example, if input_min is -1.0f and
    /// input_max is 1.0f, and we are dealing with quint16 quantized data, then a 0
    /// value in the 16-bit data should be interpreted as -1.0f, and a 65535 means 1.0f.
    /// 
    /// This operator tries to squeeze as much precision as possible into an output with
    /// a lower bit depth by calculating the actual min and max values found in the
    /// data. For example, maybe that quint16 input has no values lower than 16,384 and
    /// none higher than 49,152. That means only half the range is actually needed, all
    /// the float interpretations are between -0.5f and 0.5f, so if we want to compress
    /// the data into a quint8 output, we can use that range rather than the theoretical
    /// -1.0f to 1.0f that is suggested by the input min and max.
    /// 
    /// In practice, this is most useful for taking output from operations like
    /// QuantizedMatMul that can produce higher bit-depth outputs than their inputs and
    /// may have large potential output ranges, but in practice have a distribution of
    /// input values that only uses a small fraction of the possible range. By feeding
    /// that output into this operator, we can reduce it from 32 bits down to 8 with
    /// minimal loss of accuracy.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="input_min"></param>
    /// <param name="input_max"></param>
    /// <param name="out_type">
    /// 
    /// The type of the output. Should be a lower bit depth than Tinput.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] quantize_down_and_shrink_range(Tensor input, Tensor input_min, Tensor input_max, TF_DataType out_type, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizeDownAndShrinkRange", name) { args = new object[] { input, input_min, input_max }, attrs = new Dictionary<string, object>() { ["out_type"] = out_type } });
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
                return quantize_down_and_shrink_range_eager_fallback(input, input_min, input_max, out_type: out_type, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["input_min"] = input_min;
        keywords["input_max"] = input_max;
        keywords["out_type"] = out_type;
        var _op = tf.OpDefLib._apply_op_helper("QuantizeDownAndShrinkRange", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tinput", _op._get_attr_type("Tinput"), "out_type", _op._get_attr_type("out_type") };
            _execute.record_gradient("QuantizeDownAndShrinkRange", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantize_down_and_shrink_range_eager_fallback(Tensor input, Tensor input_min, Tensor input_max, TF_DataType out_type, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, input_min, input_max };
        object[] _attrs = new object[] { "Tinput", input.dtype, "out_type", out_type };
        var _result = _execute.execute("QuantizeDownAndShrinkRange", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizeDownAndShrinkRange", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Returns x + y element-wise, working on quantized buffers.
    /// </summary>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <param name="min_x"></param>
    /// <param name="max_x"></param>
    /// <param name="min_y"></param>
    /// <param name="max_y"></param>
    /// <param name="Toutput"></param>
    /// <returns></returns>
    public static Tensor[] quantized_add(Tensor x, Tensor y, Tensor min_x, Tensor max_x, Tensor min_y, Tensor max_y, TF_DataType Toutput = TF_DataType.TF_QINT32, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedAdd", name) { args = new object[] { x, y, min_x, max_x, min_y, max_y }, attrs = new Dictionary<string, object>() { ["Toutput"] = Toutput } });
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
                return quantized_add_eager_fallback(x, y, min_x, max_x, min_y, max_y, Toutput: Toutput, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        keywords["min_x"] = min_x;
        keywords["max_x"] = max_x;
        keywords["min_y"] = min_y;
        keywords["max_y"] = max_y;
        keywords["Toutput"] = Toutput;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedAdd", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T1", _op._get_attr_type("T1"), "T2", _op._get_attr_type("T2"), "Toutput", _op._get_attr_type("Toutput") };
            _execute.record_gradient("QuantizedAdd", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_add_eager_fallback(Tensor x, Tensor y, Tensor min_x, Tensor max_x, Tensor min_y, Tensor max_y, TF_DataType Toutput, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y, min_x, max_x, min_y, max_y };
        object[] _attrs = new object[] { "T1", x.dtype, "T2", y.dtype, "Toutput", Toutput };
        var _result = _execute.execute("QuantizedAdd", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedAdd", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Perform a quantized matrix multiplication of  `a` by the matrix `b`.
    /// </summary>
    /// <remarks>
    /// 
    /// The inputs must be two-dimensional matrices and the inner dimension of
    /// `a` (after being transposed if `transpose_a` is non-zero) must match the
    /// outer dimension of `b` (after being transposed if `transposed_b` is
    /// non-zero).
    /// 
    /// </remarks>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <param name="min_a"></param>
    /// <param name="max_a"></param>
    /// <param name="min_b"></param>
    /// <param name="max_b"></param>
    /// <param name="Toutput"></param>
    /// <param name="transpose_a">
    /// 
    /// If true, `a` is transposed before multiplication.
    /// 
    /// </param>
    /// <param name="transpose_b">
    /// 
    /// If true, `b` is transposed before multiplication.
    /// 
    /// </param>
    /// <param name="Tactivation">
    /// 
    /// The type of output produced by activation function
    /// following this operation.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] quantized_mat_mul(Tensor a, Tensor b, Tensor min_a, Tensor max_a, Tensor min_b, Tensor max_b, TF_DataType Toutput = TF_DataType.TF_QINT32, bool transpose_a = false, bool transpose_b = false, TF_DataType Tactivation = TF_DataType.TF_QUINT8, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedMatMul", name) { args = new object[] { a, b, min_a, max_a, min_b, max_b }, attrs = new Dictionary<string, object>() { ["Toutput"] = Toutput, ["transpose_a"] = transpose_a, ["transpose_b"] = transpose_b, ["Tactivation"] = Tactivation } });
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
                return quantized_mat_mul_eager_fallback(a, b, min_a, max_a, min_b, max_b, Toutput: Toutput, transpose_a: transpose_a, transpose_b: transpose_b, Tactivation: Tactivation, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["a"] = a;
        keywords["b"] = b;
        keywords["min_a"] = min_a;
        keywords["max_a"] = max_a;
        keywords["min_b"] = min_b;
        keywords["max_b"] = max_b;
        keywords["Toutput"] = Toutput;
        keywords["transpose_a"] = transpose_a;
        keywords["transpose_b"] = transpose_b;
        keywords["Tactivation"] = Tactivation;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedMatMul", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T1", _op._get_attr_type("T1"), "T2", _op._get_attr_type("T2"), "Toutput", _op._get_attr_type("Toutput"), "transpose_a", _op._get_attr_bool("transpose_a"), "transpose_b", _op._get_attr_bool("transpose_b"), "Tactivation", _op._get_attr_type("Tactivation") };
            _execute.record_gradient("QuantizedMatMul", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_mat_mul_eager_fallback(Tensor a, Tensor b, Tensor min_a, Tensor max_a, Tensor min_b, Tensor max_b, TF_DataType Toutput, bool transpose_a, bool transpose_b, TF_DataType Tactivation, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { a, b, min_a, max_a, min_b, max_b };
        object[] _attrs = new object[] { "T1", a.dtype, "T2", b.dtype, "Toutput", Toutput, "transpose_a", transpose_a, "transpose_b", transpose_b, "Tactivation", Tactivation };
        var _result = _execute.execute("QuantizedMatMul", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedMatMul", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Returns x * y element-wise, working on quantized buffers.
    /// </summary>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <param name="min_x"></param>
    /// <param name="max_x"></param>
    /// <param name="min_y"></param>
    /// <param name="max_y"></param>
    /// <param name="Toutput"></param>
    /// <returns></returns>
    public static Tensor[] quantized_mul(Tensor x, Tensor y, Tensor min_x, Tensor max_x, Tensor min_y, Tensor max_y, TF_DataType Toutput = TF_DataType.TF_QINT32, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedMul", name) { args = new object[] { x, y, min_x, max_x, min_y, max_y }, attrs = new Dictionary<string, object>() { ["Toutput"] = Toutput } });
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
                return quantized_mul_eager_fallback(x, y, min_x, max_x, min_y, max_y, Toutput: Toutput, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        keywords["min_x"] = min_x;
        keywords["max_x"] = max_x;
        keywords["min_y"] = min_y;
        keywords["max_y"] = max_y;
        keywords["Toutput"] = Toutput;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedMul", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T1", _op._get_attr_type("T1"), "T2", _op._get_attr_type("T2"), "Toutput", _op._get_attr_type("Toutput") };
            _execute.record_gradient("QuantizedMul", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_mul_eager_fallback(Tensor x, Tensor y, Tensor min_x, Tensor max_x, Tensor min_y, Tensor max_y, TF_DataType Toutput, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y, min_x, max_x, min_y, max_y };
        object[] _attrs = new object[] { "T1", x.dtype, "T2", y.dtype, "Toutput", Toutput };
        var _result = _execute.execute("QuantizedMul", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedMul", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Counts the number of occurrences of each value in an integer array.
    /// </summary>
    /// <remarks>
    /// 
    /// Outputs a vector with length `size` and the same dtype as `weights`. If
    /// `weights` are empty, then index `i` stores the number of times the value `i` is
    /// counted in `arr`. If `weights` are non-empty, then index `i` stores the sum of
    /// the value in `weights` at each index where the corresponding value in `arr` is
    /// `i`.
    /// 
    /// Values in `arr` outside of the range [0, size) are ignored.
    /// 
    /// </remarks>
    /// <param name="splits"></param>
    /// <param name="values"></param>
    /// <param name="size"></param>
    /// <param name="weights"></param>
    /// <param name="binary_output">
    /// 
    /// bool; Whether the kernel should count the appearance or number of occurrences.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor ragged_bincount(Tensor splits, Tensor values, Tensor size, Tensor weights, bool binary_output = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "RaggedBincount", name) { args = new object[] { splits, values, size, weights }, attrs = new Dictionary<string, object>() { ["binary_output"] = binary_output } });
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
                return ragged_bincount_eager_fallback(splits, values, size, weights, binary_output: binary_output, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["splits"] = splits;
        keywords["values"] = values;
        keywords["size"] = size;
        keywords["weights"] = weights;
        keywords["binary_output"] = binary_output;
        var _op = tf.OpDefLib._apply_op_helper("RaggedBincount", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tidx", _op._get_attr_type("Tidx"), "T", _op._get_attr_type("T"), "binary_output", _op._get_attr_bool("binary_output") };
            _execute.record_gradient("RaggedBincount", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor ragged_bincount_eager_fallback(Tensor splits, Tensor values, Tensor size, Tensor weights, bool binary_output, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { splits, values, size, weights };
        object[] _attrs = new object[] { "Tidx", values.dtype, "T", weights.dtype, "binary_output", binary_output };
        var _result = _execute.execute("RaggedBincount", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("RaggedBincount", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Creates a sequence of numbers.
    /// </summary>
    /// <remarks>
    /// 
    /// This operation creates a sequence of numbers that begins at `start` and
    /// extends by increments of `delta` up to but not including `limit`.
    /// 
    /// For example:
    /// 
    /// ```
    /// # 'start' is 3
    /// # 'limit' is 18
    /// # 'delta' is 3
    /// tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]
    /// ```
    /// 
    /// </remarks>
    /// <param name="start"></param>
    /// <param name="limit"></param>
    /// <param name="delta"></param>
    /// <returns></returns>
    public static Tensor range(Tensor start, Tensor limit, Tensor delta, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Range", name) { args = new object[] { start, limit, delta }, attrs = new Dictionary<string, object>() { } });
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
                return range_eager_fallback(start, limit, delta, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["start"] = start;
        keywords["limit"] = limit;
        keywords["delta"] = delta;
        var _op = tf.OpDefLib._apply_op_helper("Range", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tidx", _op._get_attr_type("Tidx") };
            _execute.record_gradient("Range", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor range_eager_fallback(Tensor start, Tensor limit, Tensor delta, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { start, limit, delta };
        object[] _attrs = new object[] { "Tidx", start.dtype };
        var _result = _execute.execute("Range", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Range", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns the real part of a complex number.
    /// </summary>
    /// <remarks>
    /// 
    /// Given a tensor `input` of complex numbers, this operation returns a tensor of
    /// type `float` that is the real part of each element in `input`. All elements in
    /// `input` must be complex numbers of the form \(a + bj\), where *a* is the real
    ///  part returned by this operation and *b* is the imaginary part.
    /// 
    /// For example:
    /// 
    /// ```
    /// # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
    /// tf.real(input) ==> [-2.25, 3.25]
    /// ```
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="Tout"></param>
    /// <returns></returns>
    public static Tensor real(Tensor input, TF_DataType Tout = TF_DataType.TF_FLOAT, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Real", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { ["Tout"] = Tout } });
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
                return real_eager_fallback(input, Tout: Tout, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["Tout"] = Tout;
        var _op = tf.OpDefLib._apply_op_helper("Real", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tout", _op._get_attr_type("Tout") };
            _execute.record_gradient("Real", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor real_eager_fallback(Tensor input, TF_DataType Tout, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "T", input.dtype, "Tout", Tout };
        var _result = _execute.execute("Real", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Real", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns x / y element-wise for real types.
    /// </summary>
    /// <remarks>
    /// 
    /// If `x` and `y` are reals, this will return the floating-point division.
    /// 
    /// *NOTE*: `Div` supports broadcasting. More about broadcasting
    /// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <returns></returns>
    public static Tensor real_div(Tensor x, Tensor y, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "RealDiv", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { } });
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
                return real_div_eager_fallback(x, y, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        var _op = tf.OpDefLib._apply_op_helper("RealDiv", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("RealDiv", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor real_div_eager_fallback(Tensor x, Tensor y, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("RealDiv", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("RealDiv", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the reciprocal of x element-wise.
    /// </summary>
    /// <remarks>
    /// 
    /// I.e., \(y = 1 / x\).
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor reciprocal(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Reciprocal", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return reciprocal_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Reciprocal", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Reciprocal", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor reciprocal_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Reciprocal", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Reciprocal", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the gradient for the inverse of `x` wrt its input.
    /// </summary>
    /// <remarks>
    /// 
    /// Specifically, `grad = -dy * y*y`, where `y = 1/x`, and `dy`
    /// is the corresponding input gradient.
    /// 
    /// </remarks>
    /// <param name="y"></param>
    /// <param name="dy"></param>
    /// <returns></returns>
    public static Tensor reciprocal_grad(Tensor y, Tensor dy, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ReciprocalGrad", name) { args = new object[] { y, dy }, attrs = new Dictionary<string, object>() { } });
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
                return reciprocal_grad_eager_fallback(y, dy, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["y"] = y;
        keywords["dy"] = dy;
        var _op = tf.OpDefLib._apply_op_helper("ReciprocalGrad", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("ReciprocalGrad", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor reciprocal_grad_eager_fallback(Tensor y, Tensor dy, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { y, dy };
        object[] _attrs = new object[] { "T", y.dtype };
        var _result = _execute.execute("ReciprocalGrad", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ReciprocalGrad", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes a range that covers the actual values present in a quantized tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Given a quantized tensor described by `(input, input_min, input_max)`, outputs a
    /// range that covers the actual values present in that tensor. This op is typically
    /// used to produce the `requested_output_min` and `requested_output_max` for
    /// `Requantize`.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="input_min"></param>
    /// <param name="input_max"></param>
    /// <returns></returns>
    public static Tensor[] requantization_range(Tensor input, Tensor input_min, Tensor input_max, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "RequantizationRange", name) { args = new object[] { input, input_min, input_max }, attrs = new Dictionary<string, object>() { } });
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
                return requantization_range_eager_fallback(input, input_min, input_max, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["input_min"] = input_min;
        keywords["input_max"] = input_max;
        var _op = tf.OpDefLib._apply_op_helper("RequantizationRange", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tinput", _op._get_attr_type("Tinput") };
            _execute.record_gradient("RequantizationRange", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] requantization_range_eager_fallback(Tensor input, Tensor input_min, Tensor input_max, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, input_min, input_max };
        object[] _attrs = new object[] { "Tinput", input.dtype };
        var _result = _execute.execute("RequantizationRange", 2, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("RequantizationRange", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Computes requantization range per channel.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="input_min"></param>
    /// <param name="input_max"></param>
    /// <param name="clip_value_max">
    /// 
    /// The maximum value of the output that needs to be clipped.
    /// Example: set this to 6 for Relu6.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] requantization_range_per_channel(Tensor input, Tensor input_min, Tensor input_max, float clip_value_max, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "RequantizationRangePerChannel", name) { args = new object[] { input, input_min, input_max }, attrs = new Dictionary<string, object>() { ["clip_value_max"] = clip_value_max } });
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
                return requantization_range_per_channel_eager_fallback(input, input_min, input_max, clip_value_max: clip_value_max, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["input_min"] = input_min;
        keywords["input_max"] = input_max;
        keywords["clip_value_max"] = clip_value_max;
        var _op = tf.OpDefLib._apply_op_helper("RequantizationRangePerChannel", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "clip_value_max", _op.get_attr("clip_value_max") };
            _execute.record_gradient("RequantizationRangePerChannel", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] requantization_range_per_channel_eager_fallback(Tensor input, Tensor input_min, Tensor input_max, float clip_value_max, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, input_min, input_max };
        object[] _attrs = new object[] { "T", input.dtype, "clip_value_max", clip_value_max };
        var _result = _execute.execute("RequantizationRangePerChannel", 2, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("RequantizationRangePerChannel", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Converts the quantized `input` tensor into a lower-precision `output`.
    /// </summary>
    /// <remarks>
    /// 
    /// Converts the quantized `input` tensor into a lower-precision `output`, using the
    /// output range specified with `requested_output_min` and `requested_output_max`.
    /// 
    /// `[input_min, input_max]` are scalar floats that specify the range for the float
    /// interpretation of the `input` data. For example, if `input_min` is -1.0f and
    /// `input_max` is 1.0f, and we are dealing with `quint16` quantized data, then a 0
    /// value in the 16-bit data should be interpreted as -1.0f, and a 65535 means 1.0f.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="input_min"></param>
    /// <param name="input_max"></param>
    /// <param name="requested_output_min"></param>
    /// <param name="requested_output_max"></param>
    /// <param name="out_type">
    /// 
    /// The type of the output. Should be a lower bit depth than Tinput.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] requantize(Tensor input, Tensor input_min, Tensor input_max, Tensor requested_output_min, Tensor requested_output_max, TF_DataType out_type, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Requantize", name) { args = new object[] { input, input_min, input_max, requested_output_min, requested_output_max }, attrs = new Dictionary<string, object>() { ["out_type"] = out_type } });
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
                return requantize_eager_fallback(input, input_min, input_max, requested_output_min, requested_output_max, out_type: out_type, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["input_min"] = input_min;
        keywords["input_max"] = input_max;
        keywords["requested_output_min"] = requested_output_min;
        keywords["requested_output_max"] = requested_output_max;
        keywords["out_type"] = out_type;
        var _op = tf.OpDefLib._apply_op_helper("Requantize", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tinput", _op._get_attr_type("Tinput"), "out_type", _op._get_attr_type("out_type") };
            _execute.record_gradient("Requantize", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] requantize_eager_fallback(Tensor input, Tensor input_min, Tensor input_max, Tensor requested_output_min, Tensor requested_output_max, TF_DataType out_type, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, input_min, input_max, requested_output_min, requested_output_max };
        object[] _attrs = new object[] { "Tinput", input.dtype, "out_type", out_type };
        var _result = _execute.execute("Requantize", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Requantize", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Requantizes input with min and max values known per channel.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="input_min"></param>
    /// <param name="input_max"></param>
    /// <param name="requested_output_min"></param>
    /// <param name="requested_output_max"></param>
    /// <param name="out_type">
    /// 
    /// The quantized type of output tensor that needs to be converted.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] requantize_per_channel(Tensor input, Tensor input_min, Tensor input_max, Tensor requested_output_min, Tensor requested_output_max, TF_DataType out_type = TF_DataType.TF_QUINT8, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "RequantizePerChannel", name) { args = new object[] { input, input_min, input_max, requested_output_min, requested_output_max }, attrs = new Dictionary<string, object>() { ["out_type"] = out_type } });
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
                return requantize_per_channel_eager_fallback(input, input_min, input_max, requested_output_min, requested_output_max, out_type: out_type, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["input_min"] = input_min;
        keywords["input_max"] = input_max;
        keywords["requested_output_min"] = requested_output_min;
        keywords["requested_output_max"] = requested_output_max;
        keywords["out_type"] = out_type;
        var _op = tf.OpDefLib._apply_op_helper("RequantizePerChannel", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "out_type", _op._get_attr_type("out_type") };
            _execute.record_gradient("RequantizePerChannel", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] requantize_per_channel_eager_fallback(Tensor input, Tensor input_min, Tensor input_max, Tensor requested_output_min, Tensor requested_output_max, TF_DataType out_type, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, input_min, input_max, requested_output_min, requested_output_max };
        object[] _attrs = new object[] { "T", input.dtype, "out_type", out_type };
        var _result = _execute.execute("RequantizePerChannel", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("RequantizePerChannel", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Returns element-wise integer closest to x.
    /// </summary>
    /// <remarks>
    /// 
    /// If the result is midway between two representable values,
    /// the even representable is chosen.
    /// For example:
    /// 
    /// ```
    /// rint(-1.5) ==> -2.0
    /// rint(0.5000001) ==> 1.0
    /// rint([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]) ==> [-2., -2., -0., 0., 2., 2., 2.]
    /// ```
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor rint(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Rint", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return rint_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Rint", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Rint", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor rint_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Rint", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Rint", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Rounds the values of a tensor to the nearest integer, element-wise.
    /// </summary>
    /// <remarks>
    /// 
    /// Rounds half to even.  Also known as bankers rounding. If you want to round
    /// according to the current system rounding mode use std::cint.
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor round(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Round", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return round_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Round", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Round", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor round_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Round", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Round", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes reciprocal of square root of x element-wise.
    /// </summary>
    /// <remarks>
    /// 
    /// I.e., \(y = 1 / sqrt{x}\).
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor rsqrt(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Rsqrt", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return rsqrt_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Rsqrt", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Rsqrt", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor rsqrt_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Rsqrt", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Rsqrt", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the gradient for the rsqrt of `x` wrt its input.
    /// </summary>
    /// <remarks>
    /// 
    /// Specifically, `grad = dy * -0.5 * y^3`, where `y = rsqrt(x)`, and `dy`
    /// is the corresponding input gradient.
    /// 
    /// </remarks>
    /// <param name="y"></param>
    /// <param name="dy"></param>
    /// <returns></returns>
    public static Tensor rsqrt_grad(Tensor y, Tensor dy, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "RsqrtGrad", name) { args = new object[] { y, dy }, attrs = new Dictionary<string, object>() { } });
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
                return rsqrt_grad_eager_fallback(y, dy, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["y"] = y;
        keywords["dy"] = dy;
        var _op = tf.OpDefLib._apply_op_helper("RsqrtGrad", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("RsqrtGrad", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor rsqrt_grad_eager_fallback(Tensor y, Tensor dy, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { y, dy };
        object[] _attrs = new object[] { "T", y.dtype };
        var _result = _execute.execute("RsqrtGrad", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("RsqrtGrad", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the maximum along segments of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Read
    /// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
    /// for an explanation of segments.
    /// 
    /// Computes a tensor such that
    /// \(output_i = max_j(data_j)\) where `max` is over `j` such
    /// that `segment_ids[j] == i`.
    /// 
    /// If the max is empty for a given segment ID `i`, `output[i] = 0`.
    /// 
    /// Caution: On CPU, values in `segment_ids` are always validated to be sorted,
    /// and an error is thrown for indices that are not increasing. On GPU, this
    /// does not throw an error for unsorted indices. On GPU, out-of-order indices
    /// result in safe but unspecified behavior, which may include treating
    /// out-of-order indices as the same as a smaller following index.
    /// 
    /// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
    /// <img style="width:100%" src="https://www.tensorflow.org/images/SegmentMax.png" alt>
    /// </div>
    /// 
    /// For example:
    /// 
    /// >>> c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
    /// >>> tf.math.segment_max(c, tf.constant([0, 0, 1])).numpy()
    /// array([[4, 3, 3, 4],
    ///        [5, 6, 7, 8]], dtype=int32)
    /// 
    /// 
    /// </remarks>
    /// <param name="data"></param>
    /// <param name="segment_ids"></param>
    /// <returns></returns>
    public static Tensor segment_max(Tensor data, Tensor segment_ids, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SegmentMax", name) { args = new object[] { data, segment_ids }, attrs = new Dictionary<string, object>() { } });
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
                return segment_max_eager_fallback(data, segment_ids, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["data"] = data;
        keywords["segment_ids"] = segment_ids;
        var _op = tf.OpDefLib._apply_op_helper("SegmentMax", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tindices", _op._get_attr_type("Tindices") };
            _execute.record_gradient("SegmentMax", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor segment_max_eager_fallback(Tensor data, Tensor segment_ids, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { data, segment_ids };
        object[] _attrs = new object[] { "T", data.dtype, "Tindices", segment_ids.dtype };
        var _result = _execute.execute("SegmentMax", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SegmentMax", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the mean along segments of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Read
    /// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
    /// for an explanation of segments.
    /// 
    /// Computes a tensor such that
    /// \(output_i = rac{sum_j data_j}{N}\) where `mean` is
    /// over `j` such that `segment_ids[j] == i` and `N` is the total number of
    /// values summed.
    /// 
    /// If the mean is empty for a given segment ID `i`, `output[i] = 0`.
    /// 
    /// Caution: On CPU, values in `segment_ids` are always validated to be sorted,
    /// and an error is thrown for indices that are not increasing. On GPU, this
    /// does not throw an error for unsorted indices. On GPU, out-of-order indices
    /// result in safe but unspecified behavior, which may include treating
    /// out-of-order indices as a smaller following index when computing the numerator
    /// of the mean.
    /// 
    /// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
    /// <img style="width:100%" src="https://www.tensorflow.org/images/SegmentMean.png" alt>
    /// </div>
    /// 
    /// For example:
    /// 
    /// >>> c = tf.constant([[1.0,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
    /// >>> tf.math.segment_mean(c, tf.constant([0, 0, 1])).numpy()
    /// array([[2.5, 2.5, 2.5, 2.5],
    ///        [5., 6., 7., 8.]], dtype=float32)
    /// 
    /// 
    /// </remarks>
    /// <param name="data"></param>
    /// <param name="segment_ids"></param>
    /// <returns></returns>
    public static Tensor segment_mean(Tensor data, Tensor segment_ids, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SegmentMean", name) { args = new object[] { data, segment_ids }, attrs = new Dictionary<string, object>() { } });
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
                return segment_mean_eager_fallback(data, segment_ids, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["data"] = data;
        keywords["segment_ids"] = segment_ids;
        var _op = tf.OpDefLib._apply_op_helper("SegmentMean", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tindices", _op._get_attr_type("Tindices") };
            _execute.record_gradient("SegmentMean", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor segment_mean_eager_fallback(Tensor data, Tensor segment_ids, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { data, segment_ids };
        object[] _attrs = new object[] { "T", data.dtype, "Tindices", segment_ids.dtype };
        var _result = _execute.execute("SegmentMean", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SegmentMean", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the minimum along segments of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Read
    /// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
    /// for an explanation of segments.
    /// 
    /// Computes a tensor such that
    /// \(output_i = min_j(data_j)\) where `min` is over `j` such
    /// that `segment_ids[j] == i`.
    /// 
    /// If the min is empty for a given segment ID `i`, `output[i] = 0`.
    /// 
    /// Caution: On CPU, values in `segment_ids` are always validated to be sorted,
    /// and an error is thrown for indices that are not increasing. On GPU, this
    /// does not throw an error for unsorted indices. On GPU, out-of-order indices
    /// result in safe but unspecified behavior, which may include treating
    /// out-of-order indices as the same as a smaller following index.
    /// 
    /// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
    /// <img style="width:100%" src="https://www.tensorflow.org/images/SegmentMin.png" alt>
    /// </div>
    /// 
    /// For example:
    /// 
    /// >>> c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
    /// >>> tf.math.segment_min(c, tf.constant([0, 0, 1])).numpy()
    /// array([[1, 2, 2, 1],
    ///        [5, 6, 7, 8]], dtype=int32)
    /// 
    /// 
    /// </remarks>
    /// <param name="data"></param>
    /// <param name="segment_ids"></param>
    /// <returns></returns>
    public static Tensor segment_min(Tensor data, Tensor segment_ids, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SegmentMin", name) { args = new object[] { data, segment_ids }, attrs = new Dictionary<string, object>() { } });
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
                return segment_min_eager_fallback(data, segment_ids, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["data"] = data;
        keywords["segment_ids"] = segment_ids;
        var _op = tf.OpDefLib._apply_op_helper("SegmentMin", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tindices", _op._get_attr_type("Tindices") };
            _execute.record_gradient("SegmentMin", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor segment_min_eager_fallback(Tensor data, Tensor segment_ids, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { data, segment_ids };
        object[] _attrs = new object[] { "T", data.dtype, "Tindices", segment_ids.dtype };
        var _result = _execute.execute("SegmentMin", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SegmentMin", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the product along segments of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Read
    /// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
    /// for an explanation of segments.
    /// 
    /// Computes a tensor such that
    /// \(output_i = prod_j data_j\) where the product is over `j` such
    /// that `segment_ids[j] == i`.
    /// 
    /// If the product is empty for a given segment ID `i`, `output[i] = 1`.
    /// 
    /// Caution: On CPU, values in `segment_ids` are always validated to be sorted,
    /// and an error is thrown for indices that are not increasing. On GPU, this
    /// does not throw an error for unsorted indices. On GPU, out-of-order indices
    /// result in safe but unspecified behavior, which may include treating
    /// out-of-order indices as the same as a smaller following index.
    /// 
    /// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
    /// <img style="width:100%" src="https://www.tensorflow.org/images/SegmentProd.png" alt>
    /// </div>
    /// 
    /// For example:
    /// 
    /// >>> c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
    /// >>> tf.math.segment_prod(c, tf.constant([0, 0, 1])).numpy()
    /// array([[4, 6, 6, 4],
    ///        [5, 6, 7, 8]], dtype=int32)
    /// 
    /// 
    /// </remarks>
    /// <param name="data"></param>
    /// <param name="segment_ids"></param>
    /// <returns></returns>
    public static Tensor segment_prod(Tensor data, Tensor segment_ids, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SegmentProd", name) { args = new object[] { data, segment_ids }, attrs = new Dictionary<string, object>() { } });
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
                return segment_prod_eager_fallback(data, segment_ids, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["data"] = data;
        keywords["segment_ids"] = segment_ids;
        var _op = tf.OpDefLib._apply_op_helper("SegmentProd", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tindices", _op._get_attr_type("Tindices") };
            _execute.record_gradient("SegmentProd", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor segment_prod_eager_fallback(Tensor data, Tensor segment_ids, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { data, segment_ids };
        object[] _attrs = new object[] { "T", data.dtype, "Tindices", segment_ids.dtype };
        var _result = _execute.execute("SegmentProd", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SegmentProd", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the sum along segments of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Read
    /// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
    /// for an explanation of segments.
    /// 
    /// Computes a tensor such that
    /// \(output_i = sum_j data_j\) where sum is over `j` such
    /// that `segment_ids[j] == i`.
    /// 
    /// If the sum is empty for a given segment ID `i`, `output[i] = 0`.
    /// 
    /// Caution: On CPU, values in `segment_ids` are always validated to be sorted,
    /// and an error is thrown for indices that are not increasing. On GPU, this
    /// does not throw an error for unsorted indices. On GPU, out-of-order indices
    /// result in safe but unspecified behavior, which may include treating
    /// out-of-order indices as the same as a smaller following index.
    /// 
    /// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
    /// <img style="width:100%" src="https://www.tensorflow.org/images/SegmentSum.png" alt>
    /// </div>
    /// 
    /// For example:
    /// 
    /// >>> c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
    /// >>> tf.math.segment_sum(c, tf.constant([0, 0, 1])).numpy()
    /// array([[5, 5, 5, 5],
    ///        [5, 6, 7, 8]], dtype=int32)
    /// 
    /// 
    /// </remarks>
    /// <param name="data"></param>
    /// <param name="segment_ids"></param>
    /// <returns></returns>
    public static Tensor segment_sum(Tensor data, Tensor segment_ids, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SegmentSum", name) { args = new object[] { data, segment_ids }, attrs = new Dictionary<string, object>() { } });
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
                return segment_sum_eager_fallback(data, segment_ids, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["data"] = data;
        keywords["segment_ids"] = segment_ids;
        var _op = tf.OpDefLib._apply_op_helper("SegmentSum", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tindices", _op._get_attr_type("Tindices") };
            _execute.record_gradient("SegmentSum", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor segment_sum_eager_fallback(Tensor data, Tensor segment_ids, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { data, segment_ids };
        object[] _attrs = new object[] { "T", data.dtype, "Tindices", segment_ids.dtype };
        var _result = _execute.execute("SegmentSum", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SegmentSum", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Selects elements from `t` or `e`, depending on `condition`.
    /// </summary>
    /// <remarks>
    /// 
    /// The `t`, and `e` tensors must all have the same shape, and the
    /// output will also have that shape.
    /// 
    /// The `condition` tensor must be a scalar if `t` and `e` are scalars.
    /// If `t` and `e` are vectors or higher rank, then `condition` must be either a
    /// scalar, a vector with size matching the first dimension of `t`, or must have
    /// the same shape as `t`.
    /// 
    /// The `condition` tensor acts as a mask that chooses, based on the value at each
    /// element, whether the corresponding element / row in the output should be
    /// taken from `t` (if true) or `e` (if false).
    /// 
    /// If `condition` is a vector and `t` and `e` are higher rank matrices, then
    /// it chooses which row (outer dimension) to copy from `t` and `e`.
    /// If `condition` has the same shape as `t` and `e`, then it chooses which
    /// element to copy from `t` and `e`.
    /// 
    /// For example:
    /// 
    /// ```python
    /// # 'condition' tensor is [[True,  False]
    /// #                        [False, True]]
    /// # 't' is [[1, 2],
    /// #         [3, 4]]
    /// # 'e' is [[5, 6],
    /// #         [7, 8]]
    /// select(condition, t, e)  # => [[1, 6], [7, 4]]
    /// 
    /// 
    /// # 'condition' tensor is [True, False]
    /// # 't' is [[1, 2],
    /// #         [3, 4]]
    /// # 'e' is [[5, 6],
    /// #         [7, 8]]
    /// select(condition, t, e) ==> [[1, 2],
    ///                              [7, 8]]
    /// 
    /// ```
    /// 
    /// </remarks>
    /// <param name="condition"></param>
    /// <param name="t"></param>
    /// <param name="e"></param>
    /// <returns></returns>
    public static Tensor select(Tensor condition, Tensor t, Tensor e, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Select", name) { args = new object[] { condition, t, e }, attrs = new Dictionary<string, object>() { } });
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
                return select_eager_fallback(condition, t, e, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["condition"] = condition;
        keywords["t"] = t;
        keywords["e"] = e;
        var _op = tf.OpDefLib._apply_op_helper("Select", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Select", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor select_eager_fallback(Tensor condition, Tensor t, Tensor e, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { condition, t, e };
        object[] _attrs = new object[] { "T", t.dtype };
        var _result = _execute.execute("Select", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Select", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="condition"></param>
    /// <param name="t"></param>
    /// <param name="e"></param>
    /// <returns></returns>
    public static Tensor select_v2(Tensor condition, Tensor t, Tensor e, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SelectV2", name) { args = new object[] { condition, t, e }, attrs = new Dictionary<string, object>() { } });
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
                return select_v2_eager_fallback(condition, t, e, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["condition"] = condition;
        keywords["t"] = t;
        keywords["e"] = e;
        var _op = tf.OpDefLib._apply_op_helper("SelectV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("SelectV2", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor select_v2_eager_fallback(Tensor condition, Tensor t, Tensor e, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { condition, t, e };
        object[] _attrs = new object[] { "T", t.dtype };
        var _result = _execute.execute("SelectV2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SelectV2", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes sigmoid of `x` element-wise.
    /// </summary>
    /// <remarks>
    /// 
    /// Specifically, `y = 1 / (1 + exp(-x))`.
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor sigmoid(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Sigmoid", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return sigmoid_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Sigmoid", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Sigmoid", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor sigmoid_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Sigmoid", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Sigmoid", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the gradient of the sigmoid of `x` wrt its input.
    /// </summary>
    /// <remarks>
    /// 
    /// Specifically, `grad = dy * y * (1 - y)`, where `y = sigmoid(x)`, and
    /// `dy` is the corresponding input gradient.
    /// 
    /// </remarks>
    /// <param name="y"></param>
    /// <param name="dy"></param>
    /// <returns></returns>
    public static Tensor sigmoid_grad(Tensor y, Tensor dy, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SigmoidGrad", name) { args = new object[] { y, dy }, attrs = new Dictionary<string, object>() { } });
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
                return sigmoid_grad_eager_fallback(y, dy, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["y"] = y;
        keywords["dy"] = dy;
        var _op = tf.OpDefLib._apply_op_helper("SigmoidGrad", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("SigmoidGrad", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor sigmoid_grad_eager_fallback(Tensor y, Tensor dy, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { y, dy };
        object[] _attrs = new object[] { "T", y.dtype };
        var _result = _execute.execute("SigmoidGrad", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SigmoidGrad", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns an element-wise indication of the sign of a number.
    /// </summary>
    /// <remarks>
    /// 
    /// `y = sign(x) = -1` if `x < 0`; 0 if `x == 0`; 1 if `x > 0`.
    /// 
    /// For complex numbers, `y = sign(x) = x / |x|` if `x != 0`, otherwise `y = 0`.
    /// 
    /// Example usage:
    /// >>> tf.math.sign([0., 2., -3.])
    /// <tf.Tensor: shape=(3,), dtype=float32, numpy=array([ 0.,  1., -1.], dtype=float32)>
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor sign(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Sign", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return sign_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Sign", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Sign", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor sign_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Sign", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Sign", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes sine of x element-wise.
    /// </summary>
    /// <remarks>
    /// 
    ///   Given an input tensor, this function computes sine of every
    ///   element in the tensor. Input range is `(-inf, inf)` and
    ///   output range is `[-1,1]`.
    /// 
    ///   ```python
    ///   x = tf.constant([-float("inf"), -9, -0.5, 1, 1.2, 200, 10, float("inf")])
    ///   tf.math.sin(x) ==> [nan -0.4121185 -0.47942555 0.84147096 0.9320391 -0.87329733 -0.54402107 nan]
    ///   ```
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor sin(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Sin", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return sin_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Sin", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Sin", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor sin_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Sin", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Sin", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes hyperbolic sine of x element-wise.
    /// </summary>
    /// <remarks>
    /// 
    ///   Given an input tensor, this function computes hyperbolic sine of every
    ///   element in the tensor. Input range is `[-inf,inf]` and output range
    ///   is `[-inf,inf]`.
    /// 
    ///   ```python
    ///   x = tf.constant([-float("inf"), -9, -0.5, 1, 1.2, 2, 10, float("inf")])
    ///   tf.math.sinh(x) ==> [-inf -4.0515420e+03 -5.2109528e-01 1.1752012e+00 1.5094614e+00 3.6268604e+00 1.1013232e+04 inf]
    ///   ```
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor sinh(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Sinh", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return sinh_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Sinh", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Sinh", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor sinh_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Sinh", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Sinh", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Generates points from the Sobol sequence.
    /// </summary>
    /// <remarks>
    /// 
    /// Creates a Sobol sequence with `num_results` samples. Each sample has dimension
    /// `dim`. Skips the first `skip` samples.
    /// 
    /// </remarks>
    /// <param name="dim"></param>
    /// <param name="num_results"></param>
    /// <param name="skip"></param>
    /// <param name="dtype">
    /// 
    /// The type of the sample. One of: `float32` or `float64`.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor sobol_sample(Tensor dim, Tensor num_results, Tensor skip, TF_DataType dtype = TF_DataType.TF_FLOAT, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SobolSample", name) { args = new object[] { dim, num_results, skip }, attrs = new Dictionary<string, object>() { ["dtype"] = dtype } });
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
                return sobol_sample_eager_fallback(dim, num_results, skip, dtype: dtype, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["dim"] = dim;
        keywords["num_results"] = num_results;
        keywords["skip"] = skip;
        keywords["dtype"] = dtype;
        var _op = tf.OpDefLib._apply_op_helper("SobolSample", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "dtype", _op._get_attr_type("dtype") };
            _execute.record_gradient("SobolSample", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor sobol_sample_eager_fallback(Tensor dim, Tensor num_results, Tensor skip, TF_DataType dtype, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { dim, num_results, skip };
        object[] _attrs = new object[] { "dtype", dtype };
        var _result = _execute.execute("SobolSample", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SobolSample", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Counts the number of occurrences of each value in an integer array.
    /// </summary>
    /// <remarks>
    /// 
    /// Outputs a vector with length `size` and the same dtype as `weights`. If
    /// `weights` are empty, then index `i` stores the number of times the value `i` is
    /// counted in `arr`. If `weights` are non-empty, then index `i` stores the sum of
    /// the value in `weights` at each index where the corresponding value in `arr` is
    /// `i`.
    /// 
    /// Values in `arr` outside of the range [0, size) are ignored.
    /// 
    /// </remarks>
    /// <param name="indices"></param>
    /// <param name="values"></param>
    /// <param name="dense_shape"></param>
    /// <param name="size"></param>
    /// <param name="weights"></param>
    /// <param name="binary_output">
    /// 
    /// bool; Whether the kernel should count the appearance or number of occurrences.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor sparse_bincount(Tensor indices, Tensor values, Tensor dense_shape, Tensor size, Tensor weights, bool binary_output = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SparseBincount", name) { args = new object[] { indices, values, dense_shape, size, weights }, attrs = new Dictionary<string, object>() { ["binary_output"] = binary_output } });
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
                return sparse_bincount_eager_fallback(indices, values, dense_shape, size, weights, binary_output: binary_output, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["indices"] = indices;
        keywords["values"] = values;
        keywords["dense_shape"] = dense_shape;
        keywords["size"] = size;
        keywords["weights"] = weights;
        keywords["binary_output"] = binary_output;
        var _op = tf.OpDefLib._apply_op_helper("SparseBincount", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tidx", _op._get_attr_type("Tidx"), "T", _op._get_attr_type("T"), "binary_output", _op._get_attr_bool("binary_output") };
            _execute.record_gradient("SparseBincount", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor sparse_bincount_eager_fallback(Tensor indices, Tensor values, Tensor dense_shape, Tensor size, Tensor weights, bool binary_output, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { indices, values, dense_shape, size, weights };
        object[] _attrs = new object[] { "Tidx", values.dtype, "T", weights.dtype, "binary_output", binary_output };
        var _result = _execute.execute("SparseBincount", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SparseBincount", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Multiply matrix "a" by matrix "b".
    /// </summary>
    /// <remarks>
    /// 
    /// The inputs must be two-dimensional matrices and the inner dimension of "a" must
    /// match the outer dimension of "b". Both "a" and "b" must be `Tensor`s not
    /// `SparseTensor`s.  This op is optimized for the case where at least one of "a" or
    /// "b" is sparse, in the sense that they have a large proportion of zero values.
    /// The breakeven for using this versus a dense matrix multiply on one platform was
    /// 30% zero values in the sparse matrix.
    /// 
    /// The gradient computation of this operation will only take advantage of sparsity
    /// in the input gradient when that gradient comes from a Relu.
    /// 
    /// </remarks>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <param name="transpose_a"></param>
    /// <param name="transpose_b"></param>
    /// <param name="a_is_sparse"></param>
    /// <param name="b_is_sparse"></param>
    /// <returns></returns>
    public static Tensor sparse_mat_mul(Tensor a, Tensor b, bool transpose_a = false, bool transpose_b = false, bool a_is_sparse = false, bool b_is_sparse = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SparseMatMul", name) { args = new object[] { a, b }, attrs = new Dictionary<string, object>() { ["transpose_a"] = transpose_a, ["transpose_b"] = transpose_b, ["a_is_sparse"] = a_is_sparse, ["b_is_sparse"] = b_is_sparse } });
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
                return sparse_mat_mul_eager_fallback(a, b, transpose_a: transpose_a, transpose_b: transpose_b, a_is_sparse: a_is_sparse, b_is_sparse: b_is_sparse, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["a"] = a;
        keywords["b"] = b;
        keywords["transpose_a"] = transpose_a;
        keywords["transpose_b"] = transpose_b;
        keywords["a_is_sparse"] = a_is_sparse;
        keywords["b_is_sparse"] = b_is_sparse;
        var _op = tf.OpDefLib._apply_op_helper("SparseMatMul", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "transpose_a", _op._get_attr_bool("transpose_a"), "transpose_b", _op._get_attr_bool("transpose_b"), "a_is_sparse", _op._get_attr_bool("a_is_sparse"), "b_is_sparse", _op._get_attr_bool("b_is_sparse"), "Ta", _op._get_attr_type("Ta"), "Tb", _op._get_attr_type("Tb") };
            _execute.record_gradient("SparseMatMul", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor sparse_mat_mul_eager_fallback(Tensor a, Tensor b, bool transpose_a, bool transpose_b, bool a_is_sparse, bool b_is_sparse, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { a, b };
        object[] _attrs = new object[] { "transpose_a", transpose_a, "transpose_b", transpose_b, "a_is_sparse", a_is_sparse, "b_is_sparse", b_is_sparse, "Ta", a.dtype, "Tb", b.dtype };
        var _result = _execute.execute("SparseMatMul", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SparseMatMul", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the mean along sparse segments of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// See `tf.sparse.segment_sum` for usage examples.
    /// 
    /// Like `SegmentMean`, but `segment_ids` can have rank less than `data`'s first
    /// dimension, selecting a subset of dimension 0, specified by `indices`.
    /// 
    /// </remarks>
    /// <param name="data"></param>
    /// <param name="indices"></param>
    /// <param name="segment_ids"></param>
    /// <returns></returns>
    public static Tensor sparse_segment_mean(Tensor data, Tensor indices, Tensor segment_ids, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SparseSegmentMean", name) { args = new object[] { data, indices, segment_ids }, attrs = new Dictionary<string, object>() { } });
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
                return sparse_segment_mean_eager_fallback(data, indices, segment_ids, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["data"] = data;
        keywords["indices"] = indices;
        keywords["segment_ids"] = segment_ids;
        var _op = tf.OpDefLib._apply_op_helper("SparseSegmentMean", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tidx", _op._get_attr_type("Tidx"), "Tsegmentids", _op._get_attr_type("Tsegmentids") };
            _execute.record_gradient("SparseSegmentMean", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor sparse_segment_mean_eager_fallback(Tensor data, Tensor indices, Tensor segment_ids, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { data, indices, segment_ids };
        object[] _attrs = new object[] { "T", data.dtype, "Tidx", indices.dtype, "Tsegmentids", segment_ids.dtype };
        var _result = _execute.execute("SparseSegmentMean", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SparseSegmentMean", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes gradients for SparseSegmentMean.
    /// </summary>
    /// <remarks>
    /// 
    /// Returns tensor "output" with same shape as grad, except for dimension 0 whose
    /// value is output_dim0.
    /// 
    /// </remarks>
    /// <param name="grad"></param>
    /// <param name="indices"></param>
    /// <param name="segment_ids"></param>
    /// <param name="output_dim0"></param>
    /// <returns></returns>
    public static Tensor sparse_segment_mean_grad(Tensor grad, Tensor indices, Tensor segment_ids, Tensor output_dim0, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SparseSegmentMeanGrad", name) { args = new object[] { grad, indices, segment_ids, output_dim0 }, attrs = new Dictionary<string, object>() { } });
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
                return sparse_segment_mean_grad_eager_fallback(grad, indices, segment_ids, output_dim0, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["grad"] = grad;
        keywords["indices"] = indices;
        keywords["segment_ids"] = segment_ids;
        keywords["output_dim0"] = output_dim0;
        var _op = tf.OpDefLib._apply_op_helper("SparseSegmentMeanGrad", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tidx", _op._get_attr_type("Tidx"), "Tsegmentids", _op._get_attr_type("Tsegmentids") };
            _execute.record_gradient("SparseSegmentMeanGrad", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor sparse_segment_mean_grad_eager_fallback(Tensor grad, Tensor indices, Tensor segment_ids, Tensor output_dim0, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { grad, indices, segment_ids, output_dim0 };
        object[] _attrs = new object[] { "T", grad.dtype, "Tidx", indices.dtype, "Tsegmentids", segment_ids.dtype };
        var _result = _execute.execute("SparseSegmentMeanGrad", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SparseSegmentMeanGrad", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the mean along sparse segments of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Like `SparseSegmentMean`, but allows missing ids in `segment_ids`. If an id is
    /// missing, the `output` tensor at that position will be zeroed.
    /// 
    /// Read
    /// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
    /// for an explanation of segments.
    /// 
    /// </remarks>
    /// <param name="data"></param>
    /// <param name="indices"></param>
    /// <param name="segment_ids"></param>
    /// <param name="num_segments"></param>
    /// <returns></returns>
    public static Tensor sparse_segment_mean_with_num_segments(Tensor data, Tensor indices, Tensor segment_ids, Tensor num_segments, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SparseSegmentMeanWithNumSegments", name) { args = new object[] { data, indices, segment_ids, num_segments }, attrs = new Dictionary<string, object>() { } });
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
                return sparse_segment_mean_with_num_segments_eager_fallback(data, indices, segment_ids, num_segments, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["data"] = data;
        keywords["indices"] = indices;
        keywords["segment_ids"] = segment_ids;
        keywords["num_segments"] = num_segments;
        var _op = tf.OpDefLib._apply_op_helper("SparseSegmentMeanWithNumSegments", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tidx", _op._get_attr_type("Tidx"), "Tnumsegments", _op._get_attr_type("Tnumsegments"), "Tsegmentids", _op._get_attr_type("Tsegmentids") };
            _execute.record_gradient("SparseSegmentMeanWithNumSegments", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor sparse_segment_mean_with_num_segments_eager_fallback(Tensor data, Tensor indices, Tensor segment_ids, Tensor num_segments, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { data, indices, segment_ids, num_segments };
        object[] _attrs = new object[] { "T", data.dtype, "Tidx", indices.dtype, "Tnumsegments", num_segments.dtype, "Tsegmentids", segment_ids.dtype };
        var _result = _execute.execute("SparseSegmentMeanWithNumSegments", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SparseSegmentMeanWithNumSegments", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the sum along sparse segments of a tensor divided by the sqrt of N.
    /// </summary>
    /// <remarks>
    /// 
    /// N is the size of the segment being reduced.
    /// 
    /// See `tf.sparse.segment_sum` for usage examples.
    /// 
    /// 
    /// </remarks>
    /// <param name="data"></param>
    /// <param name="indices"></param>
    /// <param name="segment_ids"></param>
    /// <returns></returns>
    public static Tensor sparse_segment_sqrt_n(Tensor data, Tensor indices, Tensor segment_ids, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SparseSegmentSqrtN", name) { args = new object[] { data, indices, segment_ids }, attrs = new Dictionary<string, object>() { } });
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
                return sparse_segment_sqrt_n_eager_fallback(data, indices, segment_ids, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["data"] = data;
        keywords["indices"] = indices;
        keywords["segment_ids"] = segment_ids;
        var _op = tf.OpDefLib._apply_op_helper("SparseSegmentSqrtN", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tidx", _op._get_attr_type("Tidx"), "Tsegmentids", _op._get_attr_type("Tsegmentids") };
            _execute.record_gradient("SparseSegmentSqrtN", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor sparse_segment_sqrt_n_eager_fallback(Tensor data, Tensor indices, Tensor segment_ids, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { data, indices, segment_ids };
        object[] _attrs = new object[] { "T", data.dtype, "Tidx", indices.dtype, "Tsegmentids", segment_ids.dtype };
        var _result = _execute.execute("SparseSegmentSqrtN", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SparseSegmentSqrtN", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the sum along sparse segments of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Read
    /// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
    /// for an explanation of segments.
    /// 
    /// Like `SegmentSum`, but `segment_ids` can have rank less than `data`'s first
    /// dimension, selecting a subset of dimension 0, specified by `indices`.
    /// 
    /// For example:
    /// 
    /// ```python
    /// c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])
    /// 
    /// # Select two rows, one segment.
    /// tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 0]))
    /// # => [[0 0 0 0]]
    /// 
    /// # Select two rows, two segment.
    /// tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 1]))
    /// # => [[ 1  2  3  4]
    /// #     [-1 -2 -3 -4]]
    /// 
    /// # Select all rows, two segments.
    /// tf.sparse_segment_sum(c, tf.constant([0, 1, 2]), tf.constant([0, 0, 1]))
    /// # => [[0 0 0 0]
    /// #     [5 6 7 8]]
    /// 
    /// # Which is equivalent to:
    /// tf.segment_sum(c, tf.constant([0, 0, 1]))
    /// ```
    /// 
    /// </remarks>
    /// <param name="data"></param>
    /// <param name="indices"></param>
    /// <param name="segment_ids"></param>
    /// <returns></returns>
    public static Tensor sparse_segment_sum(Tensor data, Tensor indices, Tensor segment_ids, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SparseSegmentSum", name) { args = new object[] { data, indices, segment_ids }, attrs = new Dictionary<string, object>() { } });
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
                return sparse_segment_sum_eager_fallback(data, indices, segment_ids, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["data"] = data;
        keywords["indices"] = indices;
        keywords["segment_ids"] = segment_ids;
        var _op = tf.OpDefLib._apply_op_helper("SparseSegmentSum", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tidx", _op._get_attr_type("Tidx"), "Tsegmentids", _op._get_attr_type("Tsegmentids") };
            _execute.record_gradient("SparseSegmentSum", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor sparse_segment_sum_eager_fallback(Tensor data, Tensor indices, Tensor segment_ids, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { data, indices, segment_ids };
        object[] _attrs = new object[] { "T", data.dtype, "Tidx", indices.dtype, "Tsegmentids", segment_ids.dtype };
        var _result = _execute.execute("SparseSegmentSum", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SparseSegmentSum", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes gradients for SparseSegmentSum.
    /// </summary>
    /// <remarks>
    /// 
    /// Returns tensor "output" with same shape as grad, except for dimension 0 whose
    /// value is output_dim0.
    /// 
    /// </remarks>
    /// <param name="grad"></param>
    /// <param name="indices"></param>
    /// <param name="segment_ids"></param>
    /// <param name="output_dim0"></param>
    /// <returns></returns>
    public static Tensor sparse_segment_sum_grad(Tensor grad, Tensor indices, Tensor segment_ids, Tensor output_dim0, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SparseSegmentSumGrad", name) { args = new object[] { grad, indices, segment_ids, output_dim0 }, attrs = new Dictionary<string, object>() { } });
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
                return sparse_segment_sum_grad_eager_fallback(grad, indices, segment_ids, output_dim0, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["grad"] = grad;
        keywords["indices"] = indices;
        keywords["segment_ids"] = segment_ids;
        keywords["output_dim0"] = output_dim0;
        var _op = tf.OpDefLib._apply_op_helper("SparseSegmentSumGrad", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tidx", _op._get_attr_type("Tidx"), "Tsegmentids", _op._get_attr_type("Tsegmentids") };
            _execute.record_gradient("SparseSegmentSumGrad", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor sparse_segment_sum_grad_eager_fallback(Tensor grad, Tensor indices, Tensor segment_ids, Tensor output_dim0, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { grad, indices, segment_ids, output_dim0 };
        object[] _attrs = new object[] { "T", grad.dtype, "Tidx", indices.dtype, "Tsegmentids", segment_ids.dtype };
        var _result = _execute.execute("SparseSegmentSumGrad", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SparseSegmentSumGrad", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the sum along sparse segments of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Like `SparseSegmentSum`, but allows missing ids in `segment_ids`. If an id is
    /// missing, the `output` tensor at that position will be zeroed.
    /// 
    /// Read
    /// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/sparse#Segmentation)
    /// for an explanation of segments.
    /// 
    /// For example:
    /// 
    /// ```python
    /// c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])
    /// 
    /// tf.sparse_segment_sum_with_num_segments(
    ///     c, tf.constant([0, 1]), tf.constant([0, 0]), num_segments=3)
    /// # => [[0 0 0 0]
    /// #     [0 0 0 0]
    /// #     [0 0 0 0]]
    /// 
    /// tf.sparse_segment_sum_with_num_segments(c,
    ///                                         tf.constant([0, 1]),
    ///                                         tf.constant([0, 2],
    ///                                         num_segments=4))
    /// # => [[ 1  2  3  4]
    /// #     [ 0  0  0  0]
    /// #     [-1 -2 -3 -4]
    /// #     [ 0  0  0  0]]
    /// ```
    /// 
    /// </remarks>
    /// <param name="data"></param>
    /// <param name="indices"></param>
    /// <param name="segment_ids"></param>
    /// <param name="num_segments"></param>
    /// <returns></returns>
    public static Tensor sparse_segment_sum_with_num_segments(Tensor data, Tensor indices, Tensor segment_ids, Tensor num_segments, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SparseSegmentSumWithNumSegments", name) { args = new object[] { data, indices, segment_ids, num_segments }, attrs = new Dictionary<string, object>() { } });
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
                return sparse_segment_sum_with_num_segments_eager_fallback(data, indices, segment_ids, num_segments, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["data"] = data;
        keywords["indices"] = indices;
        keywords["segment_ids"] = segment_ids;
        keywords["num_segments"] = num_segments;
        var _op = tf.OpDefLib._apply_op_helper("SparseSegmentSumWithNumSegments", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tidx", _op._get_attr_type("Tidx"), "Tnumsegments", _op._get_attr_type("Tnumsegments"), "Tsegmentids", _op._get_attr_type("Tsegmentids") };
            _execute.record_gradient("SparseSegmentSumWithNumSegments", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor sparse_segment_sum_with_num_segments_eager_fallback(Tensor data, Tensor indices, Tensor segment_ids, Tensor num_segments, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { data, indices, segment_ids, num_segments };
        object[] _attrs = new object[] { "T", data.dtype, "Tidx", indices.dtype, "Tnumsegments", num_segments.dtype, "Tsegmentids", segment_ids.dtype };
        var _result = _execute.execute("SparseSegmentSumWithNumSegments", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SparseSegmentSumWithNumSegments", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes square root of x element-wise.
    /// </summary>
    /// <remarks>
    /// 
    /// I.e., \(y = sqrt{x} = x^{1/2}\).
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor sqrt(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Sqrt", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return sqrt_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Sqrt", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Sqrt", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor sqrt_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Sqrt", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Sqrt", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the gradient for the sqrt of `x` wrt its input.
    /// </summary>
    /// <remarks>
    /// 
    /// Specifically, `grad = dy * 0.5 / y`, where `y = sqrt(x)`, and `dy`
    /// is the corresponding input gradient.
    /// 
    /// </remarks>
    /// <param name="y"></param>
    /// <param name="dy"></param>
    /// <returns></returns>
    public static Tensor sqrt_grad(Tensor y, Tensor dy, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SqrtGrad", name) { args = new object[] { y, dy }, attrs = new Dictionary<string, object>() { } });
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
                return sqrt_grad_eager_fallback(y, dy, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["y"] = y;
        keywords["dy"] = dy;
        var _op = tf.OpDefLib._apply_op_helper("SqrtGrad", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("SqrtGrad", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor sqrt_grad_eager_fallback(Tensor y, Tensor dy, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { y, dy };
        object[] _attrs = new object[] { "T", y.dtype };
        var _result = _execute.execute("SqrtGrad", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SqrtGrad", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes square of x element-wise.
    /// </summary>
    /// <remarks>
    /// 
    /// I.e., \(y = x * x = x^2\).
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor square(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Square", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return square_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Square", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Square", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor square_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Square", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Square", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns conj(x - y)(x - y) element-wise.
    /// </summary>
    /// <remarks>
    /// 
    /// *NOTE*: `SquaredDifference` supports broadcasting. More about broadcasting
    /// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <returns></returns>
    public static Tensor squared_difference(Tensor x, Tensor y, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SquaredDifference", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { } });
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
                return squared_difference_eager_fallback(x, y, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        var _op = tf.OpDefLib._apply_op_helper("SquaredDifference", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("SquaredDifference", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor squared_difference_eager_fallback(Tensor x, Tensor y, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("SquaredDifference", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SquaredDifference", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns x - y element-wise.
    /// </summary>
    /// <remarks>
    /// 
    /// *NOTE*: `Sub` supports broadcasting. More about broadcasting
    /// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <returns></returns>
    public static Tensor sub(Tensor x, Tensor y, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Sub", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { } });
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
                return sub_eager_fallback(x, y, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        var _op = tf.OpDefLib._apply_op_helper("Sub", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Sub", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor sub_eager_fallback(Tensor x, Tensor y, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Sub", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Sub", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the sum of elements across dimensions of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Reduces `input` along the dimensions given in `reduction_indices`. Unless
    /// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
    /// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
    /// retained with length 1.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="reduction_indices"></param>
    /// <param name="keep_dims">
    /// 
    /// If true, retain reduced dimensions with length 1.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor sum(Tensor input, Tensor reduction_indices, bool keep_dims = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Sum", name) { args = new object[] { input, reduction_indices }, attrs = new Dictionary<string, object>() { ["keep_dims"] = keep_dims } });
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
                return sum_eager_fallback(input, reduction_indices, keep_dims: keep_dims, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["reduction_indices"] = reduction_indices;
        keywords["keep_dims"] = keep_dims;
        var _op = tf.OpDefLib._apply_op_helper("Sum", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "keep_dims", _op._get_attr_bool("keep_dims"), "T", _op._get_attr_type("T"), "Tidx", _op._get_attr_type("Tidx") };
            _execute.record_gradient("Sum", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor sum_eager_fallback(Tensor input, Tensor reduction_indices, bool keep_dims, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, reduction_indices };
        object[] _attrs = new object[] { "keep_dims", keep_dims, "T", input.dtype, "Tidx", reduction_indices.dtype };
        var _result = _execute.execute("Sum", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Sum", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes tan of x element-wise.
    /// </summary>
    /// <remarks>
    /// 
    ///   Given an input tensor, this function computes tangent of every
    ///   element in the tensor. Input range is `(-inf, inf)` and
    ///   output range is `(-inf, inf)`. If input lies outside the boundary, `nan`
    ///   is returned.
    /// 
    ///   ```python
    ///   x = tf.constant([-float("inf"), -9, -0.5, 1, 1.2, 200, 10000, float("inf")])
    ///   tf.math.tan(x) ==> [nan 0.45231566 -0.5463025 1.5574077 2.572152 -1.7925274 0.32097113 nan]
    ///   ```
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor tan(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Tan", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return tan_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Tan", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Tan", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor tan_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Tan", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Tan", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes hyperbolic tangent of `x` element-wise.
    /// </summary>
    /// <remarks>
    /// 
    ///   Given an input tensor, this function computes hyperbolic tangent of every
    ///   element in the tensor. Input range is `[-inf, inf]` and
    ///   output range is `[-1,1]`.
    /// 
    ///   >>> x = tf.constant([-float("inf"), -5, -0.5, 1, 1.2, 2, 3, float("inf")])
    ///   >>> tf.math.tanh(x)
    ///   <tf.Tensor: shape=(8,), dtype=float32, numpy=
    ///   array([-1.0, -0.99990916, -0.46211717,  0.7615942 ,  0.8336547 ,
    ///           0.9640276 ,  0.9950547 ,  1.0], dtype=float32)>
    /// 
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <returns></returns>
    public static Tensor tanh(Tensor x, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Tanh", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { } });
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
                return tanh_eager_fallback(x, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        var _op = tf.OpDefLib._apply_op_helper("Tanh", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Tanh", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor tanh_eager_fallback(Tensor x, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Tanh", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Tanh", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the gradient for the tanh of `x` wrt its input.
    /// </summary>
    /// <remarks>
    /// 
    /// Specifically, `grad = dy * (1 - y*y)`, where `y = tanh(x)`, and `dy`
    /// is the corresponding input gradient.
    /// 
    /// </remarks>
    /// <param name="y"></param>
    /// <param name="dy"></param>
    /// <returns></returns>
    public static Tensor tanh_grad(Tensor y, Tensor dy, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TanhGrad", name) { args = new object[] { y, dy }, attrs = new Dictionary<string, object>() { } });
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
                return tanh_grad_eager_fallback(y, dy, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["y"] = y;
        keywords["dy"] = dy;
        var _op = tf.OpDefLib._apply_op_helper("TanhGrad", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("TanhGrad", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor tanh_grad_eager_fallback(Tensor y, Tensor dy, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { y, dy };
        object[] _attrs = new object[] { "T", y.dtype };
        var _result = _execute.execute("TanhGrad", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TanhGrad", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns x / y element-wise for integer types.
    /// </summary>
    /// <remarks>
    /// 
    /// Truncation designates that negative numbers will round fractional quantities
    /// toward zero. I.e. -7 / 5 = -1. This matches C semantics but it is different
    /// than Python semantics. See `FloorDiv` for a division function that matches
    /// Python Semantics.
    /// 
    /// *NOTE*: `TruncateDiv` supports broadcasting. More about broadcasting
    /// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <returns></returns>
    public static Tensor truncate_div(Tensor x, Tensor y, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TruncateDiv", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { } });
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
                return truncate_div_eager_fallback(x, y, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        var _op = tf.OpDefLib._apply_op_helper("TruncateDiv", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("TruncateDiv", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor truncate_div_eager_fallback(Tensor x, Tensor y, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("TruncateDiv", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TruncateDiv", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns element-wise remainder of division. This emulates C semantics in that
    /// </summary>
    /// <remarks>
    /// 
    /// the result here is consistent with a truncating divide. E.g. `truncate(x / y) *
    /// y + truncate_mod(x, y) = x`.
    /// 
    /// *NOTE*: `TruncateMod` supports broadcasting. More about broadcasting
    /// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <returns></returns>
    public static Tensor truncate_mod(Tensor x, Tensor y, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TruncateMod", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { } });
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
                return truncate_mod_eager_fallback(x, y, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        var _op = tf.OpDefLib._apply_op_helper("TruncateMod", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("TruncateMod", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor truncate_mod_eager_fallback(Tensor x, Tensor y, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("TruncateMod", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TruncateMod", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the maximum along segments of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Read
    /// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
    /// for an explanation of segments.
    /// 
    /// This operator is similar to `tf.math.unsorted_segment_sum`,
    /// Instead of computing the sum over segments, it computes the maximum such that:
    /// 
    /// \(output_i = max_{j...} data[j...]\) where max is over tuples `j...` such
    /// that `segment_ids[j...] == i`.
    /// 
    /// If the maximum is empty for a given segment ID `i`, it outputs the smallest
    /// possible value for the specific numeric type,
    /// `output[i] = numeric_limits<T>::lowest()`.
    /// 
    /// If the given segment ID `i` is negative, then the corresponding value is
    /// dropped, and will not be included in the result.
    /// 
    /// Caution: On CPU, values in `segment_ids` are always validated to be less than
    /// `num_segments`, and an error is thrown for out-of-bound indices. On GPU, this
    /// does not throw an error for out-of-bound indices. On Gpu, out-of-bound indices
    /// result in safe but unspecified behavior, which may include ignoring
    /// out-of-bound indices or outputting a tensor with a 0 stored in the first
    /// dimension of its shape if `num_segments` is 0.
    /// 
    /// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
    /// <img style="width:100%" src="https://www.tensorflow.org/images/UnsortedSegmentMax.png" alt>
    /// </div>
    /// 
    /// For example:
    /// 
    /// >>> c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
    /// >>> tf.math.unsorted_segment_max(c, tf.constant([0, 1, 0]), num_segments=2).numpy()
    /// array([[4, 3, 3, 4],
    ///        [5,  6, 7, 8]], dtype=int32)
    /// 
    /// 
    /// </remarks>
    /// <param name="data"></param>
    /// <param name="segment_ids"></param>
    /// <param name="num_segments"></param>
    /// <returns></returns>
    public static Tensor unsorted_segment_max(Tensor data, Tensor segment_ids, Tensor num_segments, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "UnsortedSegmentMax", name) { args = new object[] { data, segment_ids, num_segments }, attrs = new Dictionary<string, object>() { } });
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
                return unsorted_segment_max_eager_fallback(data, segment_ids, num_segments, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["data"] = data;
        keywords["segment_ids"] = segment_ids;
        keywords["num_segments"] = num_segments;
        var _op = tf.OpDefLib._apply_op_helper("UnsortedSegmentMax", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tindices", _op._get_attr_type("Tindices"), "Tnumsegments", _op._get_attr_type("Tnumsegments") };
            _execute.record_gradient("UnsortedSegmentMax", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor unsorted_segment_max_eager_fallback(Tensor data, Tensor segment_ids, Tensor num_segments, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { data, segment_ids, num_segments };
        object[] _attrs = new object[] { "T", data.dtype, "Tindices", segment_ids.dtype, "Tnumsegments", num_segments.dtype };
        var _result = _execute.execute("UnsortedSegmentMax", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("UnsortedSegmentMax", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the minimum along segments of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Read
    /// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
    /// for an explanation of segments.
    /// 
    /// This operator is similar to `tf.math.unsorted_segment_sum`,
    /// Instead of computing the sum over segments, it computes the minimum such that:
    /// 
    /// \(output_i = min_{j...} data_[j...]\) where min is over tuples `j...` such
    /// that `segment_ids[j...] == i`.
    /// 
    /// If the minimum is empty for a given segment ID `i`, it outputs the largest
    /// possible value for the specific numeric type,
    /// `output[i] = numeric_limits<T>::max()`.
    /// 
    /// For example:
    /// 
    /// >>> c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
    /// >>> tf.math.unsorted_segment_min(c, tf.constant([0, 1, 0]), num_segments=2).numpy()
    /// array([[1, 2, 2, 1],
    ///        [5, 6, 7, 8]], dtype=int32)
    /// 
    /// If the given segment ID `i` is negative, then the corresponding value is
    /// dropped, and will not be included in the result.
    /// 
    /// Caution: On CPU, values in `segment_ids` are always validated to be less than
    /// `num_segments`, and an error is thrown for out-of-bound indices. On GPU, this
    /// does not throw an error for out-of-bound indices. On Gpu, out-of-bound indices
    /// result in safe but unspecified behavior, which may include ignoring
    /// out-of-bound indices or outputting a tensor with a 0 stored in the first
    /// dimension of its shape if `num_segments` is 0.
    /// 
    /// </remarks>
    /// <param name="data"></param>
    /// <param name="segment_ids"></param>
    /// <param name="num_segments"></param>
    /// <returns></returns>
    public static Tensor unsorted_segment_min(Tensor data, Tensor segment_ids, Tensor num_segments, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "UnsortedSegmentMin", name) { args = new object[] { data, segment_ids, num_segments }, attrs = new Dictionary<string, object>() { } });
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
                return unsorted_segment_min_eager_fallback(data, segment_ids, num_segments, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["data"] = data;
        keywords["segment_ids"] = segment_ids;
        keywords["num_segments"] = num_segments;
        var _op = tf.OpDefLib._apply_op_helper("UnsortedSegmentMin", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tindices", _op._get_attr_type("Tindices"), "Tnumsegments", _op._get_attr_type("Tnumsegments") };
            _execute.record_gradient("UnsortedSegmentMin", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor unsorted_segment_min_eager_fallback(Tensor data, Tensor segment_ids, Tensor num_segments, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { data, segment_ids, num_segments };
        object[] _attrs = new object[] { "T", data.dtype, "Tindices", segment_ids.dtype, "Tnumsegments", num_segments.dtype };
        var _result = _execute.execute("UnsortedSegmentMin", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("UnsortedSegmentMin", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the product along segments of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Read
    /// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
    /// for an explanation of segments.
    /// 
    /// This operator is similar to `tf.math.unsorted_segment_sum`,
    /// Instead of computing the sum over segments, it computes the product of all
    /// entries belonging to a segment such that:
    /// 
    /// \(output_i = prod_{j...} data[j...]\) where the product is over tuples
    /// `j...` such that `segment_ids[j...] == i`.
    /// 
    /// For example:
    /// 
    /// >>> c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
    /// >>> tf.math.unsorted_segment_prod(c, tf.constant([0, 1, 0]), num_segments=2).numpy()
    /// array([[4, 6, 6, 4],
    ///        [5, 6, 7, 8]], dtype=int32)
    /// 
    /// If there is no entry for a given segment ID `i`, it outputs 1.
    /// 
    /// If the given segment ID `i` is negative, then the corresponding value is
    /// dropped, and will not be included in the result.
    /// Caution: On CPU, values in `segment_ids` are always validated to be less than
    /// `num_segments`, and an error is thrown for out-of-bound indices. On GPU, this
    /// does not throw an error for out-of-bound indices. On Gpu, out-of-bound indices
    /// result in safe but unspecified behavior, which may include ignoring
    /// out-of-bound indices or outputting a tensor with a 0 stored in the first
    /// dimension of its shape if `num_segments` is 0.
    /// 
    /// 
    /// </remarks>
    /// <param name="data"></param>
    /// <param name="segment_ids"></param>
    /// <param name="num_segments"></param>
    /// <returns></returns>
    public static Tensor unsorted_segment_prod(Tensor data, Tensor segment_ids, Tensor num_segments, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "UnsortedSegmentProd", name) { args = new object[] { data, segment_ids, num_segments }, attrs = new Dictionary<string, object>() { } });
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
                return unsorted_segment_prod_eager_fallback(data, segment_ids, num_segments, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["data"] = data;
        keywords["segment_ids"] = segment_ids;
        keywords["num_segments"] = num_segments;
        var _op = tf.OpDefLib._apply_op_helper("UnsortedSegmentProd", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tindices", _op._get_attr_type("Tindices"), "Tnumsegments", _op._get_attr_type("Tnumsegments") };
            _execute.record_gradient("UnsortedSegmentProd", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor unsorted_segment_prod_eager_fallback(Tensor data, Tensor segment_ids, Tensor num_segments, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { data, segment_ids, num_segments };
        object[] _attrs = new object[] { "T", data.dtype, "Tindices", segment_ids.dtype, "Tnumsegments", num_segments.dtype };
        var _result = _execute.execute("UnsortedSegmentProd", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("UnsortedSegmentProd", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the sum along segments of a tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Read
    /// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
    /// for an explanation of segments.
    /// 
    /// Computes a tensor such that
    /// \(output[i] = sum_{j...} data[j...]\) where the sum is over tuples `j...` such
    /// that `segment_ids[j...] == i`.  Unlike `SegmentSum`, `segment_ids`
    /// need not be sorted and need not cover all values in the full
    /// range of valid values.
    /// 
    /// If the sum is empty for a given segment ID `i`, `output[i] = 0`.
    /// If the given segment ID `i` is negative, the value is dropped and will not be
    /// added to the sum of the segment.
    /// 
    /// `num_segments` should equal the number of distinct segment IDs.
    /// 
    /// Caution: On CPU, values in `segment_ids` are always validated to be less than
    /// `num_segments`, and an error is thrown for out-of-bound indices. On GPU, this
    /// does not throw an error for out-of-bound indices. On Gpu, out-of-bound indices
    /// result in safe but unspecified behavior, which may include ignoring
    /// out-of-bound indices or outputting a tensor with a 0 stored in the first
    /// dimension of its shape if `num_segments` is 0.
    /// 
    /// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
    /// <img style="width:100%" src="https://www.tensorflow.org/images/UnsortedSegmentSum.png" alt>
    /// </div>
    /// 
    /// >>> c = [[1,2,3,4], [5,6,7,8], [4,3,2,1]]
    /// >>> tf.math.unsorted_segment_sum(c, [0, 1, 0], num_segments=2).numpy()
    /// array([[5, 5, 5, 5],
    ///        [5, 6, 7, 8]], dtype=int32)
    /// 
    /// 
    /// 
    /// </remarks>
    /// <param name="data"></param>
    /// <param name="segment_ids"></param>
    /// <param name="num_segments"></param>
    /// <returns></returns>
    public static Tensor unsorted_segment_sum(Tensor data, Tensor segment_ids, Tensor num_segments, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "UnsortedSegmentSum", name) { args = new object[] { data, segment_ids, num_segments }, attrs = new Dictionary<string, object>() { } });
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
                return unsorted_segment_sum_eager_fallback(data, segment_ids, num_segments, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["data"] = data;
        keywords["segment_ids"] = segment_ids;
        keywords["num_segments"] = num_segments;
        var _op = tf.OpDefLib._apply_op_helper("UnsortedSegmentSum", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tindices", _op._get_attr_type("Tindices"), "Tnumsegments", _op._get_attr_type("Tnumsegments") };
            _execute.record_gradient("UnsortedSegmentSum", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor unsorted_segment_sum_eager_fallback(Tensor data, Tensor segment_ids, Tensor num_segments, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { data, segment_ids, num_segments };
        object[] _attrs = new object[] { "T", data.dtype, "Tindices", segment_ids.dtype, "Tnumsegments", num_segments.dtype };
        var _result = _execute.execute("UnsortedSegmentSum", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("UnsortedSegmentSum", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns 0 if x == 0, and x / y otherwise, elementwise.
    /// </summary>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <returns></returns>
    public static Tensor xdivy(Tensor x, Tensor y, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Xdivy", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { } });
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
                return xdivy_eager_fallback(x, y, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        var _op = tf.OpDefLib._apply_op_helper("Xdivy", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Xdivy", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor xdivy_eager_fallback(Tensor x, Tensor y, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Xdivy", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Xdivy", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns 0 if x == 0, and x * log1p(y) otherwise, elementwise.
    /// </summary>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <returns></returns>
    public static Tensor xlog1py(Tensor x, Tensor y, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Xlog1py", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { } });
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
                return xlog1py_eager_fallback(x, y, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        var _op = tf.OpDefLib._apply_op_helper("Xlog1py", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Xlog1py", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor xlog1py_eager_fallback(Tensor x, Tensor y, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Xlog1py", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Xlog1py", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns 0 if x == 0, and x * log(y) otherwise, elementwise.
    /// </summary>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <returns></returns>
    public static Tensor xlogy(Tensor x, Tensor y, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Xlogy", name) { args = new object[] { x, y }, attrs = new Dictionary<string, object>() { } });
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
                return xlogy_eager_fallback(x, y, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["y"] = y;
        var _op = tf.OpDefLib._apply_op_helper("Xlogy", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Xlogy", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor xlogy_eager_fallback(Tensor x, Tensor y, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, y };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Xlogy", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Xlogy", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Compute the Hurwitz zeta function \\(\zeta(x, q)\\).
    /// </summary>
    /// <remarks>
    /// 
    /// The Hurwitz zeta function is defined as:
    /// 
    /// 
    /// \(zeta(x, q) = sum_{n=0}^{infty} (q + n)^{-x}\)
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="q"></param>
    /// <returns></returns>
    public static Tensor zeta(Tensor x, Tensor q, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Zeta", name) { args = new object[] { x, q }, attrs = new Dictionary<string, object>() { } });
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
                return zeta_eager_fallback(x, q, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["q"] = q;
        var _op = tf.OpDefLib._apply_op_helper("Zeta", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Zeta", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor zeta_eager_fallback(Tensor x, Tensor q, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, q };
        object[] _attrs = new object[] { "T", x.dtype };
        var _result = _execute.execute("Zeta", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Zeta", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
}
