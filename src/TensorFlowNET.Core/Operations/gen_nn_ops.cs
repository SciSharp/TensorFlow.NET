/*Wrappers around TensorFlow ops. This file is MACHINE GENERATED! Do not edit.*/

using Tensorflow.Eager;
using Tensorflow.Contexts;
using Tensorflow.Exceptions;
using static Tensorflow.Binding;

namespace Tensorflow;

public static class gen_nn_ops
{
    /// <summary>
    /// Returns min/max k values and their indices of the input operand in an approximate manner.
    /// </summary>
    /// <remarks>
    /// 
    /// See https://arxiv.org/abs/2206.14286 for the algorithm details.
    /// This op is only optimized on TPU currently.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="k">
    /// Specifies the number of min/max-k.
    /// </param>
    /// <param name="reduction_dimension">
    /// Integer dimension along which to search. Default: -1.
    /// </param>
    /// <param name="recall_target">
    /// Recall target for the approximation. Range in (0,1]
    /// </param>
    /// <param name="is_max_k">
    /// When true, computes max-k; otherwise computes min-k.
    /// </param>
    /// <param name="reduction_input_size_override">
    /// 
    /// When set to a positive value, it overrides the size determined by
    /// `input[reduction_dim]` for evaluating the recall. This option is useful when
    /// the given `input` is only a subset of the overall computation in SPMD or
    /// distributed pipelines, where the true input size cannot be deferred by the
    /// `input` shape.
    /// 
    /// </param>
    /// <param name="aggregate_to_topk">
    /// 
    /// When true, aggregates approximate results to top-k. When false, returns the
    /// approximate results. The number of the approximate results is implementation
    /// defined and is greater equals to the specified `k`.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] approx_top_k(Tensor input, int k = 0, int reduction_dimension = -1, float recall_target = 0.95f, bool is_max_k = true, int reduction_input_size_override = -1, bool aggregate_to_topk = true, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ApproxTopK", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { ["k"] = k, ["reduction_dimension"] = reduction_dimension, ["recall_target"] = recall_target, ["is_max_k"] = is_max_k, ["reduction_input_size_override"] = reduction_input_size_override, ["aggregate_to_topk"] = aggregate_to_topk } });
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
                return approx_top_k_eager_fallback(input, k: k, reduction_dimension: reduction_dimension, recall_target: recall_target, is_max_k: is_max_k, reduction_input_size_override: reduction_input_size_override, aggregate_to_topk: aggregate_to_topk, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["k"] = k;
        keywords["reduction_dimension"] = reduction_dimension;
        keywords["recall_target"] = recall_target;
        keywords["is_max_k"] = is_max_k;
        keywords["reduction_input_size_override"] = reduction_input_size_override;
        keywords["aggregate_to_topk"] = aggregate_to_topk;
        var _op = tf.OpDefLib._apply_op_helper("ApproxTopK", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "k", _op._get_attr_int("k"), "reduction_dimension", _op._get_attr_int("reduction_dimension"), "recall_target", _op.get_attr("recall_target"), "is_max_k", _op._get_attr_bool("is_max_k"), "reduction_input_size_override", _op._get_attr_int("reduction_input_size_override"), "aggregate_to_topk", _op._get_attr_bool("aggregate_to_topk"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("ApproxTopK", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] approx_top_k_eager_fallback(Tensor input, int k, int reduction_dimension, float recall_target, bool is_max_k, int reduction_input_size_override, bool aggregate_to_topk, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "k", k, "reduction_dimension", reduction_dimension, "recall_target", recall_target, "is_max_k", is_max_k, "reduction_input_size_override", reduction_input_size_override, "aggregate_to_topk", aggregate_to_topk, "T", input.dtype };
        var _result = _execute.execute("ApproxTopK", 2, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ApproxTopK", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Performs average pooling on the input.
    /// </summary>
    /// <remarks>
    /// 
    /// Each entry in `output` is the mean of the corresponding size `ksize`
    /// window in `value`.
    /// 
    /// </remarks>
    /// <param name="value"></param>
    /// <param name="ksize">
    /// 
    /// The size of the sliding window for each dimension of `value`.
    /// 
    /// </param>
    /// <param name="strides">
    /// 
    /// The stride of the sliding window for each dimension of `value`.
    /// 
    /// </param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <param name="data_format">
    /// 
    /// Specify the data format of the input and output data. With the
    /// default format "NHWC", the data is stored in the order of:
    ///     [batch, in_height, in_width, in_channels].
    /// Alternatively, the format could be "NCHW", the data storage order of:
    ///     [batch, in_channels, in_height, in_width].
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor avg_pool(Tensor value, int[] ksize, int[] strides, string padding, string data_format = "NHWC", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "AvgPool", name) { args = new object[] { value }, attrs = new Dictionary<string, object>() { ["ksize"] = ksize, ["strides"] = strides, ["padding"] = padding, ["data_format"] = data_format } });
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
                return avg_pool_eager_fallback(value, ksize: ksize, strides: strides, padding: padding, data_format: data_format, name: name, ctx: _ctx);
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
        keywords["value"] = value;
        keywords["ksize"] = ksize;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["data_format"] = data_format;
        var _op = tf.OpDefLib._apply_op_helper("AvgPool", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "ksize", _op.get_attr("ksize"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "data_format", _op.get_attr("data_format"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("AvgPool", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor avg_pool_eager_fallback(Tensor value, int[] ksize, int[] strides, string padding, string data_format, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { value };
        object[] _attrs = new object[] { "ksize", ksize, "strides", strides, "padding", padding, "data_format", data_format, "T", value.dtype };
        var _result = _execute.execute("AvgPool", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("AvgPool", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Performs 3D average pooling on the input.
    /// </summary>
    /// <remarks>
    /// 
    /// Each entry in `output` is the mean of the corresponding size `ksize` window in
    /// `value`.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="ksize">
    /// 
    /// 1-D tensor of length 5. The size of the window for each dimension of
    /// the input tensor. Must have `ksize[0] = ksize[4] = 1`.
    /// 
    /// </param>
    /// <param name="strides">
    /// 
    /// 1-D tensor of length 5. The stride of the sliding window for each
    /// dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    /// 
    /// </param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <param name="data_format">
    /// 
    /// The data format of the input and output data. With the
    /// default format "NDHWC", the data is stored in the order of:
    ///     [batch, in_depth, in_height, in_width, in_channels].
    /// Alternatively, the format could be "NCDHW", the data storage order is:
    ///     [batch, in_channels, in_depth, in_height, in_width].
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor avg_pool3d(Tensor input, int[] ksize, int[] strides, string padding, string data_format = "NDHWC", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "AvgPool3D", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { ["ksize"] = ksize, ["strides"] = strides, ["padding"] = padding, ["data_format"] = data_format } });
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
                return avg_pool3d_eager_fallback(input, ksize: ksize, strides: strides, padding: padding, data_format: data_format, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        if (data_format is null)
        {
            data_format = "NDHWC";
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["ksize"] = ksize;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["data_format"] = data_format;
        var _op = tf.OpDefLib._apply_op_helper("AvgPool3D", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "ksize", _op.get_attr("ksize"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "data_format", _op.get_attr("data_format"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("AvgPool3D", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor avg_pool3d_eager_fallback(Tensor input, int[] ksize, int[] strides, string padding, string data_format, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "ksize", ksize, "strides", strides, "padding", padding, "data_format", data_format, "T", input.dtype };
        var _result = _execute.execute("AvgPool3D", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("AvgPool3D", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes gradients of average pooling function.
    /// </summary>
    /// <param name="orig_input_shape"></param>
    /// <param name="grad"></param>
    /// <param name="ksize">
    /// 
    /// 1-D tensor of length 5. The size of the window for each dimension of
    /// the input tensor. Must have `ksize[0] = ksize[4] = 1`.
    /// 
    /// </param>
    /// <param name="strides">
    /// 
    /// 1-D tensor of length 5. The stride of the sliding window for each
    /// dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    /// 
    /// </param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <param name="data_format">
    /// 
    /// The data format of the input and output data. With the
    /// default format "NDHWC", the data is stored in the order of:
    ///     [batch, in_depth, in_height, in_width, in_channels].
    /// Alternatively, the format could be "NCDHW", the data storage order is:
    ///     [batch, in_channels, in_depth, in_height, in_width].
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor avg_pool3d_grad(Tensor orig_input_shape, Tensor grad, int[] ksize, int[] strides, string padding, string data_format = "NDHWC", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "AvgPool3DGrad", name) { args = new object[] { orig_input_shape, grad }, attrs = new Dictionary<string, object>() { ["ksize"] = ksize, ["strides"] = strides, ["padding"] = padding, ["data_format"] = data_format } });
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
                return avg_pool3d_grad_eager_fallback(orig_input_shape, grad, ksize: ksize, strides: strides, padding: padding, data_format: data_format, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        if (data_format is null)
        {
            data_format = "NDHWC";
        }
        Dictionary<string, object> keywords = new();
        keywords["orig_input_shape"] = orig_input_shape;
        keywords["grad"] = grad;
        keywords["ksize"] = ksize;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["data_format"] = data_format;
        var _op = tf.OpDefLib._apply_op_helper("AvgPool3DGrad", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "ksize", _op.get_attr("ksize"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "data_format", _op.get_attr("data_format"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("AvgPool3DGrad", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor avg_pool3d_grad_eager_fallback(Tensor orig_input_shape, Tensor grad, int[] ksize, int[] strides, string padding, string data_format, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { orig_input_shape, grad };
        object[] _attrs = new object[] { "ksize", ksize, "strides", strides, "padding", padding, "data_format", data_format, "T", grad.dtype };
        var _result = _execute.execute("AvgPool3DGrad", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("AvgPool3DGrad", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes gradients of the average pooling function.
    /// </summary>
    /// <param name="orig_input_shape"></param>
    /// <param name="grad"></param>
    /// <param name="ksize">
    /// 
    /// The size of the sliding window for each dimension of the input.
    /// 
    /// </param>
    /// <param name="strides">
    /// 
    /// The stride of the sliding window for each dimension of the input.
    /// 
    /// </param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <param name="data_format">
    /// 
    /// Specify the data format of the input and output data. With the
    /// default format "NHWC", the data is stored in the order of:
    ///     [batch, in_height, in_width, in_channels].
    /// Alternatively, the format could be "NCHW", the data storage order of:
    ///     [batch, in_channels, in_height, in_width].
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor avg_pool_grad(Tensor orig_input_shape, Tensor grad, int[] ksize, int[] strides, string padding, string data_format = "NHWC", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "AvgPoolGrad", name) { args = new object[] { orig_input_shape, grad }, attrs = new Dictionary<string, object>() { ["ksize"] = ksize, ["strides"] = strides, ["padding"] = padding, ["data_format"] = data_format } });
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
                return avg_pool_grad_eager_fallback(orig_input_shape, grad, ksize: ksize, strides: strides, padding: padding, data_format: data_format, name: name, ctx: _ctx);
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
        keywords["orig_input_shape"] = orig_input_shape;
        keywords["grad"] = grad;
        keywords["ksize"] = ksize;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["data_format"] = data_format;
        var _op = tf.OpDefLib._apply_op_helper("AvgPoolGrad", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "ksize", _op.get_attr("ksize"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "data_format", _op.get_attr("data_format"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("AvgPoolGrad", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor avg_pool_grad_eager_fallback(Tensor orig_input_shape, Tensor grad, int[] ksize, int[] strides, string padding, string data_format, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { orig_input_shape, grad };
        object[] _attrs = new object[] { "ksize", ksize, "strides", strides, "padding", padding, "data_format", data_format, "T", grad.dtype };
        var _result = _execute.execute("AvgPoolGrad", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("AvgPoolGrad", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Batch normalization.
    /// </summary>
    /// <remarks>
    /// 
    /// This op is deprecated. Prefer `tf.nn.batch_normalization`.
    /// 
    /// </remarks>
    /// <param name="t"></param>
    /// <param name="m"></param>
    /// <param name="v"></param>
    /// <param name="beta"></param>
    /// <param name="gamma"></param>
    /// <param name="variance_epsilon">
    /// 
    /// A small float number to avoid dividing by 0.
    /// 
    /// </param>
    /// <param name="scale_after_normalization">
    /// 
    /// A bool indicating whether the resulted tensor
    /// needs to be multiplied with gamma.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor batch_norm_with_global_normalization(Tensor t, Tensor m, Tensor v, Tensor beta, Tensor gamma, float variance_epsilon, bool scale_after_normalization, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "BatchNormWithGlobalNormalization", name) { args = new object[] { t, m, v, beta, gamma }, attrs = new Dictionary<string, object>() { ["variance_epsilon"] = variance_epsilon, ["scale_after_normalization"] = scale_after_normalization } });
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
                return batch_norm_with_global_normalization_eager_fallback(t, m, v, beta, gamma, variance_epsilon: variance_epsilon, scale_after_normalization: scale_after_normalization, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["t"] = t;
        keywords["m"] = m;
        keywords["v"] = v;
        keywords["beta"] = beta;
        keywords["gamma"] = gamma;
        keywords["variance_epsilon"] = variance_epsilon;
        keywords["scale_after_normalization"] = scale_after_normalization;
        var _op = tf.OpDefLib._apply_op_helper("BatchNormWithGlobalNormalization", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "variance_epsilon", _op.get_attr("variance_epsilon"), "scale_after_normalization", _op._get_attr_bool("scale_after_normalization") };
            _execute.record_gradient("BatchNormWithGlobalNormalization", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor batch_norm_with_global_normalization_eager_fallback(Tensor t, Tensor m, Tensor v, Tensor beta, Tensor gamma, float variance_epsilon, bool scale_after_normalization, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { t, m, v, beta, gamma };
        object[] _attrs = new object[] { "T", t.dtype, "variance_epsilon", variance_epsilon, "scale_after_normalization", scale_after_normalization };
        var _result = _execute.execute("BatchNormWithGlobalNormalization", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("BatchNormWithGlobalNormalization", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Gradients for batch normalization.
    /// </summary>
    /// <remarks>
    /// 
    /// This op is deprecated. See `tf.nn.batch_normalization`.
    /// 
    /// </remarks>
    /// <param name="t"></param>
    /// <param name="m"></param>
    /// <param name="v"></param>
    /// <param name="gamma"></param>
    /// <param name="backprop"></param>
    /// <param name="variance_epsilon">
    /// 
    /// A small float number to avoid dividing by 0.
    /// 
    /// </param>
    /// <param name="scale_after_normalization">
    /// 
    /// A bool indicating whether the resulted tensor
    /// needs to be multiplied with gamma.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] batch_norm_with_global_normalization_grad(Tensor t, Tensor m, Tensor v, Tensor gamma, Tensor backprop, float variance_epsilon, bool scale_after_normalization, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "BatchNormWithGlobalNormalizationGrad", name) { args = new object[] { t, m, v, gamma, backprop }, attrs = new Dictionary<string, object>() { ["variance_epsilon"] = variance_epsilon, ["scale_after_normalization"] = scale_after_normalization } });
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
                return batch_norm_with_global_normalization_grad_eager_fallback(t, m, v, gamma, backprop, variance_epsilon: variance_epsilon, scale_after_normalization: scale_after_normalization, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["t"] = t;
        keywords["m"] = m;
        keywords["v"] = v;
        keywords["gamma"] = gamma;
        keywords["backprop"] = backprop;
        keywords["variance_epsilon"] = variance_epsilon;
        keywords["scale_after_normalization"] = scale_after_normalization;
        var _op = tf.OpDefLib._apply_op_helper("BatchNormWithGlobalNormalizationGrad", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "variance_epsilon", _op.get_attr("variance_epsilon"), "scale_after_normalization", _op._get_attr_bool("scale_after_normalization") };
            _execute.record_gradient("BatchNormWithGlobalNormalizationGrad", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] batch_norm_with_global_normalization_grad_eager_fallback(Tensor t, Tensor m, Tensor v, Tensor gamma, Tensor backprop, float variance_epsilon, bool scale_after_normalization, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { t, m, v, gamma, backprop };
        object[] _attrs = new object[] { "T", t.dtype, "variance_epsilon", variance_epsilon, "scale_after_normalization", scale_after_normalization };
        var _result = _execute.execute("BatchNormWithGlobalNormalizationGrad", 5, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("BatchNormWithGlobalNormalizationGrad", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Adds `bias` to `value`.
    /// </summary>
    /// <remarks>
    /// 
    /// This is a special case of `tf.add` where `bias` is restricted to be 1-D.
    /// Broadcasting is supported, so `value` may have any number of dimensions.
    /// 
    /// </remarks>
    /// <param name="value"></param>
    /// <param name="bias"></param>
    /// <param name="data_format">
    /// 
    /// Specify the data format of the input and output data. With the
    /// default format "NHWC", the bias tensor will be added to the last dimension
    /// of the value tensor.
    /// Alternatively, the format could be "NCHW", the data storage order of:
    ///     [batch, in_channels, in_height, in_width].
    /// The tensor will be added to "in_channels", the third-to-the-last
    ///     dimension.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor bias_add(Tensor value, Tensor bias, string data_format = "NHWC", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "BiasAdd", name) { args = new object[] { value, bias }, attrs = new Dictionary<string, object>() { ["data_format"] = data_format } });
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
                return bias_add_eager_fallback(value, bias, data_format: data_format, name: name, ctx: _ctx);
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
        keywords["value"] = value;
        keywords["bias"] = bias;
        keywords["data_format"] = data_format;
        var _op = tf.OpDefLib._apply_op_helper("BiasAdd", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "data_format", _op.get_attr("data_format") };
            _execute.record_gradient("BiasAdd", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor bias_add_eager_fallback(Tensor value, Tensor bias, string data_format, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { value, bias };
        object[] _attrs = new object[] { "T", value.dtype, "data_format", data_format };
        var _result = _execute.execute("BiasAdd", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("BiasAdd", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// The backward operation for "BiasAdd" on the "bias" tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// It accumulates all the values from out_backprop into the feature dimension.
    /// For NHWC data format, the feature dimension is the last. For NCHW data format,
    /// the feature dimension is the third-to-last.
    /// 
    /// </remarks>
    /// <param name="out_backprop"></param>
    /// <param name="data_format">
    /// 
    /// Specify the data format of the input and output data. With the
    /// default format "NHWC", the bias tensor will be added to the last dimension
    /// of the value tensor.
    /// Alternatively, the format could be "NCHW", the data storage order of:
    ///     [batch, in_channels, in_height, in_width].
    /// The tensor will be added to "in_channels", the third-to-the-last
    ///     dimension.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor bias_add_grad(Tensor out_backprop, string data_format = "NHWC", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "BiasAddGrad", name) { args = new object[] { out_backprop }, attrs = new Dictionary<string, object>() { ["data_format"] = data_format } });
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
                return bias_add_grad_eager_fallback(out_backprop, data_format: data_format, name: name, ctx: _ctx);
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
        keywords["out_backprop"] = out_backprop;
        keywords["data_format"] = data_format;
        var _op = tf.OpDefLib._apply_op_helper("BiasAddGrad", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "data_format", _op.get_attr("data_format") };
            _execute.record_gradient("BiasAddGrad", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor bias_add_grad_eager_fallback(Tensor out_backprop, string data_format, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { out_backprop };
        object[] _attrs = new object[] { "T", out_backprop.dtype, "data_format", data_format };
        var _result = _execute.execute("BiasAddGrad", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("BiasAddGrad", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Adds `bias` to `value`.
    /// </summary>
    /// <remarks>
    /// 
    /// This is a deprecated version of BiasAdd and will be soon removed.
    /// 
    /// This is a special case of `tf.add` where `bias` is restricted to be 1-D.
    /// Broadcasting is supported, so `value` may have any number of dimensions.
    /// 
    /// </remarks>
    /// <param name="value"></param>
    /// <param name="bias"></param>
    /// <returns></returns>
    public static Tensor bias_add_v1(Tensor value, Tensor bias, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "BiasAddV1", name) { args = new object[] { value, bias }, attrs = new Dictionary<string, object>() { } });
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
                return bias_add_v1_eager_fallback(value, bias, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["value"] = value;
        keywords["bias"] = bias;
        var _op = tf.OpDefLib._apply_op_helper("BiasAddV1", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("BiasAddV1", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor bias_add_v1_eager_fallback(Tensor value, Tensor bias, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { value, bias };
        object[] _attrs = new object[] { "T", value.dtype };
        var _result = _execute.execute("BiasAddV1", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("BiasAddV1", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes a 2-D convolution given 4-D `input` and `filter` tensors.
    /// </summary>
    /// <remarks>
    /// 
    /// Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
    /// and a filter / kernel tensor of shape
    /// `[filter_height, filter_width, in_channels, out_channels]`, this op
    /// performs the following:
    /// 
    /// 1. Flattens the filter to a 2-D matrix with shape
    ///    `[filter_height * filter_width * in_channels, output_channels]`.
    /// 2. Extracts image patches from the input tensor to form a *virtual*
    ///    tensor of shape `[batch, out_height, out_width,
    ///    filter_height * filter_width * in_channels]`.
    /// 3. For each patch, right-multiplies the filter matrix and the image patch
    ///    vector.
    /// 
    /// In detail, with the default NHWC format,
    /// 
    ///     output[b, i, j, k] =
    ///         sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
    ///                         filter[di, dj, q, k]
    /// 
    /// Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
    /// horizontal and vertices strides, `strides = [1, stride, stride, 1]`.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="filter"></param>
    /// <param name="strides">
    /// 
    /// 1-D tensor of length 4.  The stride of the sliding window for each
    /// dimension of `input`. The dimension order is determined by the value of
    /// `data_format`, see below for details.
    /// 
    /// </param>
    /// <param name="use_cudnn_on_gpu"></param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <param name="explicit_paddings">
    /// 
    /// If `padding` is `"EXPLICIT"`, the list of explicit padding amounts. For the ith
    /// dimension, the amount of padding inserted before and after the dimension is
    /// `explicit_paddings[2 * i]` and `explicit_paddings[2 * i + 1]`, respectively. If
    /// `padding` is not `"EXPLICIT"`, `explicit_paddings` must be empty.
    /// 
    /// </param>
    /// <param name="data_format">
    /// 
    /// Specify the data format of the input and output data. With the
    /// default format "NHWC", the data is stored in the order of:
    ///     [batch, height, width, channels].
    /// Alternatively, the format could be "NCHW", the data storage order of:
    ///     [batch, channels, height, width].
    /// 
    /// </param>
    /// <param name="dilations">
    /// 
    /// 1-D tensor of length 4.  The dilation factor for each dimension of
    /// `input`. If set to k > 1, there will be k-1 skipped cells between each
    /// filter element on that dimension. The dimension order is determined by the
    /// value of `data_format`, see above for details. Dilations in the batch and
    /// depth dimensions must be 1.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor conv2d(Tensor input, Tensor filter, int[] strides, string padding, bool use_cudnn_on_gpu = true, int[] explicit_paddings = null, string data_format = "NHWC", int[] dilations = null, string? name = null)
    {
        var _ctx = tf.Context;
        if (explicit_paddings is null)
        {
            explicit_paddings = new int[] { };
        }
        if (dilations is null)
        {
            dilations = new int[] { 1, 1, 1, 1 };
        }
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Conv2D", name) { args = new object[] { input, filter }, attrs = new Dictionary<string, object>() { ["strides"] = strides, ["use_cudnn_on_gpu"] = use_cudnn_on_gpu, ["padding"] = padding, ["explicit_paddings"] = explicit_paddings, ["data_format"] = data_format, ["dilations"] = dilations } });
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
                return conv2d_eager_fallback(input, filter, strides: strides, use_cudnn_on_gpu: use_cudnn_on_gpu, padding: padding, explicit_paddings: explicit_paddings, data_format: data_format, dilations: dilations, name: name, ctx: _ctx);
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
        keywords["filter"] = filter;
        keywords["strides"] = strides;
        keywords["use_cudnn_on_gpu"] = use_cudnn_on_gpu;
        keywords["padding"] = padding;
        keywords["explicit_paddings"] = explicit_paddings;
        keywords["data_format"] = data_format;
        keywords["dilations"] = dilations;
        var _op = tf.OpDefLib._apply_op_helper("Conv2D", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "strides", _op.get_attr("strides"), "use_cudnn_on_gpu", _op._get_attr_bool("use_cudnn_on_gpu"), "padding", _op.get_attr("padding"), "explicit_paddings", _op.get_attr("explicit_paddings"), "data_format", _op.get_attr("data_format"), "dilations", _op.get_attr("dilations") };
            _execute.record_gradient("Conv2D", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor conv2d_eager_fallback(Tensor input, Tensor filter, int[] strides, bool use_cudnn_on_gpu, string padding, int[] explicit_paddings, string data_format, int[] dilations, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, filter };
        object[] _attrs = new object[] { "T", input.dtype, "strides", strides, "use_cudnn_on_gpu", use_cudnn_on_gpu, "padding", padding, "explicit_paddings", explicit_paddings, "data_format", data_format, "dilations", dilations };
        var _result = _execute.execute("Conv2D", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Conv2D", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the gradients of convolution with respect to the filter.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="filter_sizes"></param>
    /// <param name="out_backprop"></param>
    /// <param name="strides">
    /// 
    /// The stride of the sliding window for each dimension of the input
    /// of the convolution. Must be in the same order as the dimension specified with
    /// format.
    /// 
    /// </param>
    /// <param name="use_cudnn_on_gpu"></param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <param name="explicit_paddings">
    /// 
    /// If `padding` is `"EXPLICIT"`, the list of explicit padding amounts. For the ith
    /// dimension, the amount of padding inserted before and after the dimension is
    /// `explicit_paddings[2 * i]` and `explicit_paddings[2 * i + 1]`, respectively. If
    /// `padding` is not `"EXPLICIT"`, `explicit_paddings` must be empty.
    /// 
    /// </param>
    /// <param name="data_format">
    /// 
    /// Specify the data format of the input and output data. With the
    /// default format "NHWC", the data is stored in the order of:
    ///     [batch, in_height, in_width, in_channels].
    /// Alternatively, the format could be "NCHW", the data storage order of:
    ///     [batch, in_channels, in_height, in_width].
    /// 
    /// </param>
    /// <param name="dilations">
    /// 
    /// 1-D tensor of length 4.  The dilation factor for each dimension of
    /// `input`. If set to k > 1, there will be k-1 skipped cells between each filter
    /// element on that dimension. The dimension order is determined by the value of
    /// `data_format`, see above for details. Dilations in the batch and depth
    /// dimensions must be 1.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor conv2d_backprop_filter(Tensor input, Tensor filter_sizes, Tensor out_backprop, int[] strides, string padding, bool use_cudnn_on_gpu = true, int[] explicit_paddings = null, string data_format = "NHWC", int[] dilations = null, string? name = null)
    {
        var _ctx = tf.Context;
        if (explicit_paddings is null)
        {
            explicit_paddings = new int[] { };
        }
        if (dilations is null)
        {
            dilations = new int[] { 1, 1, 1, 1 };
        }
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Conv2DBackpropFilter", name) { args = new object[] { input, filter_sizes, out_backprop }, attrs = new Dictionary<string, object>() { ["strides"] = strides, ["use_cudnn_on_gpu"] = use_cudnn_on_gpu, ["padding"] = padding, ["explicit_paddings"] = explicit_paddings, ["data_format"] = data_format, ["dilations"] = dilations } });
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
                return conv2d_backprop_filter_eager_fallback(input, filter_sizes, out_backprop, strides: strides, use_cudnn_on_gpu: use_cudnn_on_gpu, padding: padding, explicit_paddings: explicit_paddings, data_format: data_format, dilations: dilations, name: name, ctx: _ctx);
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
        keywords["filter_sizes"] = filter_sizes;
        keywords["out_backprop"] = out_backprop;
        keywords["strides"] = strides;
        keywords["use_cudnn_on_gpu"] = use_cudnn_on_gpu;
        keywords["padding"] = padding;
        keywords["explicit_paddings"] = explicit_paddings;
        keywords["data_format"] = data_format;
        keywords["dilations"] = dilations;
        var _op = tf.OpDefLib._apply_op_helper("Conv2DBackpropFilter", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "strides", _op.get_attr("strides"), "use_cudnn_on_gpu", _op._get_attr_bool("use_cudnn_on_gpu"), "padding", _op.get_attr("padding"), "explicit_paddings", _op.get_attr("explicit_paddings"), "data_format", _op.get_attr("data_format"), "dilations", _op.get_attr("dilations") };
            _execute.record_gradient("Conv2DBackpropFilter", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor conv2d_backprop_filter_eager_fallback(Tensor input, Tensor filter_sizes, Tensor out_backprop, int[] strides, bool use_cudnn_on_gpu, string padding, int[] explicit_paddings, string data_format, int[] dilations, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, filter_sizes, out_backprop };
        object[] _attrs = new object[] { "T", input.dtype, "strides", strides, "use_cudnn_on_gpu", use_cudnn_on_gpu, "padding", padding, "explicit_paddings", explicit_paddings, "data_format", data_format, "dilations", dilations };
        var _result = _execute.execute("Conv2DBackpropFilter", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Conv2DBackpropFilter", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the gradients of convolution with respect to the input.
    /// </summary>
    /// <param name="input_sizes"></param>
    /// <param name="filter"></param>
    /// <param name="out_backprop"></param>
    /// <param name="strides">
    /// 
    /// The stride of the sliding window for each dimension of the input
    /// of the convolution. Must be in the same order as the dimension specified with
    /// format.
    /// 
    /// </param>
    /// <param name="use_cudnn_on_gpu"></param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <param name="explicit_paddings">
    /// 
    /// If `padding` is `"EXPLICIT"`, the list of explicit padding amounts. For the ith
    /// dimension, the amount of padding inserted before and after the dimension is
    /// `explicit_paddings[2 * i]` and `explicit_paddings[2 * i + 1]`, respectively. If
    /// `padding` is not `"EXPLICIT"`, `explicit_paddings` must be empty.
    /// 
    /// </param>
    /// <param name="data_format">
    /// 
    /// Specify the data format of the input and output data. With the
    /// default format "NHWC", the data is stored in the order of:
    ///     [batch, in_height, in_width, in_channels].
    /// Alternatively, the format could be "NCHW", the data storage order of:
    ///     [batch, in_channels, in_height, in_width].
    /// 
    /// </param>
    /// <param name="dilations">
    /// 
    /// 1-D tensor of length 4.  The dilation factor for each dimension of
    /// `input`. If set to k > 1, there will be k-1 skipped cells between each filter
    /// element on that dimension. The dimension order is determined by the value of
    /// `data_format`, see above for details. Dilations in the batch and depth
    /// dimensions must be 1.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor conv2d_backprop_input(Tensor input_sizes, Tensor filter, Tensor out_backprop, int[] strides, string padding, bool use_cudnn_on_gpu = true, int[] explicit_paddings = null, string data_format = "NHWC", int[] dilations = null, string? name = null)
    {
        var _ctx = tf.Context;
        if (explicit_paddings is null)
        {
            explicit_paddings = new int[] { };
        }
        if (dilations is null)
        {
            dilations = new int[] { 1, 1, 1, 1 };
        }
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Conv2DBackpropInput", name) { args = new object[] { input_sizes, filter, out_backprop }, attrs = new Dictionary<string, object>() { ["strides"] = strides, ["use_cudnn_on_gpu"] = use_cudnn_on_gpu, ["padding"] = padding, ["explicit_paddings"] = explicit_paddings, ["data_format"] = data_format, ["dilations"] = dilations } });
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
                return conv2d_backprop_input_eager_fallback(input_sizes, filter, out_backprop, strides: strides, use_cudnn_on_gpu: use_cudnn_on_gpu, padding: padding, explicit_paddings: explicit_paddings, data_format: data_format, dilations: dilations, name: name, ctx: _ctx);
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
        keywords["input_sizes"] = input_sizes;
        keywords["filter"] = filter;
        keywords["out_backprop"] = out_backprop;
        keywords["strides"] = strides;
        keywords["use_cudnn_on_gpu"] = use_cudnn_on_gpu;
        keywords["padding"] = padding;
        keywords["explicit_paddings"] = explicit_paddings;
        keywords["data_format"] = data_format;
        keywords["dilations"] = dilations;
        var _op = tf.OpDefLib._apply_op_helper("Conv2DBackpropInput", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "strides", _op.get_attr("strides"), "use_cudnn_on_gpu", _op._get_attr_bool("use_cudnn_on_gpu"), "padding", _op.get_attr("padding"), "explicit_paddings", _op.get_attr("explicit_paddings"), "data_format", _op.get_attr("data_format"), "dilations", _op.get_attr("dilations") };
            _execute.record_gradient("Conv2DBackpropInput", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor conv2d_backprop_input_eager_fallback(Tensor input_sizes, Tensor filter, Tensor out_backprop, int[] strides, bool use_cudnn_on_gpu, string padding, int[] explicit_paddings, string data_format, int[] dilations, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input_sizes, filter, out_backprop };
        object[] _attrs = new object[] { "T", filter.dtype, "strides", strides, "use_cudnn_on_gpu", use_cudnn_on_gpu, "padding", padding, "explicit_paddings", explicit_paddings, "data_format", data_format, "dilations", dilations };
        var _result = _execute.execute("Conv2DBackpropInput", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Conv2DBackpropInput", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes a 3-D convolution given 5-D `input` and `filter` tensors.
    /// </summary>
    /// <remarks>
    /// 
    /// In signal processing, cross-correlation is a measure of similarity of
    /// two waveforms as a function of a time-lag applied to one of them. This
    /// is also known as a sliding dot product or sliding inner-product.
    /// 
    /// Our Conv3D implements a form of cross-correlation.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="filter"></param>
    /// <param name="strides">
    /// 
    /// 1-D tensor of length 5. The stride of the sliding window for each
    /// dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    /// 
    /// </param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <param name="data_format">
    /// 
    /// The data format of the input and output data. With the
    /// default format "NDHWC", the data is stored in the order of:
    ///     [batch, in_depth, in_height, in_width, in_channels].
    /// Alternatively, the format could be "NCDHW", the data storage order is:
    ///     [batch, in_channels, in_depth, in_height, in_width].
    /// 
    /// </param>
    /// <param name="dilations">
    /// 
    /// 1-D tensor of length 5.  The dilation factor for each dimension of
    /// `input`. If set to k > 1, there will be k-1 skipped cells between each
    /// filter element on that dimension. The dimension order is determined by the
    /// value of `data_format`, see above for details. Dilations in the batch and
    /// depth dimensions must be 1.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor conv3d(Tensor input, Tensor filter, int[] strides, string padding, string data_format = "NDHWC", int[] dilations = null, string? name = null)
    {
        var _ctx = tf.Context;
        if (dilations is null)
        {
            dilations = new int[] { 1, 1, 1, 1, 1 };
        }
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Conv3D", name) { args = new object[] { input, filter }, attrs = new Dictionary<string, object>() { ["strides"] = strides, ["padding"] = padding, ["data_format"] = data_format, ["dilations"] = dilations } });
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
                return conv3d_eager_fallback(input, filter, strides: strides, padding: padding, data_format: data_format, dilations: dilations, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        if (data_format is null)
        {
            data_format = "NDHWC";
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["filter"] = filter;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["data_format"] = data_format;
        keywords["dilations"] = dilations;
        var _op = tf.OpDefLib._apply_op_helper("Conv3D", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "data_format", _op.get_attr("data_format"), "dilations", _op.get_attr("dilations") };
            _execute.record_gradient("Conv3D", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor conv3d_eager_fallback(Tensor input, Tensor filter, int[] strides, string padding, string data_format, int[] dilations, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, filter };
        object[] _attrs = new object[] { "T", input.dtype, "strides", strides, "padding", padding, "data_format", data_format, "dilations", dilations };
        var _result = _execute.execute("Conv3D", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Conv3D", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the gradients of 3-D convolution with respect to the filter.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="filter"></param>
    /// <param name="out_backprop"></param>
    /// <param name="strides">
    /// 
    /// 1-D tensor of length 5. The stride of the sliding window for each
    /// dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    /// 
    /// </param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <param name="dilations"></param>
    /// <returns></returns>
    public static Tensor conv3d_backprop_filter(Tensor input, Tensor filter, Tensor out_backprop, int[] strides, string padding, int[] dilations = null, string? name = null)
    {
        var _ctx = tf.Context;
        if (dilations is null)
        {
            dilations = new int[] { 1, 1, 1, 1, 1 };
        }
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Conv3DBackpropFilter", name) { args = new object[] { input, filter, out_backprop }, attrs = new Dictionary<string, object>() { ["strides"] = strides, ["padding"] = padding, ["dilations"] = dilations } });
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
                return conv3d_backprop_filter_eager_fallback(input, filter, out_backprop, strides: strides, padding: padding, dilations: dilations, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["filter"] = filter;
        keywords["out_backprop"] = out_backprop;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["dilations"] = dilations;
        var _op = tf.OpDefLib._apply_op_helper("Conv3DBackpropFilter", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "dilations", _op.get_attr("dilations") };
            _execute.record_gradient("Conv3DBackpropFilter", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor conv3d_backprop_filter_eager_fallback(Tensor input, Tensor filter, Tensor out_backprop, int[] strides, string padding, int[] dilations, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, filter, out_backprop };
        object[] _attrs = new object[] { "T", input.dtype, "strides", strides, "padding", padding, "dilations", dilations };
        var _result = _execute.execute("Conv3DBackpropFilter", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Conv3DBackpropFilter", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the gradients of 3-D convolution with respect to the filter.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="filter_sizes"></param>
    /// <param name="out_backprop"></param>
    /// <param name="strides">
    /// 
    /// 1-D tensor of length 5. The stride of the sliding window for each
    /// dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    /// 
    /// </param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <param name="data_format">
    /// 
    /// The data format of the input and output data. With the
    /// default format "NDHWC", the data is stored in the order of:
    ///     [batch, in_depth, in_height, in_width, in_channels].
    /// Alternatively, the format could be "NCDHW", the data storage order is:
    ///     [batch, in_channels, in_depth, in_height, in_width].
    /// 
    /// </param>
    /// <param name="dilations">
    /// 
    /// 1-D tensor of length 5.  The dilation factor for each dimension of
    /// `input`. If set to k > 1, there will be k-1 skipped cells between each
    /// filter element on that dimension. The dimension order is determined by the
    /// value of `data_format`, see above for details. Dilations in the batch and
    /// depth dimensions must be 1.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor conv3d_backprop_filter_v2(Tensor input, Tensor filter_sizes, Tensor out_backprop, int[] strides, string padding, string data_format = "NDHWC", int[] dilations = null, string? name = null)
    {
        var _ctx = tf.Context;
        if (dilations is null)
        {
            dilations = new int[] { 1, 1, 1, 1, 1 };
        }
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Conv3DBackpropFilterV2", name) { args = new object[] { input, filter_sizes, out_backprop }, attrs = new Dictionary<string, object>() { ["strides"] = strides, ["padding"] = padding, ["data_format"] = data_format, ["dilations"] = dilations } });
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
                return conv3d_backprop_filter_v2_eager_fallback(input, filter_sizes, out_backprop, strides: strides, padding: padding, data_format: data_format, dilations: dilations, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        if (data_format is null)
        {
            data_format = "NDHWC";
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["filter_sizes"] = filter_sizes;
        keywords["out_backprop"] = out_backprop;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["data_format"] = data_format;
        keywords["dilations"] = dilations;
        var _op = tf.OpDefLib._apply_op_helper("Conv3DBackpropFilterV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "data_format", _op.get_attr("data_format"), "dilations", _op.get_attr("dilations") };
            _execute.record_gradient("Conv3DBackpropFilterV2", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor conv3d_backprop_filter_v2_eager_fallback(Tensor input, Tensor filter_sizes, Tensor out_backprop, int[] strides, string padding, string data_format, int[] dilations, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, filter_sizes, out_backprop };
        object[] _attrs = new object[] { "T", input.dtype, "strides", strides, "padding", padding, "data_format", data_format, "dilations", dilations };
        var _result = _execute.execute("Conv3DBackpropFilterV2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Conv3DBackpropFilterV2", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the gradients of 3-D convolution with respect to the input.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="filter"></param>
    /// <param name="out_backprop"></param>
    /// <param name="strides">
    /// 
    /// 1-D tensor of length 5. The stride of the sliding window for each
    /// dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    /// 
    /// </param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <param name="dilations"></param>
    /// <returns></returns>
    public static Tensor conv3d_backprop_input(Tensor input, Tensor filter, Tensor out_backprop, int[] strides, string padding, int[] dilations = null, string? name = null)
    {
        var _ctx = tf.Context;
        if (dilations is null)
        {
            dilations = new int[] { 1, 1, 1, 1, 1 };
        }
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Conv3DBackpropInput", name) { args = new object[] { input, filter, out_backprop }, attrs = new Dictionary<string, object>() { ["strides"] = strides, ["padding"] = padding, ["dilations"] = dilations } });
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
                return conv3d_backprop_input_eager_fallback(input, filter, out_backprop, strides: strides, padding: padding, dilations: dilations, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["filter"] = filter;
        keywords["out_backprop"] = out_backprop;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["dilations"] = dilations;
        var _op = tf.OpDefLib._apply_op_helper("Conv3DBackpropInput", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "dilations", _op.get_attr("dilations") };
            _execute.record_gradient("Conv3DBackpropInput", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor conv3d_backprop_input_eager_fallback(Tensor input, Tensor filter, Tensor out_backprop, int[] strides, string padding, int[] dilations, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, filter, out_backprop };
        object[] _attrs = new object[] { "T", input.dtype, "strides", strides, "padding", padding, "dilations", dilations };
        var _result = _execute.execute("Conv3DBackpropInput", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Conv3DBackpropInput", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the gradients of 3-D convolution with respect to the input.
    /// </summary>
    /// <param name="input_sizes"></param>
    /// <param name="filter"></param>
    /// <param name="out_backprop"></param>
    /// <param name="strides">
    /// 
    /// 1-D tensor of length 5. The stride of the sliding window for each
    /// dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    /// 
    /// </param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <param name="data_format">
    /// 
    /// The data format of the input and output data. With the
    /// default format "NDHWC", the data is stored in the order of:
    ///     [batch, in_depth, in_height, in_width, in_channels].
    /// Alternatively, the format could be "NCDHW", the data storage order is:
    ///     [batch, in_channels, in_depth, in_height, in_width].
    /// 
    /// </param>
    /// <param name="dilations">
    /// 
    /// 1-D tensor of length 5.  The dilation factor for each dimension of
    /// `input`. If set to k > 1, there will be k-1 skipped cells between each
    /// filter element on that dimension. The dimension order is determined by the
    /// value of `data_format`, see above for details. Dilations in the batch and
    /// depth dimensions must be 1.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor conv3d_backprop_input_v2(Tensor input_sizes, Tensor filter, Tensor out_backprop, int[] strides, string padding, string data_format = "NDHWC", int[] dilations = null, string? name = null)
    {
        var _ctx = tf.Context;
        if (dilations is null)
        {
            dilations = new int[] { 1, 1, 1, 1, 1 };
        }
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Conv3DBackpropInputV2", name) { args = new object[] { input_sizes, filter, out_backprop }, attrs = new Dictionary<string, object>() { ["strides"] = strides, ["padding"] = padding, ["data_format"] = data_format, ["dilations"] = dilations } });
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
                return conv3d_backprop_input_v2_eager_fallback(input_sizes, filter, out_backprop, strides: strides, padding: padding, data_format: data_format, dilations: dilations, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        if (data_format is null)
        {
            data_format = "NDHWC";
        }
        Dictionary<string, object> keywords = new();
        keywords["input_sizes"] = input_sizes;
        keywords["filter"] = filter;
        keywords["out_backprop"] = out_backprop;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["data_format"] = data_format;
        keywords["dilations"] = dilations;
        var _op = tf.OpDefLib._apply_op_helper("Conv3DBackpropInputV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "data_format", _op.get_attr("data_format"), "dilations", _op.get_attr("dilations"), "Tshape", _op._get_attr_type("Tshape") };
            _execute.record_gradient("Conv3DBackpropInputV2", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor conv3d_backprop_input_v2_eager_fallback(Tensor input_sizes, Tensor filter, Tensor out_backprop, int[] strides, string padding, string data_format, int[] dilations, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input_sizes, filter, out_backprop };
        object[] _attrs = new object[] { "T", filter.dtype, "strides", strides, "padding", padding, "data_format", data_format, "dilations", dilations, "Tshape", input_sizes.dtype };
        var _result = _execute.execute("Conv3DBackpropInputV2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Conv3DBackpropInputV2", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns the dimension index in the destination data format given the one in
    /// </summary>
    /// <remarks>
    /// 
    /// the source data format.
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="src_format">
    /// 
    /// source data format.
    /// 
    /// </param>
    /// <param name="dst_format">
    /// 
    /// destination data format.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor data_format_dim_map(Tensor x, string src_format = "NHWC", string dst_format = "NCHW", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "DataFormatDimMap", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { ["src_format"] = src_format, ["dst_format"] = dst_format } });
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
                return data_format_dim_map_eager_fallback(x, src_format: src_format, dst_format: dst_format, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        if (src_format is null)
        {
            src_format = "NHWC";
        }
        if (dst_format is null)
        {
            dst_format = "NCHW";
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["src_format"] = src_format;
        keywords["dst_format"] = dst_format;
        var _op = tf.OpDefLib._apply_op_helper("DataFormatDimMap", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "src_format", _op.get_attr("src_format"), "dst_format", _op.get_attr("dst_format") };
            _execute.record_gradient("DataFormatDimMap", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor data_format_dim_map_eager_fallback(Tensor x, string src_format, string dst_format, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype, "src_format", src_format, "dst_format", dst_format };
        var _result = _execute.execute("DataFormatDimMap", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("DataFormatDimMap", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Permute input tensor from `src_format` to `dst_format`.
    /// </summary>
    /// <remarks>
    /// 
    /// Given source and destination format strings of length n=4 or 5, the input
    /// tensor must be a vector of size n or n-2, or a 2D tensor of shape
    /// (n, 2) or (n-2, 2).
    /// 
    /// If the first dimension of the input tensor is n-2, it is assumed that
    /// non-spatial dimensions are omitted (i.e `N`, `C`).
    /// 
    /// For example, with `src_format` of `NHWC`, `dst_format` of `NCHW`, and input:
    /// ```
    /// [1, 2, 3, 4]
    /// ```
    /// , the output will be:
    /// ```
    /// [1, 4, 2, 3]
    /// ```
    /// With `src_format` of `NDHWC`, `dst_format` of `NCDHW`, and input:
    /// ```
    /// [[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]]
    /// ```
    /// , the output will be:
    /// ```
    /// [[1, 6], [5, 10], [2, 7], [3, 8], [4, 9]]
    /// ```
    /// With `src_format` of `NHWC`, `dst_format` of `NCHW`, and input:
    /// ```
    /// [1, 2]
    /// ```
    /// , the output will be:
    /// ```
    /// [1, 2]
    /// ```
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="src_format">
    /// 
    /// source data format.
    /// 
    /// </param>
    /// <param name="dst_format">
    /// 
    /// destination data format.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor data_format_vec_permute(Tensor x, string src_format = "NHWC", string dst_format = "NCHW", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "DataFormatVecPermute", name) { args = new object[] { x }, attrs = new Dictionary<string, object>() { ["src_format"] = src_format, ["dst_format"] = dst_format } });
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
                return data_format_vec_permute_eager_fallback(x, src_format: src_format, dst_format: dst_format, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        if (src_format is null)
        {
            src_format = "NHWC";
        }
        if (dst_format is null)
        {
            dst_format = "NCHW";
        }
        Dictionary<string, object> keywords = new();
        keywords["x"] = x;
        keywords["src_format"] = src_format;
        keywords["dst_format"] = dst_format;
        var _op = tf.OpDefLib._apply_op_helper("DataFormatVecPermute", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "src_format", _op.get_attr("src_format"), "dst_format", _op.get_attr("dst_format") };
            _execute.record_gradient("DataFormatVecPermute", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor data_format_vec_permute_eager_fallback(Tensor x, string src_format, string dst_format, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x };
        object[] _attrs = new object[] { "T", x.dtype, "src_format", src_format, "dst_format", dst_format };
        var _result = _execute.execute("DataFormatVecPermute", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("DataFormatVecPermute", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes a 2-D depthwise convolution given 4-D `input` and `filter` tensors.
    /// </summary>
    /// <remarks>
    /// 
    /// Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
    /// and a filter / kernel tensor of shape
    /// `[filter_height, filter_width, in_channels, channel_multiplier]`, containing
    /// `in_channels` convolutional filters of depth 1, `depthwise_conv2d` applies
    /// a different filter to each input channel (expanding from 1 channel to
    /// `channel_multiplier` channels for each), then concatenates the results
    /// together. Thus, the output has `in_channels * channel_multiplier` channels.
    /// 
    /// ```
    /// for k in 0..in_channels-1
    ///   for q in 0..channel_multiplier-1
    ///     output[b, i, j, k * channel_multiplier + q] =
    ///       sum_{di, dj} input[b, strides[1] * i + di, strides[2] * j + dj, k] *
    ///                         filter[di, dj, k, q]
    /// ```
    /// 
    /// Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
    /// horizontal and vertices strides, `strides = [1, stride, stride, 1]`.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="filter"></param>
    /// <param name="strides">
    /// 
    /// 1-D of length 4.  The stride of the sliding window for each dimension
    /// of `input`.
    /// 
    /// </param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <param name="explicit_paddings"></param>
    /// <param name="data_format">
    /// 
    /// Specify the data format of the input and output data. With the
    /// default format "NHWC", the data is stored in the order of:
    ///     [batch, height, width, channels].
    /// Alternatively, the format could be "NCHW", the data storage order of:
    ///     [batch, channels, height, width].
    /// 
    /// </param>
    /// <param name="dilations">
    /// 
    /// 1-D tensor of length 4.  The dilation factor for each dimension of
    /// `input`. If set to k > 1, there will be k-1 skipped cells between each filter
    /// element on that dimension. The dimension order is determined by the value of
    /// `data_format`, see above for details. Dilations in the batch and depth
    /// dimensions must be 1.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor depthwise_conv2d_native(Tensor input, Tensor filter, int[] strides, string padding, int[] explicit_paddings = null, string data_format = "NHWC", int[] dilations = null, string? name = null)
    {
        var _ctx = tf.Context;
        if (explicit_paddings is null)
        {
            explicit_paddings = new int[] { };
        }
        if (dilations is null)
        {
            dilations = new int[] { 1, 1, 1, 1 };
        }
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "DepthwiseConv2dNative", name) { args = new object[] { input, filter }, attrs = new Dictionary<string, object>() { ["strides"] = strides, ["padding"] = padding, ["explicit_paddings"] = explicit_paddings, ["data_format"] = data_format, ["dilations"] = dilations } });
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
                return depthwise_conv2d_native_eager_fallback(input, filter, strides: strides, padding: padding, explicit_paddings: explicit_paddings, data_format: data_format, dilations: dilations, name: name, ctx: _ctx);
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
        keywords["filter"] = filter;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["explicit_paddings"] = explicit_paddings;
        keywords["data_format"] = data_format;
        keywords["dilations"] = dilations;
        var _op = tf.OpDefLib._apply_op_helper("DepthwiseConv2dNative", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "explicit_paddings", _op.get_attr("explicit_paddings"), "data_format", _op.get_attr("data_format"), "dilations", _op.get_attr("dilations") };
            _execute.record_gradient("DepthwiseConv2dNative", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor depthwise_conv2d_native_eager_fallback(Tensor input, Tensor filter, int[] strides, string padding, int[] explicit_paddings, string data_format, int[] dilations, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, filter };
        object[] _attrs = new object[] { "T", input.dtype, "strides", strides, "padding", padding, "explicit_paddings", explicit_paddings, "data_format", data_format, "dilations", dilations };
        var _result = _execute.execute("DepthwiseConv2dNative", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("DepthwiseConv2dNative", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the gradients of depthwise convolution with respect to the filter.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="filter_sizes"></param>
    /// <param name="out_backprop"></param>
    /// <param name="strides">
    /// 
    /// The stride of the sliding window for each dimension of the input
    /// of the convolution.
    /// 
    /// </param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <param name="explicit_paddings"></param>
    /// <param name="data_format">
    /// 
    /// Specify the data format of the input and output data. With the
    /// default format "NHWC", the data is stored in the order of:
    ///     [batch, height, width, channels].
    /// Alternatively, the format could be "NCHW", the data storage order of:
    ///     [batch, channels, height, width].
    /// 
    /// </param>
    /// <param name="dilations">
    /// 
    /// 1-D tensor of length 4.  The dilation factor for each dimension of
    /// `input`. If set to k > 1, there will be k-1 skipped cells between each filter
    /// element on that dimension. The dimension order is determined by the value of
    /// `data_format`, see above for details. Dilations in the batch and depth
    /// dimensions must be 1.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor depthwise_conv2d_native_backprop_filter(Tensor input, Tensor filter_sizes, Tensor out_backprop, int[] strides, string padding, int[] explicit_paddings = null, string data_format = "NHWC", int[] dilations = null, string? name = null)
    {
        var _ctx = tf.Context;
        if (explicit_paddings is null)
        {
            explicit_paddings = new int[] { };
        }
        if (dilations is null)
        {
            dilations = new int[] { 1, 1, 1, 1 };
        }
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "DepthwiseConv2dNativeBackpropFilter", name) { args = new object[] { input, filter_sizes, out_backprop }, attrs = new Dictionary<string, object>() { ["strides"] = strides, ["padding"] = padding, ["explicit_paddings"] = explicit_paddings, ["data_format"] = data_format, ["dilations"] = dilations } });
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
                return depthwise_conv2d_native_backprop_filter_eager_fallback(input, filter_sizes, out_backprop, strides: strides, padding: padding, explicit_paddings: explicit_paddings, data_format: data_format, dilations: dilations, name: name, ctx: _ctx);
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
        keywords["filter_sizes"] = filter_sizes;
        keywords["out_backprop"] = out_backprop;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["explicit_paddings"] = explicit_paddings;
        keywords["data_format"] = data_format;
        keywords["dilations"] = dilations;
        var _op = tf.OpDefLib._apply_op_helper("DepthwiseConv2dNativeBackpropFilter", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "explicit_paddings", _op.get_attr("explicit_paddings"), "data_format", _op.get_attr("data_format"), "dilations", _op.get_attr("dilations") };
            _execute.record_gradient("DepthwiseConv2dNativeBackpropFilter", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor depthwise_conv2d_native_backprop_filter_eager_fallback(Tensor input, Tensor filter_sizes, Tensor out_backprop, int[] strides, string padding, int[] explicit_paddings, string data_format, int[] dilations, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, filter_sizes, out_backprop };
        object[] _attrs = new object[] { "T", input.dtype, "strides", strides, "padding", padding, "explicit_paddings", explicit_paddings, "data_format", data_format, "dilations", dilations };
        var _result = _execute.execute("DepthwiseConv2dNativeBackpropFilter", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("DepthwiseConv2dNativeBackpropFilter", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the gradients of depthwise convolution with respect to the input.
    /// </summary>
    /// <param name="input_sizes"></param>
    /// <param name="filter"></param>
    /// <param name="out_backprop"></param>
    /// <param name="strides">
    /// 
    /// The stride of the sliding window for each dimension of the input
    /// of the convolution.
    /// 
    /// </param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <param name="explicit_paddings"></param>
    /// <param name="data_format">
    /// 
    /// Specify the data format of the input and output data. With the
    /// default format "NHWC", the data is stored in the order of:
    ///     [batch, height, width, channels].
    /// Alternatively, the format could be "NCHW", the data storage order of:
    ///     [batch, channels, height, width].
    /// 
    /// </param>
    /// <param name="dilations">
    /// 
    /// 1-D tensor of length 4.  The dilation factor for each dimension of
    /// `input`. If set to k > 1, there will be k-1 skipped cells between each filter
    /// element on that dimension. The dimension order is determined by the value of
    /// `data_format`, see above for details. Dilations in the batch and depth
    /// dimensions must be 1.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor depthwise_conv2d_native_backprop_input(Tensor input_sizes, Tensor filter, Tensor out_backprop, int[] strides, string padding, int[] explicit_paddings = null, string data_format = "NHWC", int[] dilations = null, string? name = null)
    {
        var _ctx = tf.Context;
        if (explicit_paddings is null)
        {
            explicit_paddings = new int[] { };
        }
        if (dilations is null)
        {
            dilations = new int[] { 1, 1, 1, 1 };
        }
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "DepthwiseConv2dNativeBackpropInput", name) { args = new object[] { input_sizes, filter, out_backprop }, attrs = new Dictionary<string, object>() { ["strides"] = strides, ["padding"] = padding, ["explicit_paddings"] = explicit_paddings, ["data_format"] = data_format, ["dilations"] = dilations } });
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
                return depthwise_conv2d_native_backprop_input_eager_fallback(input_sizes, filter, out_backprop, strides: strides, padding: padding, explicit_paddings: explicit_paddings, data_format: data_format, dilations: dilations, name: name, ctx: _ctx);
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
        keywords["input_sizes"] = input_sizes;
        keywords["filter"] = filter;
        keywords["out_backprop"] = out_backprop;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["explicit_paddings"] = explicit_paddings;
        keywords["data_format"] = data_format;
        keywords["dilations"] = dilations;
        var _op = tf.OpDefLib._apply_op_helper("DepthwiseConv2dNativeBackpropInput", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "explicit_paddings", _op.get_attr("explicit_paddings"), "data_format", _op.get_attr("data_format"), "dilations", _op.get_attr("dilations") };
            _execute.record_gradient("DepthwiseConv2dNativeBackpropInput", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor depthwise_conv2d_native_backprop_input_eager_fallback(Tensor input_sizes, Tensor filter, Tensor out_backprop, int[] strides, string padding, int[] explicit_paddings, string data_format, int[] dilations, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input_sizes, filter, out_backprop };
        object[] _attrs = new object[] { "T", filter.dtype, "strides", strides, "padding", padding, "explicit_paddings", explicit_paddings, "data_format", data_format, "dilations", dilations };
        var _result = _execute.execute("DepthwiseConv2dNativeBackpropInput", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("DepthwiseConv2dNativeBackpropInput", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the grayscale dilation of 4-D `input` and 3-D `filter` tensors.
    /// </summary>
    /// <remarks>
    /// 
    /// The `input` tensor has shape `[batch, in_height, in_width, depth]` and the
    /// `filter` tensor has shape `[filter_height, filter_width, depth]`, i.e., each
    /// input channel is processed independently of the others with its own structuring
    /// function. The `output` tensor has shape
    /// `[batch, out_height, out_width, depth]`. The spatial dimensions of the output
    /// tensor depend on the `padding` algorithm. We currently only support the default
    /// "NHWC" `data_format`.
    /// 
    /// In detail, the grayscale morphological 2-D dilation is the max-sum correlation
    /// (for consistency with `conv2d`, we use unmirrored filters):
    /// 
    ///     output[b, y, x, c] =
    ///        max_{dy, dx} input[b,
    ///                           strides[1] * y + rates[1] * dy,
    ///                           strides[2] * x + rates[2] * dx,
    ///                           c] +
    ///                     filter[dy, dx, c]
    /// 
    /// Max-pooling is a special case when the filter has size equal to the pooling
    /// kernel size and contains all zeros.
    /// 
    /// Note on duality: The dilation of `input` by the `filter` is equal to the
    /// negation of the erosion of `-input` by the reflected `filter`.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="filter"></param>
    /// <param name="strides">
    /// 
    /// The stride of the sliding window for each dimension of the input
    /// tensor. Must be: `[1, stride_height, stride_width, 1]`.
    /// 
    /// </param>
    /// <param name="rates">
    /// 
    /// The input stride for atrous morphological dilation. Must be:
    /// `[1, rate_height, rate_width, 1]`.
    /// 
    /// </param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor dilation2d(Tensor input, Tensor filter, int[] strides, int[] rates, string padding, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Dilation2D", name) { args = new object[] { input, filter }, attrs = new Dictionary<string, object>() { ["strides"] = strides, ["rates"] = rates, ["padding"] = padding } });
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
                return dilation2d_eager_fallback(input, filter, strides: strides, rates: rates, padding: padding, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["filter"] = filter;
        keywords["strides"] = strides;
        keywords["rates"] = rates;
        keywords["padding"] = padding;
        var _op = tf.OpDefLib._apply_op_helper("Dilation2D", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "strides", _op.get_attr("strides"), "rates", _op.get_attr("rates"), "padding", _op.get_attr("padding") };
            _execute.record_gradient("Dilation2D", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor dilation2d_eager_fallback(Tensor input, Tensor filter, int[] strides, int[] rates, string padding, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, filter };
        object[] _attrs = new object[] { "T", input.dtype, "strides", strides, "rates", rates, "padding", padding };
        var _result = _execute.execute("Dilation2D", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Dilation2D", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the gradient of morphological 2-D dilation with respect to the filter.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="filter"></param>
    /// <param name="out_backprop"></param>
    /// <param name="strides">
    /// 
    /// 1-D of length 4. The stride of the sliding window for each dimension of
    /// the input tensor. Must be: `[1, stride_height, stride_width, 1]`.
    /// 
    /// </param>
    /// <param name="rates">
    /// 
    /// 1-D of length 4. The input stride for atrous morphological dilation.
    /// Must be: `[1, rate_height, rate_width, 1]`.
    /// 
    /// </param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor dilation2d_backprop_filter(Tensor input, Tensor filter, Tensor out_backprop, int[] strides, int[] rates, string padding, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Dilation2DBackpropFilter", name) { args = new object[] { input, filter, out_backprop }, attrs = new Dictionary<string, object>() { ["strides"] = strides, ["rates"] = rates, ["padding"] = padding } });
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
                return dilation2d_backprop_filter_eager_fallback(input, filter, out_backprop, strides: strides, rates: rates, padding: padding, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["filter"] = filter;
        keywords["out_backprop"] = out_backprop;
        keywords["strides"] = strides;
        keywords["rates"] = rates;
        keywords["padding"] = padding;
        var _op = tf.OpDefLib._apply_op_helper("Dilation2DBackpropFilter", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "strides", _op.get_attr("strides"), "rates", _op.get_attr("rates"), "padding", _op.get_attr("padding") };
            _execute.record_gradient("Dilation2DBackpropFilter", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor dilation2d_backprop_filter_eager_fallback(Tensor input, Tensor filter, Tensor out_backprop, int[] strides, int[] rates, string padding, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, filter, out_backprop };
        object[] _attrs = new object[] { "T", input.dtype, "strides", strides, "rates", rates, "padding", padding };
        var _result = _execute.execute("Dilation2DBackpropFilter", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Dilation2DBackpropFilter", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the gradient of morphological 2-D dilation with respect to the input.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="filter"></param>
    /// <param name="out_backprop"></param>
    /// <param name="strides">
    /// 
    /// 1-D of length 4. The stride of the sliding window for each dimension of
    /// the input tensor. Must be: `[1, stride_height, stride_width, 1]`.
    /// 
    /// </param>
    /// <param name="rates">
    /// 
    /// 1-D of length 4. The input stride for atrous morphological dilation.
    /// Must be: `[1, rate_height, rate_width, 1]`.
    /// 
    /// </param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor dilation2d_backprop_input(Tensor input, Tensor filter, Tensor out_backprop, int[] strides, int[] rates, string padding, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Dilation2DBackpropInput", name) { args = new object[] { input, filter, out_backprop }, attrs = new Dictionary<string, object>() { ["strides"] = strides, ["rates"] = rates, ["padding"] = padding } });
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
                return dilation2d_backprop_input_eager_fallback(input, filter, out_backprop, strides: strides, rates: rates, padding: padding, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["filter"] = filter;
        keywords["out_backprop"] = out_backprop;
        keywords["strides"] = strides;
        keywords["rates"] = rates;
        keywords["padding"] = padding;
        var _op = tf.OpDefLib._apply_op_helper("Dilation2DBackpropInput", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "strides", _op.get_attr("strides"), "rates", _op.get_attr("rates"), "padding", _op.get_attr("padding") };
            _execute.record_gradient("Dilation2DBackpropInput", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor dilation2d_backprop_input_eager_fallback(Tensor input, Tensor filter, Tensor out_backprop, int[] strides, int[] rates, string padding, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, filter, out_backprop };
        object[] _attrs = new object[] { "T", input.dtype, "strides", strides, "rates", rates, "padding", padding };
        var _result = _execute.execute("Dilation2DBackpropInput", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Dilation2DBackpropInput", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes the exponential linear function.
    /// </summary>
    /// <remarks>
    /// 
    /// The ELU function is defined as:
    /// 
    ///  * $ e ^ x - 1 $ if $ x < 0 $
    ///  * $ x $ if $ x >= 0 $
    /// 
    /// Examples:
    /// 
    /// >>> tf.nn.elu(1.0)
    /// <tf.Tensor: shape=(), dtype=float32, numpy=1.0>
    /// >>> tf.nn.elu(0.0)
    /// <tf.Tensor: shape=(), dtype=float32, numpy=0.0>
    /// >>> tf.nn.elu(-1000.0)
    /// <tf.Tensor: shape=(), dtype=float32, numpy=-1.0>
    /// 
    /// See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
    /// ](http://arxiv.org/abs/1511.07289)
    /// 
    /// </remarks>
    /// <param name="features"></param>
    /// <returns></returns>
    public static Tensor elu(Tensor features, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Elu", name) { args = new object[] { features }, attrs = new Dictionary<string, object>() { } });
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
                return elu_eager_fallback(features, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["features"] = features;
        var _op = tf.OpDefLib._apply_op_helper("Elu", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Elu", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor elu_eager_fallback(Tensor features, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { features };
        object[] _attrs = new object[] { "T", features.dtype };
        var _result = _execute.execute("Elu", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Elu", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes gradients for the exponential linear (Elu) operation.
    /// </summary>
    /// <param name="gradients"></param>
    /// <param name="outputs"></param>
    /// <returns></returns>
    public static Tensor elu_grad(Tensor gradients, Tensor outputs, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "EluGrad", name) { args = new object[] { gradients, outputs }, attrs = new Dictionary<string, object>() { } });
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
                return elu_grad_eager_fallback(gradients, outputs, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["gradients"] = gradients;
        keywords["outputs"] = outputs;
        var _op = tf.OpDefLib._apply_op_helper("EluGrad", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("EluGrad", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor elu_grad_eager_fallback(Tensor gradients, Tensor outputs, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { gradients, outputs };
        object[] _attrs = new object[] { "T", gradients.dtype };
        var _result = _execute.execute("EluGrad", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("EluGrad", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Performs fractional average pooling on the input.
    /// </summary>
    /// <remarks>
    /// 
    /// Fractional average pooling is similar to Fractional max pooling in the pooling
    /// region generation step. The only difference is that after pooling regions are
    /// generated, a mean operation is performed instead of a max operation in each
    /// pooling region.
    /// 
    /// </remarks>
    /// <param name="value"></param>
    /// <param name="pooling_ratio">
    /// 
    /// Pooling ratio for each dimension of `value`, currently only
    /// supports row and col dimension and should be >= 1.0. For example, a valid
    /// pooling ratio looks like [1.0, 1.44, 1.73, 1.0]. The first and last elements
    /// must be 1.0 because we don't allow pooling on batch and channels
    /// dimensions. 1.44 and 1.73 are pooling ratio on height and width dimensions
    /// respectively.
    /// 
    /// </param>
    /// <param name="pseudo_random">
    /// 
    /// When set to True, generates the pooling sequence in a
    /// pseudorandom fashion, otherwise, in a random fashion. Check paper [Benjamin
    /// Graham, Fractional Max-Pooling](http://arxiv.org/abs/1412.6071) for
    /// difference between pseudorandom and random.
    /// 
    /// </param>
    /// <param name="overlapping">
    /// 
    /// When set to True, it means when pooling, the values at the boundary
    /// of adjacent pooling cells are used by both cells. For example:
    /// 
    /// `index  0  1  2  3  4`
    /// 
    /// `value  20 5  16 3  7`
    /// 
    /// If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
    /// The result would be [41/3, 26/3] for fractional avg pooling.
    /// 
    /// </param>
    /// <param name="deterministic">
    /// 
    /// When set to True, a fixed pooling region will be used when
    /// iterating over a FractionalAvgPool node in the computation graph. Mainly used
    /// in unit test to make FractionalAvgPool deterministic.
    /// 
    /// </param>
    /// <param name="seed">
    /// 
    /// If either seed or seed2 are set to be non-zero, the random number
    /// generator is seeded by the given seed.  Otherwise, it is seeded by a
    /// random seed.
    /// 
    /// </param>
    /// <param name="seed2">
    /// 
    /// An second seed to avoid seed collision.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] fractional_avg_pool(Tensor value, float[] pooling_ratio, bool pseudo_random = false, bool overlapping = false, bool deterministic = false, int seed = 0, int seed2 = 0, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "FractionalAvgPool", name) { args = new object[] { value }, attrs = new Dictionary<string, object>() { ["pooling_ratio"] = pooling_ratio, ["pseudo_random"] = pseudo_random, ["overlapping"] = overlapping, ["deterministic"] = deterministic, ["seed"] = seed, ["seed2"] = seed2 } });
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
                return fractional_avg_pool_eager_fallback(value, pooling_ratio: pooling_ratio, pseudo_random: pseudo_random, overlapping: overlapping, deterministic: deterministic, seed: seed, seed2: seed2, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["value"] = value;
        keywords["pooling_ratio"] = pooling_ratio;
        keywords["pseudo_random"] = pseudo_random;
        keywords["overlapping"] = overlapping;
        keywords["deterministic"] = deterministic;
        keywords["seed"] = seed;
        keywords["seed2"] = seed2;
        var _op = tf.OpDefLib._apply_op_helper("FractionalAvgPool", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "pooling_ratio", _op.get_attr("pooling_ratio"), "pseudo_random", _op._get_attr_bool("pseudo_random"), "overlapping", _op._get_attr_bool("overlapping"), "deterministic", _op._get_attr_bool("deterministic"), "seed", _op._get_attr_int("seed"), "seed2", _op._get_attr_int("seed2"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("FractionalAvgPool", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] fractional_avg_pool_eager_fallback(Tensor value, float[] pooling_ratio, bool pseudo_random, bool overlapping, bool deterministic, int seed, int seed2, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { value };
        object[] _attrs = new object[] { "pooling_ratio", pooling_ratio, "pseudo_random", pseudo_random, "overlapping", overlapping, "deterministic", deterministic, "seed", seed, "seed2", seed2, "T", value.dtype };
        var _result = _execute.execute("FractionalAvgPool", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("FractionalAvgPool", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Computes gradient of the FractionalAvgPool function.
    /// </summary>
    /// <remarks>
    /// 
    /// Unlike FractionalMaxPoolGrad, we don't need to find arg_max for
    /// FractionalAvgPoolGrad, we just need to evenly back-propagate each element of
    /// out_backprop to those indices that form the same pooling cell. Therefore, we
    /// just need to know the shape of original input tensor, instead of the whole
    /// tensor.
    /// 
    /// </remarks>
    /// <param name="orig_input_tensor_shape"></param>
    /// <param name="out_backprop"></param>
    /// <param name="row_pooling_sequence"></param>
    /// <param name="col_pooling_sequence"></param>
    /// <param name="overlapping">
    /// 
    /// When set to True, it means when pooling, the values at the boundary
    /// of adjacent pooling cells are used by both cells. For example:
    /// 
    /// `index  0  1  2  3  4`
    /// 
    /// `value  20 5  16 3  7`
    /// 
    /// If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
    /// The result would be [41/3, 26/3] for fractional avg pooling.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor fractional_avg_pool_grad(Tensor orig_input_tensor_shape, Tensor out_backprop, Tensor row_pooling_sequence, Tensor col_pooling_sequence, bool overlapping = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "FractionalAvgPoolGrad", name) { args = new object[] { orig_input_tensor_shape, out_backprop, row_pooling_sequence, col_pooling_sequence }, attrs = new Dictionary<string, object>() { ["overlapping"] = overlapping } });
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
                return fractional_avg_pool_grad_eager_fallback(orig_input_tensor_shape, out_backprop, row_pooling_sequence, col_pooling_sequence, overlapping: overlapping, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["orig_input_tensor_shape"] = orig_input_tensor_shape;
        keywords["out_backprop"] = out_backprop;
        keywords["row_pooling_sequence"] = row_pooling_sequence;
        keywords["col_pooling_sequence"] = col_pooling_sequence;
        keywords["overlapping"] = overlapping;
        var _op = tf.OpDefLib._apply_op_helper("FractionalAvgPoolGrad", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "overlapping", _op._get_attr_bool("overlapping"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("FractionalAvgPoolGrad", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor fractional_avg_pool_grad_eager_fallback(Tensor orig_input_tensor_shape, Tensor out_backprop, Tensor row_pooling_sequence, Tensor col_pooling_sequence, bool overlapping, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { orig_input_tensor_shape, out_backprop, row_pooling_sequence, col_pooling_sequence };
        object[] _attrs = new object[] { "overlapping", overlapping, "T", out_backprop.dtype };
        var _result = _execute.execute("FractionalAvgPoolGrad", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("FractionalAvgPoolGrad", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Performs fractional max pooling on the input.
    /// </summary>
    /// <remarks>
    /// 
    /// Fractional max pooling is slightly different than regular max pooling.  In
    /// regular max pooling, you downsize an input set by taking the maximum value of
    /// smaller N x N subsections of the set (often 2x2), and try to reduce the set by
    /// a factor of N, where N is an integer.  Fractional max pooling, as you might
    /// expect from the word "fractional", means that the overall reduction ratio N
    /// does not have to be an integer.
    /// 
    /// The sizes of the pooling regions are generated randomly but are fairly uniform.
    /// For example, let's look at the height dimension, and the constraints on the
    /// list of rows that will be pool boundaries.
    /// 
    /// First we define the following:
    /// 
    /// 1.  input_row_length : the number of rows from the input set
    /// 2.  output_row_length : which will be smaller than the input
    /// 3.  alpha = input_row_length / output_row_length : our reduction ratio
    /// 4.  K = floor(alpha)
    /// 5.  row_pooling_sequence : this is the result list of pool boundary rows
    /// 
    /// Then, row_pooling_sequence should satisfy:
    /// 
    /// 1.  a[0] = 0 : the first value of the sequence is 0
    /// 2.  a[end] = input_row_length : the last value of the sequence is the size
    /// 3.  K <= (a[i+1] - a[i]) <= K+1 : all intervals are K or K+1 size
    /// 4.  length(row_pooling_sequence) = output_row_length+1
    /// 
    /// For more details on fractional max pooling, see this paper:
    /// [Benjamin Graham, Fractional Max-Pooling](http://arxiv.org/abs/1412.6071)
    /// 
    /// </remarks>
    /// <param name="value"></param>
    /// <param name="pooling_ratio">
    /// 
    /// Pooling ratio for each dimension of `value`, currently only
    /// supports row and col dimension and should be >= 1.0. For example, a valid
    /// pooling ratio looks like [1.0, 1.44, 1.73, 1.0]. The first and last elements
    /// must be 1.0 because we don't allow pooling on batch and channels
    /// dimensions. 1.44 and 1.73 are pooling ratio on height and width dimensions
    /// respectively.
    /// 
    /// </param>
    /// <param name="pseudo_random">
    /// 
    /// When set to True, generates the pooling sequence in a
    /// pseudorandom fashion, otherwise, in a random fashion. Check paper [Benjamin
    /// Graham, Fractional Max-Pooling](http://arxiv.org/abs/1412.6071) for
    /// difference between pseudorandom and random.
    /// 
    /// </param>
    /// <param name="overlapping">
    /// 
    /// When set to True, it means when pooling, the values at the boundary
    /// of adjacent pooling cells are used by both cells. For example:
    /// 
    /// `index  0  1  2  3  4`
    /// 
    /// `value  20 5  16 3  7`
    /// 
    /// If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
    /// The result would be [20, 16] for fractional max pooling.
    /// 
    /// </param>
    /// <param name="deterministic">
    /// 
    /// When set to True, a fixed pooling region will be used when
    /// iterating over a FractionalMaxPool node in the computation graph. Mainly used
    /// in unit test to make FractionalMaxPool deterministic.
    /// 
    /// </param>
    /// <param name="seed">
    /// 
    /// If either seed or seed2 are set to be non-zero, the random number
    /// generator is seeded by the given seed.  Otherwise, it is seeded by a
    /// random seed.
    /// 
    /// </param>
    /// <param name="seed2">
    /// 
    /// An second seed to avoid seed collision.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] fractional_max_pool(Tensor value, float[] pooling_ratio, bool pseudo_random = false, bool overlapping = false, bool deterministic = false, int seed = 0, int seed2 = 0, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "FractionalMaxPool", name) { args = new object[] { value }, attrs = new Dictionary<string, object>() { ["pooling_ratio"] = pooling_ratio, ["pseudo_random"] = pseudo_random, ["overlapping"] = overlapping, ["deterministic"] = deterministic, ["seed"] = seed, ["seed2"] = seed2 } });
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
                return fractional_max_pool_eager_fallback(value, pooling_ratio: pooling_ratio, pseudo_random: pseudo_random, overlapping: overlapping, deterministic: deterministic, seed: seed, seed2: seed2, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["value"] = value;
        keywords["pooling_ratio"] = pooling_ratio;
        keywords["pseudo_random"] = pseudo_random;
        keywords["overlapping"] = overlapping;
        keywords["deterministic"] = deterministic;
        keywords["seed"] = seed;
        keywords["seed2"] = seed2;
        var _op = tf.OpDefLib._apply_op_helper("FractionalMaxPool", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "pooling_ratio", _op.get_attr("pooling_ratio"), "pseudo_random", _op._get_attr_bool("pseudo_random"), "overlapping", _op._get_attr_bool("overlapping"), "deterministic", _op._get_attr_bool("deterministic"), "seed", _op._get_attr_int("seed"), "seed2", _op._get_attr_int("seed2"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("FractionalMaxPool", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] fractional_max_pool_eager_fallback(Tensor value, float[] pooling_ratio, bool pseudo_random, bool overlapping, bool deterministic, int seed, int seed2, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { value };
        object[] _attrs = new object[] { "pooling_ratio", pooling_ratio, "pseudo_random", pseudo_random, "overlapping", overlapping, "deterministic", deterministic, "seed", seed, "seed2", seed2, "T", value.dtype };
        var _result = _execute.execute("FractionalMaxPool", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("FractionalMaxPool", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Computes gradient of the FractionalMaxPool function.
    /// </summary>
    /// <param name="orig_input"></param>
    /// <param name="orig_output"></param>
    /// <param name="out_backprop"></param>
    /// <param name="row_pooling_sequence"></param>
    /// <param name="col_pooling_sequence"></param>
    /// <param name="overlapping">
    /// 
    /// When set to True, it means when pooling, the values at the boundary
    /// of adjacent pooling cells are used by both cells. For example:
    /// 
    /// `index  0  1  2  3  4`
    /// 
    /// `value  20 5  16 3  7`
    /// 
    /// If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
    /// The result would be [20, 16] for fractional max pooling.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor fractional_max_pool_grad(Tensor orig_input, Tensor orig_output, Tensor out_backprop, Tensor row_pooling_sequence, Tensor col_pooling_sequence, bool overlapping = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "FractionalMaxPoolGrad", name) { args = new object[] { orig_input, orig_output, out_backprop, row_pooling_sequence, col_pooling_sequence }, attrs = new Dictionary<string, object>() { ["overlapping"] = overlapping } });
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
                return fractional_max_pool_grad_eager_fallback(orig_input, orig_output, out_backprop, row_pooling_sequence, col_pooling_sequence, overlapping: overlapping, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["orig_input"] = orig_input;
        keywords["orig_output"] = orig_output;
        keywords["out_backprop"] = out_backprop;
        keywords["row_pooling_sequence"] = row_pooling_sequence;
        keywords["col_pooling_sequence"] = col_pooling_sequence;
        keywords["overlapping"] = overlapping;
        var _op = tf.OpDefLib._apply_op_helper("FractionalMaxPoolGrad", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "overlapping", _op._get_attr_bool("overlapping"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("FractionalMaxPoolGrad", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor fractional_max_pool_grad_eager_fallback(Tensor orig_input, Tensor orig_output, Tensor out_backprop, Tensor row_pooling_sequence, Tensor col_pooling_sequence, bool overlapping, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { orig_input, orig_output, out_backprop, row_pooling_sequence, col_pooling_sequence };
        object[] _attrs = new object[] { "overlapping", overlapping, "T", orig_input.dtype };
        var _result = _execute.execute("FractionalMaxPoolGrad", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("FractionalMaxPoolGrad", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Batch normalization.
    /// </summary>
    /// <remarks>
    /// 
    /// Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
    /// The size of 1D Tensors matches the dimension C of the 4D Tensors.
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="scale"></param>
    /// <param name="offset"></param>
    /// <param name="mean"></param>
    /// <param name="variance"></param>
    /// <param name="epsilon">
    /// 
    /// A small float number added to the variance of x.
    /// 
    /// </param>
    /// <param name="exponential_avg_factor"></param>
    /// <param name="data_format">
    /// 
    /// The data format for x and y. Either "NHWC" (default) or "NCHW".
    /// 
    /// </param>
    /// <param name="is_training">
    /// 
    /// A bool value to indicate the operation is for training (default)
    /// or inference.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] fused_batch_norm(Tensor x, Tensor scale, Tensor offset, Tensor mean, Tensor variance, float epsilon = 0.0001f, float exponential_avg_factor = 1f, string data_format = "NHWC", bool is_training = true, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "FusedBatchNorm", name) { args = new object[] { x, scale, offset, mean, variance }, attrs = new Dictionary<string, object>() { ["epsilon"] = epsilon, ["exponential_avg_factor"] = exponential_avg_factor, ["data_format"] = data_format, ["is_training"] = is_training } });
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
                return fused_batch_norm_eager_fallback(x, scale, offset, mean, variance, epsilon: epsilon, exponential_avg_factor: exponential_avg_factor, data_format: data_format, is_training: is_training, name: name, ctx: _ctx);
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
        keywords["x"] = x;
        keywords["scale"] = scale;
        keywords["offset"] = offset;
        keywords["mean"] = mean;
        keywords["variance"] = variance;
        keywords["epsilon"] = epsilon;
        keywords["exponential_avg_factor"] = exponential_avg_factor;
        keywords["data_format"] = data_format;
        keywords["is_training"] = is_training;
        var _op = tf.OpDefLib._apply_op_helper("FusedBatchNorm", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "epsilon", _op.get_attr("epsilon"), "exponential_avg_factor", _op.get_attr("exponential_avg_factor"), "data_format", _op.get_attr("data_format"), "is_training", _op._get_attr_bool("is_training") };
            _execute.record_gradient("FusedBatchNorm", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] fused_batch_norm_eager_fallback(Tensor x, Tensor scale, Tensor offset, Tensor mean, Tensor variance, float epsilon, float exponential_avg_factor, string data_format, bool is_training, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, scale, offset, mean, variance };
        object[] _attrs = new object[] { "T", x.dtype, "epsilon", epsilon, "exponential_avg_factor", exponential_avg_factor, "data_format", data_format, "is_training", is_training };
        var _result = _execute.execute("FusedBatchNorm", 5, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("FusedBatchNorm", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Gradient for batch normalization.
    /// </summary>
    /// <remarks>
    /// 
    /// Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
    /// The size of 1D Tensors matches the dimension C of the 4D Tensors.
    /// 
    /// </remarks>
    /// <param name="y_backprop"></param>
    /// <param name="x"></param>
    /// <param name="scale"></param>
    /// <param name="reserve_space_1"></param>
    /// <param name="reserve_space_2"></param>
    /// <param name="epsilon">
    /// 
    /// A small float number added to the variance of x.
    /// 
    /// </param>
    /// <param name="data_format">
    /// 
    /// The data format for y_backprop, x, x_backprop.
    /// Either "NHWC" (default) or "NCHW".
    /// 
    /// </param>
    /// <param name="is_training">
    /// 
    /// A bool value to indicate the operation is for training (default)
    /// or inference.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] fused_batch_norm_grad(Tensor y_backprop, Tensor x, Tensor scale, Tensor reserve_space_1, Tensor reserve_space_2, float epsilon = 0.0001f, string data_format = "NHWC", bool is_training = true, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "FusedBatchNormGrad", name) { args = new object[] { y_backprop, x, scale, reserve_space_1, reserve_space_2 }, attrs = new Dictionary<string, object>() { ["epsilon"] = epsilon, ["data_format"] = data_format, ["is_training"] = is_training } });
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
                return fused_batch_norm_grad_eager_fallback(y_backprop, x, scale, reserve_space_1, reserve_space_2, epsilon: epsilon, data_format: data_format, is_training: is_training, name: name, ctx: _ctx);
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
        keywords["y_backprop"] = y_backprop;
        keywords["x"] = x;
        keywords["scale"] = scale;
        keywords["reserve_space_1"] = reserve_space_1;
        keywords["reserve_space_2"] = reserve_space_2;
        keywords["epsilon"] = epsilon;
        keywords["data_format"] = data_format;
        keywords["is_training"] = is_training;
        var _op = tf.OpDefLib._apply_op_helper("FusedBatchNormGrad", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "epsilon", _op.get_attr("epsilon"), "data_format", _op.get_attr("data_format"), "is_training", _op._get_attr_bool("is_training") };
            _execute.record_gradient("FusedBatchNormGrad", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] fused_batch_norm_grad_eager_fallback(Tensor y_backprop, Tensor x, Tensor scale, Tensor reserve_space_1, Tensor reserve_space_2, float epsilon, string data_format, bool is_training, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { y_backprop, x, scale, reserve_space_1, reserve_space_2 };
        object[] _attrs = new object[] { "T", y_backprop.dtype, "epsilon", epsilon, "data_format", data_format, "is_training", is_training };
        var _result = _execute.execute("FusedBatchNormGrad", 5, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("FusedBatchNormGrad", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Gradient for batch normalization.
    /// </summary>
    /// <remarks>
    /// 
    /// Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
    /// The size of 1D Tensors matches the dimension C of the 4D Tensors.
    /// 
    /// </remarks>
    /// <param name="y_backprop"></param>
    /// <param name="x"></param>
    /// <param name="scale"></param>
    /// <param name="reserve_space_1"></param>
    /// <param name="reserve_space_2"></param>
    /// <param name="epsilon">
    /// 
    /// A small float number added to the variance of x.
    /// 
    /// </param>
    /// <param name="data_format">
    /// 
    /// The data format for y_backprop, x, x_backprop.
    /// Either "NHWC" (default) or "NCHW".
    /// 
    /// </param>
    /// <param name="is_training">
    /// 
    /// A bool value to indicate the operation is for training (default)
    /// or inference.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] fused_batch_norm_grad_v2(Tensor y_backprop, Tensor x, Tensor scale, Tensor reserve_space_1, Tensor reserve_space_2, float epsilon = 0.0001f, string data_format = "NHWC", bool is_training = true, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "FusedBatchNormGradV2", name) { args = new object[] { y_backprop, x, scale, reserve_space_1, reserve_space_2 }, attrs = new Dictionary<string, object>() { ["epsilon"] = epsilon, ["data_format"] = data_format, ["is_training"] = is_training } });
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
                return fused_batch_norm_grad_v2_eager_fallback(y_backprop, x, scale, reserve_space_1, reserve_space_2, epsilon: epsilon, data_format: data_format, is_training: is_training, name: name, ctx: _ctx);
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
        keywords["y_backprop"] = y_backprop;
        keywords["x"] = x;
        keywords["scale"] = scale;
        keywords["reserve_space_1"] = reserve_space_1;
        keywords["reserve_space_2"] = reserve_space_2;
        keywords["epsilon"] = epsilon;
        keywords["data_format"] = data_format;
        keywords["is_training"] = is_training;
        var _op = tf.OpDefLib._apply_op_helper("FusedBatchNormGradV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "U", _op._get_attr_type("U"), "epsilon", _op.get_attr("epsilon"), "data_format", _op.get_attr("data_format"), "is_training", _op._get_attr_bool("is_training") };
            _execute.record_gradient("FusedBatchNormGradV2", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] fused_batch_norm_grad_v2_eager_fallback(Tensor y_backprop, Tensor x, Tensor scale, Tensor reserve_space_1, Tensor reserve_space_2, float epsilon, string data_format, bool is_training, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { y_backprop, x, scale, reserve_space_1, reserve_space_2 };
        object[] _attrs = new object[] { "T", y_backprop.dtype, "U", reserve_space_1.dtype, "epsilon", epsilon, "data_format", data_format, "is_training", is_training };
        var _result = _execute.execute("FusedBatchNormGradV2", 5, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("FusedBatchNormGradV2", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Gradient for batch normalization.
    /// </summary>
    /// <remarks>
    /// 
    /// Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
    /// The size of 1D Tensors matches the dimension C of the 4D Tensors.
    /// 
    /// </remarks>
    /// <param name="y_backprop"></param>
    /// <param name="x"></param>
    /// <param name="scale"></param>
    /// <param name="reserve_space_1"></param>
    /// <param name="reserve_space_2"></param>
    /// <param name="reserve_space_3"></param>
    /// <param name="epsilon">
    /// 
    /// A small float number added to the variance of x.
    /// 
    /// </param>
    /// <param name="data_format">
    /// 
    /// The data format for y_backprop, x, x_backprop.
    /// Either "NHWC" (default) or "NCHW".
    /// 
    /// </param>
    /// <param name="is_training">
    /// 
    /// A bool value to indicate the operation is for training (default)
    /// or inference.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] fused_batch_norm_grad_v3(Tensor y_backprop, Tensor x, Tensor scale, Tensor reserve_space_1, Tensor reserve_space_2, Tensor reserve_space_3, float epsilon = 0.0001f, string data_format = "NHWC", bool is_training = true, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "FusedBatchNormGradV3", name) { args = new object[] { y_backprop, x, scale, reserve_space_1, reserve_space_2, reserve_space_3 }, attrs = new Dictionary<string, object>() { ["epsilon"] = epsilon, ["data_format"] = data_format, ["is_training"] = is_training } });
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
                return fused_batch_norm_grad_v3_eager_fallback(y_backprop, x, scale, reserve_space_1, reserve_space_2, reserve_space_3, epsilon: epsilon, data_format: data_format, is_training: is_training, name: name, ctx: _ctx);
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
        keywords["y_backprop"] = y_backprop;
        keywords["x"] = x;
        keywords["scale"] = scale;
        keywords["reserve_space_1"] = reserve_space_1;
        keywords["reserve_space_2"] = reserve_space_2;
        keywords["reserve_space_3"] = reserve_space_3;
        keywords["epsilon"] = epsilon;
        keywords["data_format"] = data_format;
        keywords["is_training"] = is_training;
        var _op = tf.OpDefLib._apply_op_helper("FusedBatchNormGradV3", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "U", _op._get_attr_type("U"), "epsilon", _op.get_attr("epsilon"), "data_format", _op.get_attr("data_format"), "is_training", _op._get_attr_bool("is_training") };
            _execute.record_gradient("FusedBatchNormGradV3", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] fused_batch_norm_grad_v3_eager_fallback(Tensor y_backprop, Tensor x, Tensor scale, Tensor reserve_space_1, Tensor reserve_space_2, Tensor reserve_space_3, float epsilon, string data_format, bool is_training, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { y_backprop, x, scale, reserve_space_1, reserve_space_2, reserve_space_3 };
        object[] _attrs = new object[] { "T", y_backprop.dtype, "U", reserve_space_1.dtype, "epsilon", epsilon, "data_format", data_format, "is_training", is_training };
        var _result = _execute.execute("FusedBatchNormGradV3", 5, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("FusedBatchNormGradV3", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Batch normalization.
    /// </summary>
    /// <remarks>
    /// 
    /// Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
    /// The size of 1D Tensors matches the dimension C of the 4D Tensors.
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="scale"></param>
    /// <param name="offset"></param>
    /// <param name="mean"></param>
    /// <param name="variance"></param>
    /// <param name="epsilon">
    /// 
    /// A small float number added to the variance of x.
    /// 
    /// </param>
    /// <param name="exponential_avg_factor"></param>
    /// <param name="data_format">
    /// 
    /// The data format for x and y. Either "NHWC" (default) or "NCHW".
    /// 
    /// </param>
    /// <param name="is_training">
    /// 
    /// A bool value to indicate the operation is for training (default)
    /// or inference.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] fused_batch_norm_v2(Tensor x, Tensor scale, Tensor offset, Tensor mean, Tensor variance, float epsilon = 0.0001f, float exponential_avg_factor = 1f, string data_format = "NHWC", bool is_training = true, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "FusedBatchNormV2", name) { args = new object[] { x, scale, offset, mean, variance }, attrs = new Dictionary<string, object>() { ["epsilon"] = epsilon, ["exponential_avg_factor"] = exponential_avg_factor, ["data_format"] = data_format, ["is_training"] = is_training } });
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
                return fused_batch_norm_v2_eager_fallback(x, scale, offset, mean, variance, epsilon: epsilon, exponential_avg_factor: exponential_avg_factor, data_format: data_format, is_training: is_training, name: name, ctx: _ctx);
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
        keywords["x"] = x;
        keywords["scale"] = scale;
        keywords["offset"] = offset;
        keywords["mean"] = mean;
        keywords["variance"] = variance;
        keywords["epsilon"] = epsilon;
        keywords["exponential_avg_factor"] = exponential_avg_factor;
        keywords["data_format"] = data_format;
        keywords["is_training"] = is_training;
        var _op = tf.OpDefLib._apply_op_helper("FusedBatchNormV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "U", _op._get_attr_type("U"), "epsilon", _op.get_attr("epsilon"), "exponential_avg_factor", _op.get_attr("exponential_avg_factor"), "data_format", _op.get_attr("data_format"), "is_training", _op._get_attr_bool("is_training") };
            _execute.record_gradient("FusedBatchNormV2", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] fused_batch_norm_v2_eager_fallback(Tensor x, Tensor scale, Tensor offset, Tensor mean, Tensor variance, float epsilon, float exponential_avg_factor, string data_format, bool is_training, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, scale, offset, mean, variance };
        object[] _attrs = new object[] { "T", x.dtype, "U", scale.dtype, "epsilon", epsilon, "exponential_avg_factor", exponential_avg_factor, "data_format", data_format, "is_training", is_training };
        var _result = _execute.execute("FusedBatchNormV2", 5, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("FusedBatchNormV2", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Batch normalization.
    /// </summary>
    /// <remarks>
    /// 
    /// Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
    /// The size of 1D Tensors matches the dimension C of the 4D Tensors.
    /// 
    /// </remarks>
    /// <param name="x"></param>
    /// <param name="scale"></param>
    /// <param name="offset"></param>
    /// <param name="mean"></param>
    /// <param name="variance"></param>
    /// <param name="epsilon">
    /// 
    /// A small float number added to the variance of x.
    /// 
    /// </param>
    /// <param name="exponential_avg_factor"></param>
    /// <param name="data_format">
    /// 
    /// The data format for x and y. Either "NHWC" (default) or "NCHW".
    /// 
    /// </param>
    /// <param name="is_training">
    /// 
    /// A bool value to indicate the operation is for training (default)
    /// or inference.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] fused_batch_norm_v3(Tensor x, Tensor scale, Tensor offset, Tensor mean, Tensor variance, float epsilon = 0.0001f, float exponential_avg_factor = 1f, string data_format = "NHWC", bool is_training = true, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "FusedBatchNormV3", name) { args = new object[] { x, scale, offset, mean, variance }, attrs = new Dictionary<string, object>() { ["epsilon"] = epsilon, ["exponential_avg_factor"] = exponential_avg_factor, ["data_format"] = data_format, ["is_training"] = is_training } });
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
                return fused_batch_norm_v3_eager_fallback(x, scale, offset, mean, variance, epsilon: epsilon, exponential_avg_factor: exponential_avg_factor, data_format: data_format, is_training: is_training, name: name, ctx: _ctx);
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
        keywords["x"] = x;
        keywords["scale"] = scale;
        keywords["offset"] = offset;
        keywords["mean"] = mean;
        keywords["variance"] = variance;
        keywords["epsilon"] = epsilon;
        keywords["exponential_avg_factor"] = exponential_avg_factor;
        keywords["data_format"] = data_format;
        keywords["is_training"] = is_training;
        var _op = tf.OpDefLib._apply_op_helper("FusedBatchNormV3", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "U", _op._get_attr_type("U"), "epsilon", _op.get_attr("epsilon"), "exponential_avg_factor", _op.get_attr("exponential_avg_factor"), "data_format", _op.get_attr("data_format"), "is_training", _op._get_attr_bool("is_training") };
            _execute.record_gradient("FusedBatchNormV3", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] fused_batch_norm_v3_eager_fallback(Tensor x, Tensor scale, Tensor offset, Tensor mean, Tensor variance, float epsilon, float exponential_avg_factor, string data_format, bool is_training, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { x, scale, offset, mean, variance };
        object[] _attrs = new object[] { "T", x.dtype, "U", scale.dtype, "epsilon", epsilon, "exponential_avg_factor", exponential_avg_factor, "data_format", data_format, "is_training", is_training };
        var _result = _execute.execute("FusedBatchNormV3", 6, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("FusedBatchNormV3", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Performs a padding as a preprocess during a convolution.
    /// </summary>
    /// <remarks>
    /// 
    /// Similar to FusedResizeAndPadConv2d, this op allows for an optimized
    /// implementation where the spatial padding transformation stage is fused with the
    /// im2col lookup, but in this case without the bilinear filtering required for
    /// resizing. Fusing the padding prevents the need to write out the intermediate
    /// results as whole tensors, reducing memory pressure, and we can get some latency
    /// gains by merging the transformation calculations.
    /// The data_format attribute for Conv2D isn't supported by this op, and 'NHWC'
    /// order is used instead.
    /// Internally this op uses a single per-graph scratch buffer, which means that it
    /// will block if multiple versions are being run in parallel. This is because this
    /// operator is primarily an optimization to minimize memory usage.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="paddings"></param>
    /// <param name="filter"></param>
    /// <param name="mode"></param>
    /// <param name="strides">
    /// 
    /// 1-D of length 4.  The stride of the sliding window for each dimension
    /// of `input`. Must be in the same order as the dimension specified with format.
    /// 
    /// </param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor fused_pad_conv2d(Tensor input, Tensor paddings, Tensor filter, string mode, int[] strides, string padding, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "FusedPadConv2D", name) { args = new object[] { input, paddings, filter }, attrs = new Dictionary<string, object>() { ["mode"] = mode, ["strides"] = strides, ["padding"] = padding } });
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
                return fused_pad_conv2d_eager_fallback(input, paddings, filter, mode: mode, strides: strides, padding: padding, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["paddings"] = paddings;
        keywords["filter"] = filter;
        keywords["mode"] = mode;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        var _op = tf.OpDefLib._apply_op_helper("FusedPadConv2D", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "mode", _op.get_attr("mode"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding") };
            _execute.record_gradient("FusedPadConv2D", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor fused_pad_conv2d_eager_fallback(Tensor input, Tensor paddings, Tensor filter, string mode, int[] strides, string padding, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, paddings, filter };
        object[] _attrs = new object[] { "T", input.dtype, "mode", mode, "strides", strides, "padding", padding };
        var _result = _execute.execute("FusedPadConv2D", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("FusedPadConv2D", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Performs a resize and padding as a preprocess during a convolution.
    /// </summary>
    /// <remarks>
    /// 
    /// It's often possible to do spatial transformations more efficiently as part of
    /// the packing stage of a convolution, so this op allows for an optimized
    /// implementation where these stages are fused together. This prevents the need to
    /// write out the intermediate results as whole tensors, reducing memory pressure,
    /// and we can get some latency gains by merging the transformation calculations.
    /// The data_format attribute for Conv2D isn't supported by this op, and defaults to
    /// 'NHWC' order.
    /// Internally this op uses a single per-graph scratch buffer, which means that it
    /// will block if multiple versions are being run in parallel. This is because this
    /// operator is primarily an optimization to minimize memory usage.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="size"></param>
    /// <param name="paddings"></param>
    /// <param name="filter"></param>
    /// <param name="resize_align_corners">
    /// 
    /// If true, the centers of the 4 corner pixels of the input and output tensors are
    /// aligned, preserving the values at the corner pixels. Defaults to false.
    /// 
    /// </param>
    /// <param name="mode"></param>
    /// <param name="strides">
    /// 
    /// 1-D of length 4.  The stride of the sliding window for each dimension
    /// of `input`. Must be in the same order as the dimension specified with format.
    /// 
    /// </param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor fused_resize_and_pad_conv2d(Tensor input, Tensor size, Tensor paddings, Tensor filter, string mode, int[] strides, string padding, bool resize_align_corners = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "FusedResizeAndPadConv2D", name) { args = new object[] { input, size, paddings, filter }, attrs = new Dictionary<string, object>() { ["resize_align_corners"] = resize_align_corners, ["mode"] = mode, ["strides"] = strides, ["padding"] = padding } });
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
                return fused_resize_and_pad_conv2d_eager_fallback(input, size, paddings, filter, resize_align_corners: resize_align_corners, mode: mode, strides: strides, padding: padding, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["size"] = size;
        keywords["paddings"] = paddings;
        keywords["filter"] = filter;
        keywords["resize_align_corners"] = resize_align_corners;
        keywords["mode"] = mode;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        var _op = tf.OpDefLib._apply_op_helper("FusedResizeAndPadConv2D", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "resize_align_corners", _op._get_attr_bool("resize_align_corners"), "mode", _op.get_attr("mode"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding") };
            _execute.record_gradient("FusedResizeAndPadConv2D", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor fused_resize_and_pad_conv2d_eager_fallback(Tensor input, Tensor size, Tensor paddings, Tensor filter, bool resize_align_corners, string mode, int[] strides, string padding, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, size, paddings, filter };
        object[] _attrs = new object[] { "T", input.dtype, "resize_align_corners", resize_align_corners, "mode", mode, "strides", strides, "padding", padding };
        var _result = _execute.execute("FusedResizeAndPadConv2D", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("FusedResizeAndPadConv2D", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Says whether the targets are in the top `K` predictions.
    /// </summary>
    /// <remarks>
    /// 
    /// This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the
    /// prediction for the target class is among the top `k` predictions among
    /// all predictions for example `i`. Note that the behavior of `InTopK` differs
    /// from the `TopK` op in its handling of ties; if multiple classes have the
    /// same prediction value and straddle the top-`k` boundary, all of those
    /// classes are considered to be in the top `k`.
    /// 
    /// More formally, let
    /// 
    ///   \(predictions_i\) be the predictions for all classes for example `i`,
    ///   \(targets_i\) be the target class for example `i`,
    ///   \(out_i\) be the output for example `i`,
    /// 
    /// $$out_i = predictions_{i, targets_i} in TopKIncludingTies(predictions_i)$$
    /// 
    /// </remarks>
    /// <param name="predictions"></param>
    /// <param name="targets"></param>
    /// <param name="k">
    /// 
    /// Number of top elements to look at for computing precision.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor in_top_k(Tensor predictions, Tensor targets, int k = 0, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "InTopK", name) { args = new object[] { predictions, targets }, attrs = new Dictionary<string, object>() { ["k"] = k } });
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
                return in_top_k_eager_fallback(predictions, targets, k: k, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["predictions"] = predictions;
        keywords["targets"] = targets;
        keywords["k"] = k;
        var _op = tf.OpDefLib._apply_op_helper("InTopK", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "k", _op._get_attr_int("k"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("InTopK", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor in_top_k_eager_fallback(Tensor predictions, Tensor targets, int k, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { predictions, targets };
        object[] _attrs = new object[] { "k", k, "T", targets.dtype };
        var _result = _execute.execute("InTopK", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("InTopK", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Says whether the targets are in the top `K` predictions.
    /// </summary>
    /// <remarks>
    /// 
    /// This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the
    /// prediction for the target class is among the top `k` predictions among
    /// all predictions for example `i`. Note that the behavior of `InTopK` differs
    /// from the `TopK` op in its handling of ties; if multiple classes have the
    /// same prediction value and straddle the top-`k` boundary, all of those
    /// classes are considered to be in the top `k`.
    /// 
    /// More formally, let
    /// 
    ///   \(predictions_i\) be the predictions for all classes for example `i`,
    ///   \(targets_i\) be the target class for example `i`,
    ///   \(out_i\) be the output for example `i`,
    /// 
    /// $$out_i = predictions_{i, targets_i} in TopKIncludingTies(predictions_i)$$
    /// 
    /// </remarks>
    /// <param name="predictions"></param>
    /// <param name="targets"></param>
    /// <param name="k"></param>
    /// <returns></returns>
    public static Tensor in_top_kv2(Tensor predictions, Tensor targets, Tensor k, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "InTopKV2", name) { args = new object[] { predictions, targets, k }, attrs = new Dictionary<string, object>() { } });
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
                return in_top_kv2_eager_fallback(predictions, targets, k, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["predictions"] = predictions;
        keywords["targets"] = targets;
        keywords["k"] = k;
        var _op = tf.OpDefLib._apply_op_helper("InTopKV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("InTopKV2", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor in_top_kv2_eager_fallback(Tensor predictions, Tensor targets, Tensor k, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { predictions, targets, k };
        object[] _attrs = new object[] { "T", targets.dtype };
        var _result = _execute.execute("InTopKV2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("InTopKV2", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Solves a batch of isotonic regression problems.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="output_dtype">
    /// Dtype of output.
    /// </param>
    /// <returns></returns>
    public static Tensor[] isotonic_regression(Tensor input, TF_DataType output_dtype = TF_DataType.TF_FLOAT, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "IsotonicRegression", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { ["output_dtype"] = output_dtype } });
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
                return isotonic_regression_eager_fallback(input, output_dtype: output_dtype, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["output_dtype"] = output_dtype;
        var _op = tf.OpDefLib._apply_op_helper("IsotonicRegression", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "output_dtype", _op._get_attr_type("output_dtype") };
            _execute.record_gradient("IsotonicRegression", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] isotonic_regression_eager_fallback(Tensor input, TF_DataType output_dtype, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "T", input.dtype, "output_dtype", output_dtype };
        var _result = _execute.execute("IsotonicRegression", 2, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("IsotonicRegression", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Local Response Normalization.
    /// </summary>
    /// <remarks>
    /// 
    /// The 4-D `input` tensor is treated as a 3-D array of 1-D vectors (along the last
    /// dimension), and each vector is normalized independently.  Within a given vector,
    /// each component is divided by the weighted, squared sum of inputs within
    /// `depth_radius`.  In detail,
    /// 
    ///     sqr_sum[a, b, c, d] =
    ///         sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
    ///     output = input / (bias + alpha * sqr_sum) ** beta
    /// 
    /// For details, see [Krizhevsky et al., ImageNet classification with deep
    /// convolutional neural networks (NIPS 2012)](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="depth_radius">
    /// 
    /// 0-D.  Half-width of the 1-D normalization window.
    /// 
    /// </param>
    /// <param name="bias">
    /// 
    /// An offset (usually positive to avoid dividing by 0).
    /// 
    /// </param>
    /// <param name="alpha">
    /// 
    /// A scale factor, usually positive.
    /// 
    /// </param>
    /// <param name="beta">
    /// 
    /// An exponent.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor lrn(Tensor input, int depth_radius = 5, float bias = 1f, float alpha = 1f, float beta = 0.5f, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "LRN", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { ["depth_radius"] = depth_radius, ["bias"] = bias, ["alpha"] = alpha, ["beta"] = beta } });
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
                return lrn_eager_fallback(input, depth_radius: depth_radius, bias: bias, alpha: alpha, beta: beta, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["depth_radius"] = depth_radius;
        keywords["bias"] = bias;
        keywords["alpha"] = alpha;
        keywords["beta"] = beta;
        var _op = tf.OpDefLib._apply_op_helper("LRN", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "depth_radius", _op._get_attr_int("depth_radius"), "bias", _op.get_attr("bias"), "alpha", _op.get_attr("alpha"), "beta", _op.get_attr("beta"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("LRN", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor lrn_eager_fallback(Tensor input, int depth_radius, float bias, float alpha, float beta, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "depth_radius", depth_radius, "bias", bias, "alpha", alpha, "beta", beta, "T", input.dtype };
        var _result = _execute.execute("LRN", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("LRN", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes rectified linear: `max(features, features * alpha)`.
    /// </summary>
    /// <param name="features"></param>
    /// <param name="alpha"></param>
    /// <returns></returns>
    public static Tensor leaky_relu(Tensor features, float alpha = 0.2f, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "LeakyRelu", name) { args = new object[] { features }, attrs = new Dictionary<string, object>() { ["alpha"] = alpha } });
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
                return leaky_relu_eager_fallback(features, alpha: alpha, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["features"] = features;
        keywords["alpha"] = alpha;
        var _op = tf.OpDefLib._apply_op_helper("LeakyRelu", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "alpha", _op.get_attr("alpha"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("LeakyRelu", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor leaky_relu_eager_fallback(Tensor features, float alpha, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { features };
        object[] _attrs = new object[] { "alpha", alpha, "T", features.dtype };
        var _result = _execute.execute("LeakyRelu", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("LeakyRelu", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes rectified linear gradients for a LeakyRelu operation.
    /// </summary>
    /// <param name="gradients"></param>
    /// <param name="features"></param>
    /// <param name="alpha"></param>
    /// <returns></returns>
    public static Tensor leaky_relu_grad(Tensor gradients, Tensor features, float alpha = 0.2f, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "LeakyReluGrad", name) { args = new object[] { gradients, features }, attrs = new Dictionary<string, object>() { ["alpha"] = alpha } });
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
                return leaky_relu_grad_eager_fallback(gradients, features, alpha: alpha, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["gradients"] = gradients;
        keywords["features"] = features;
        keywords["alpha"] = alpha;
        var _op = tf.OpDefLib._apply_op_helper("LeakyReluGrad", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "alpha", _op.get_attr("alpha"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("LeakyReluGrad", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor leaky_relu_grad_eager_fallback(Tensor gradients, Tensor features, float alpha, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { gradients, features };
        object[] _attrs = new object[] { "alpha", alpha, "T", gradients.dtype };
        var _result = _execute.execute("LeakyReluGrad", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("LeakyReluGrad", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes log softmax activations.
    /// </summary>
    /// <remarks>
    /// 
    /// For each batch `i` and class `j` we have
    /// 
    ///     logsoftmax[i, j] = logits[i, j] - log(sum(exp(logits[i])))
    /// 
    /// </remarks>
    /// <param name="logits"></param>
    /// <returns></returns>
    public static Tensor log_softmax(Tensor logits, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "LogSoftmax", name) { args = new object[] { logits }, attrs = new Dictionary<string, object>() { } });
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
                return log_softmax_eager_fallback(logits, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["logits"] = logits;
        var _op = tf.OpDefLib._apply_op_helper("LogSoftmax", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("LogSoftmax", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor log_softmax_eager_fallback(Tensor logits, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { logits };
        object[] _attrs = new object[] { "T", logits.dtype };
        var _result = _execute.execute("LogSoftmax", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("LogSoftmax", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Performs max pooling on the input.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="ksize">
    /// 
    /// The size of the window for each dimension of the input tensor.
    /// 
    /// </param>
    /// <param name="strides">
    /// 
    /// The stride of the sliding window for each dimension of the
    /// input tensor.
    /// 
    /// </param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <param name="explicit_paddings"></param>
    /// <param name="data_format">
    /// 
    /// Specify the data format of the input and output data. With the
    /// default format "NHWC", the data is stored in the order of:
    ///     [batch, in_height, in_width, in_channels].
    /// Alternatively, the format could be "NCHW", the data storage order of:
    ///     [batch, in_channels, in_height, in_width].
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor max_pool(Tensor input, int[] ksize, int[] strides, string padding, int[] explicit_paddings = null, string data_format = "NHWC", string? name = null)
    {
        var _ctx = tf.Context;
        if (explicit_paddings is null)
        {
            explicit_paddings = new int[] { };
        }
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "MaxPool", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { ["ksize"] = ksize, ["strides"] = strides, ["padding"] = padding, ["explicit_paddings"] = explicit_paddings, ["data_format"] = data_format } });
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
                return max_pool_eager_fallback(input, ksize: ksize, strides: strides, padding: padding, explicit_paddings: explicit_paddings, data_format: data_format, name: name, ctx: _ctx);
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
        keywords["ksize"] = ksize;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["explicit_paddings"] = explicit_paddings;
        keywords["data_format"] = data_format;
        var _op = tf.OpDefLib._apply_op_helper("MaxPool", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "ksize", _op.get_attr("ksize"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "explicit_paddings", _op.get_attr("explicit_paddings"), "data_format", _op.get_attr("data_format") };
            _execute.record_gradient("MaxPool", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor max_pool_eager_fallback(Tensor input, int[] ksize, int[] strides, string padding, int[] explicit_paddings, string data_format, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "T", input.dtype, "ksize", ksize, "strides", strides, "padding", padding, "explicit_paddings", explicit_paddings, "data_format", data_format };
        var _result = _execute.execute("MaxPool", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("MaxPool", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Performs 3D max pooling on the input.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="ksize">
    /// 
    /// 1-D tensor of length 5. The size of the window for each dimension of
    /// the input tensor. Must have `ksize[0] = ksize[4] = 1`.
    /// 
    /// </param>
    /// <param name="strides">
    /// 
    /// 1-D tensor of length 5. The stride of the sliding window for each
    /// dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    /// 
    /// </param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <param name="data_format">
    /// 
    /// The data format of the input and output data. With the
    /// default format "NDHWC", the data is stored in the order of:
    ///     [batch, in_depth, in_height, in_width, in_channels].
    /// Alternatively, the format could be "NCDHW", the data storage order is:
    ///     [batch, in_channels, in_depth, in_height, in_width].
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor max_pool3d(Tensor input, int[] ksize, int[] strides, string padding, string data_format = "NDHWC", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "MaxPool3D", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { ["ksize"] = ksize, ["strides"] = strides, ["padding"] = padding, ["data_format"] = data_format } });
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
                return max_pool3d_eager_fallback(input, ksize: ksize, strides: strides, padding: padding, data_format: data_format, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        if (data_format is null)
        {
            data_format = "NDHWC";
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["ksize"] = ksize;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["data_format"] = data_format;
        var _op = tf.OpDefLib._apply_op_helper("MaxPool3D", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "ksize", _op.get_attr("ksize"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "data_format", _op.get_attr("data_format"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("MaxPool3D", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor max_pool3d_eager_fallback(Tensor input, int[] ksize, int[] strides, string padding, string data_format, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "ksize", ksize, "strides", strides, "padding", padding, "data_format", data_format, "T", input.dtype };
        var _result = _execute.execute("MaxPool3D", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("MaxPool3D", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes gradients of 3D max pooling function.
    /// </summary>
    /// <param name="orig_input"></param>
    /// <param name="orig_output"></param>
    /// <param name="grad"></param>
    /// <param name="ksize">
    /// 
    /// 1-D tensor of length 5. The size of the window for each dimension of
    /// the input tensor. Must have `ksize[0] = ksize[4] = 1`.
    /// 
    /// </param>
    /// <param name="strides">
    /// 
    /// 1-D tensor of length 5. The stride of the sliding window for each
    /// dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    /// 
    /// </param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <param name="data_format">
    /// 
    /// The data format of the input and output data. With the
    /// default format "NDHWC", the data is stored in the order of:
    ///     [batch, in_depth, in_height, in_width, in_channels].
    /// Alternatively, the format could be "NCDHW", the data storage order is:
    ///     [batch, in_channels, in_depth, in_height, in_width].
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor max_pool3d_grad(Tensor orig_input, Tensor orig_output, Tensor grad, int[] ksize, int[] strides, string padding, string data_format = "NDHWC", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "MaxPool3DGrad", name) { args = new object[] { orig_input, orig_output, grad }, attrs = new Dictionary<string, object>() { ["ksize"] = ksize, ["strides"] = strides, ["padding"] = padding, ["data_format"] = data_format } });
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
                return max_pool3d_grad_eager_fallback(orig_input, orig_output, grad, ksize: ksize, strides: strides, padding: padding, data_format: data_format, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        if (data_format is null)
        {
            data_format = "NDHWC";
        }
        Dictionary<string, object> keywords = new();
        keywords["orig_input"] = orig_input;
        keywords["orig_output"] = orig_output;
        keywords["grad"] = grad;
        keywords["ksize"] = ksize;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["data_format"] = data_format;
        var _op = tf.OpDefLib._apply_op_helper("MaxPool3DGrad", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "ksize", _op.get_attr("ksize"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "data_format", _op.get_attr("data_format"), "T", _op._get_attr_type("T"), "TInput", _op._get_attr_type("TInput") };
            _execute.record_gradient("MaxPool3DGrad", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor max_pool3d_grad_eager_fallback(Tensor orig_input, Tensor orig_output, Tensor grad, int[] ksize, int[] strides, string padding, string data_format, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { orig_input, orig_output, grad };
        object[] _attrs = new object[] { "ksize", ksize, "strides", strides, "padding", padding, "data_format", data_format, "T", grad.dtype, "TInput", orig_input.dtype };
        var _result = _execute.execute("MaxPool3DGrad", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("MaxPool3DGrad", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes second-order gradients of the maxpooling function.
    /// </summary>
    /// <param name="orig_input"></param>
    /// <param name="orig_output"></param>
    /// <param name="grad"></param>
    /// <param name="ksize">
    /// 
    /// 1-D tensor of length 5. The size of the window for each dimension of
    /// the input tensor. Must have `ksize[0] = ksize[4] = 1`.
    /// 
    /// </param>
    /// <param name="strides">
    /// 
    /// 1-D tensor of length 5. The stride of the sliding window for each
    /// dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    /// 
    /// </param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <param name="data_format">
    /// 
    /// The data format of the input and output data. With the
    /// default format "NDHWC", the data is stored in the order of:
    ///     [batch, in_depth, in_height, in_width, in_channels].
    /// Alternatively, the format could be "NCDHW", the data storage order is:
    ///     [batch, in_channels, in_depth, in_height, in_width].
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor max_pool3d_grad_grad(Tensor orig_input, Tensor orig_output, Tensor grad, int[] ksize, int[] strides, string padding, string data_format = "NDHWC", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "MaxPool3DGradGrad", name) { args = new object[] { orig_input, orig_output, grad }, attrs = new Dictionary<string, object>() { ["ksize"] = ksize, ["strides"] = strides, ["padding"] = padding, ["data_format"] = data_format } });
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
                return max_pool3d_grad_grad_eager_fallback(orig_input, orig_output, grad, ksize: ksize, strides: strides, padding: padding, data_format: data_format, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        if (data_format is null)
        {
            data_format = "NDHWC";
        }
        Dictionary<string, object> keywords = new();
        keywords["orig_input"] = orig_input;
        keywords["orig_output"] = orig_output;
        keywords["grad"] = grad;
        keywords["ksize"] = ksize;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["data_format"] = data_format;
        var _op = tf.OpDefLib._apply_op_helper("MaxPool3DGradGrad", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "ksize", _op.get_attr("ksize"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "data_format", _op.get_attr("data_format"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("MaxPool3DGradGrad", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor max_pool3d_grad_grad_eager_fallback(Tensor orig_input, Tensor orig_output, Tensor grad, int[] ksize, int[] strides, string padding, string data_format, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { orig_input, orig_output, grad };
        object[] _attrs = new object[] { "ksize", ksize, "strides", strides, "padding", padding, "data_format", data_format, "T", orig_input.dtype };
        var _result = _execute.execute("MaxPool3DGradGrad", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("MaxPool3DGradGrad", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes gradients of the maxpooling function.
    /// </summary>
    /// <param name="orig_input"></param>
    /// <param name="orig_output"></param>
    /// <param name="grad"></param>
    /// <param name="ksize">
    /// 
    /// The size of the window for each dimension of the input tensor.
    /// 
    /// </param>
    /// <param name="strides">
    /// 
    /// The stride of the sliding window for each dimension of the
    /// input tensor.
    /// 
    /// </param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <param name="explicit_paddings"></param>
    /// <param name="data_format">
    /// 
    /// Specify the data format of the input and output data. With the
    /// default format "NHWC", the data is stored in the order of:
    ///     [batch, in_height, in_width, in_channels].
    /// Alternatively, the format could be "NCHW", the data storage order of:
    ///     [batch, in_channels, in_height, in_width].
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor max_pool_grad(Tensor orig_input, Tensor orig_output, Tensor grad, int[] ksize, int[] strides, string padding, int[] explicit_paddings = null, string data_format = "NHWC", string? name = null)
    {
        var _ctx = tf.Context;
        if (explicit_paddings is null)
        {
            explicit_paddings = new int[] { };
        }
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "MaxPoolGrad", name) { args = new object[] { orig_input, orig_output, grad }, attrs = new Dictionary<string, object>() { ["ksize"] = ksize, ["strides"] = strides, ["padding"] = padding, ["explicit_paddings"] = explicit_paddings, ["data_format"] = data_format } });
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
                return max_pool_grad_eager_fallback(orig_input, orig_output, grad, ksize: ksize, strides: strides, padding: padding, explicit_paddings: explicit_paddings, data_format: data_format, name: name, ctx: _ctx);
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
        keywords["orig_input"] = orig_input;
        keywords["orig_output"] = orig_output;
        keywords["grad"] = grad;
        keywords["ksize"] = ksize;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["explicit_paddings"] = explicit_paddings;
        keywords["data_format"] = data_format;
        var _op = tf.OpDefLib._apply_op_helper("MaxPoolGrad", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "ksize", _op.get_attr("ksize"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "explicit_paddings", _op.get_attr("explicit_paddings"), "data_format", _op.get_attr("data_format"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("MaxPoolGrad", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor max_pool_grad_eager_fallback(Tensor orig_input, Tensor orig_output, Tensor grad, int[] ksize, int[] strides, string padding, int[] explicit_paddings, string data_format, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { orig_input, orig_output, grad };
        object[] _attrs = new object[] { "ksize", ksize, "strides", strides, "padding", padding, "explicit_paddings", explicit_paddings, "data_format", data_format, "T", orig_input.dtype };
        var _result = _execute.execute("MaxPoolGrad", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("MaxPoolGrad", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes second-order gradients of the maxpooling function.
    /// </summary>
    /// <param name="orig_input"></param>
    /// <param name="orig_output"></param>
    /// <param name="grad"></param>
    /// <param name="ksize">
    /// 
    /// The size of the window for each dimension of the input tensor.
    /// 
    /// </param>
    /// <param name="strides">
    /// 
    /// The stride of the sliding window for each dimension of the
    /// input tensor.
    /// 
    /// </param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <param name="data_format">
    /// 
    /// Specify the data format of the input and output data. With the
    /// default format "NHWC", the data is stored in the order of:
    ///     [batch, in_height, in_width, in_channels].
    /// Alternatively, the format could be "NCHW", the data storage order of:
    ///     [batch, in_channels, in_height, in_width].
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor max_pool_grad_grad(Tensor orig_input, Tensor orig_output, Tensor grad, int[] ksize, int[] strides, string padding, string data_format = "NHWC", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "MaxPoolGradGrad", name) { args = new object[] { orig_input, orig_output, grad }, attrs = new Dictionary<string, object>() { ["ksize"] = ksize, ["strides"] = strides, ["padding"] = padding, ["data_format"] = data_format } });
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
                return max_pool_grad_grad_eager_fallback(orig_input, orig_output, grad, ksize: ksize, strides: strides, padding: padding, data_format: data_format, name: name, ctx: _ctx);
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
        keywords["orig_input"] = orig_input;
        keywords["orig_output"] = orig_output;
        keywords["grad"] = grad;
        keywords["ksize"] = ksize;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["data_format"] = data_format;
        var _op = tf.OpDefLib._apply_op_helper("MaxPoolGradGrad", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "ksize", _op.get_attr("ksize"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "data_format", _op.get_attr("data_format"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("MaxPoolGradGrad", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor max_pool_grad_grad_eager_fallback(Tensor orig_input, Tensor orig_output, Tensor grad, int[] ksize, int[] strides, string padding, string data_format, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { orig_input, orig_output, grad };
        object[] _attrs = new object[] { "ksize", ksize, "strides", strides, "padding", padding, "data_format", data_format, "T", orig_input.dtype };
        var _result = _execute.execute("MaxPoolGradGrad", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("MaxPoolGradGrad", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes second-order gradients of the maxpooling function.
    /// </summary>
    /// <param name="orig_input"></param>
    /// <param name="orig_output"></param>
    /// <param name="grad"></param>
    /// <param name="ksize"></param>
    /// <param name="strides"></param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <param name="data_format">
    /// 
    /// Specify the data format of the input and output data. With the
    /// default format "NHWC", the data is stored in the order of:
    ///     [batch, in_height, in_width, in_channels].
    /// Alternatively, the format could be "NCHW", the data storage order of:
    ///     [batch, in_channels, in_height, in_width].
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor max_pool_grad_grad_v2(Tensor orig_input, Tensor orig_output, Tensor grad, Tensor ksize, Tensor strides, string padding, string data_format = "NHWC", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "MaxPoolGradGradV2", name) { args = new object[] { orig_input, orig_output, grad, ksize, strides }, attrs = new Dictionary<string, object>() { ["padding"] = padding, ["data_format"] = data_format } });
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
                return max_pool_grad_grad_v2_eager_fallback(orig_input, orig_output, grad, ksize, strides, padding: padding, data_format: data_format, name: name, ctx: _ctx);
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
        keywords["orig_input"] = orig_input;
        keywords["orig_output"] = orig_output;
        keywords["grad"] = grad;
        keywords["ksize"] = ksize;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["data_format"] = data_format;
        var _op = tf.OpDefLib._apply_op_helper("MaxPoolGradGradV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "padding", _op.get_attr("padding"), "data_format", _op.get_attr("data_format"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("MaxPoolGradGradV2", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor max_pool_grad_grad_v2_eager_fallback(Tensor orig_input, Tensor orig_output, Tensor grad, Tensor ksize, Tensor strides, string padding, string data_format, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { orig_input, orig_output, grad, ksize, strides };
        object[] _attrs = new object[] { "padding", padding, "data_format", data_format, "T", orig_input.dtype };
        var _result = _execute.execute("MaxPoolGradGradV2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("MaxPoolGradGradV2", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes second-order gradients of the maxpooling function.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="grad"></param>
    /// <param name="argmax"></param>
    /// <param name="ksize">
    /// 
    /// The size of the window for each dimension of the input tensor.
    /// 
    /// </param>
    /// <param name="strides">
    /// 
    /// The stride of the sliding window for each dimension of the
    /// input tensor.
    /// 
    /// </param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <param name="include_batch_in_index">
    /// 
    /// Whether to include batch dimension in flattened index of `argmax`.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor max_pool_grad_grad_with_argmax(Tensor input, Tensor grad, Tensor argmax, int[] ksize, int[] strides, string padding, bool include_batch_in_index = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "MaxPoolGradGradWithArgmax", name) { args = new object[] { input, grad, argmax }, attrs = new Dictionary<string, object>() { ["ksize"] = ksize, ["strides"] = strides, ["padding"] = padding, ["include_batch_in_index"] = include_batch_in_index } });
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
                return max_pool_grad_grad_with_argmax_eager_fallback(input, grad, argmax, ksize: ksize, strides: strides, padding: padding, include_batch_in_index: include_batch_in_index, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["grad"] = grad;
        keywords["argmax"] = argmax;
        keywords["ksize"] = ksize;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["include_batch_in_index"] = include_batch_in_index;
        var _op = tf.OpDefLib._apply_op_helper("MaxPoolGradGradWithArgmax", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "ksize", _op.get_attr("ksize"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "include_batch_in_index", _op._get_attr_bool("include_batch_in_index"), "Targmax", _op._get_attr_type("Targmax"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("MaxPoolGradGradWithArgmax", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor max_pool_grad_grad_with_argmax_eager_fallback(Tensor input, Tensor grad, Tensor argmax, int[] ksize, int[] strides, string padding, bool include_batch_in_index, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, grad, argmax };
        object[] _attrs = new object[] { "ksize", ksize, "strides", strides, "padding", padding, "include_batch_in_index", include_batch_in_index, "Targmax", argmax.dtype, "T", input.dtype };
        var _result = _execute.execute("MaxPoolGradGradWithArgmax", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("MaxPoolGradGradWithArgmax", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes gradients of the maxpooling function.
    /// </summary>
    /// <param name="orig_input"></param>
    /// <param name="orig_output"></param>
    /// <param name="grad"></param>
    /// <param name="ksize"></param>
    /// <param name="strides"></param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <param name="data_format">
    /// 
    /// Specify the data format of the input and output data. With the
    /// default format "NHWC", the data is stored in the order of:
    ///     [batch, in_height, in_width, in_channels].
    /// Alternatively, the format could be "NCHW", the data storage order of:
    ///     [batch, in_channels, in_height, in_width].
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor max_pool_grad_v2(Tensor orig_input, Tensor orig_output, Tensor grad, Tensor ksize, Tensor strides, string padding, string data_format = "NHWC", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "MaxPoolGradV2", name) { args = new object[] { orig_input, orig_output, grad, ksize, strides }, attrs = new Dictionary<string, object>() { ["padding"] = padding, ["data_format"] = data_format } });
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
                return max_pool_grad_v2_eager_fallback(orig_input, orig_output, grad, ksize, strides, padding: padding, data_format: data_format, name: name, ctx: _ctx);
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
        keywords["orig_input"] = orig_input;
        keywords["orig_output"] = orig_output;
        keywords["grad"] = grad;
        keywords["ksize"] = ksize;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["data_format"] = data_format;
        var _op = tf.OpDefLib._apply_op_helper("MaxPoolGradV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "padding", _op.get_attr("padding"), "data_format", _op.get_attr("data_format"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("MaxPoolGradV2", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor max_pool_grad_v2_eager_fallback(Tensor orig_input, Tensor orig_output, Tensor grad, Tensor ksize, Tensor strides, string padding, string data_format, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { orig_input, orig_output, grad, ksize, strides };
        object[] _attrs = new object[] { "padding", padding, "data_format", data_format, "T", orig_input.dtype };
        var _result = _execute.execute("MaxPoolGradV2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("MaxPoolGradV2", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes gradients of the maxpooling function.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="grad"></param>
    /// <param name="argmax"></param>
    /// <param name="ksize">
    /// 
    /// The size of the window for each dimension of the input tensor.
    /// 
    /// </param>
    /// <param name="strides">
    /// 
    /// The stride of the sliding window for each dimension of the
    /// input tensor.
    /// 
    /// </param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <param name="include_batch_in_index">
    /// 
    /// Whether to include batch dimension in flattened index of `argmax`.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor max_pool_grad_with_argmax(Tensor input, Tensor grad, Tensor argmax, int[] ksize, int[] strides, string padding, bool include_batch_in_index = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "MaxPoolGradWithArgmax", name) { args = new object[] { input, grad, argmax }, attrs = new Dictionary<string, object>() { ["ksize"] = ksize, ["strides"] = strides, ["padding"] = padding, ["include_batch_in_index"] = include_batch_in_index } });
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
                return max_pool_grad_with_argmax_eager_fallback(input, grad, argmax, ksize: ksize, strides: strides, padding: padding, include_batch_in_index: include_batch_in_index, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["grad"] = grad;
        keywords["argmax"] = argmax;
        keywords["ksize"] = ksize;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["include_batch_in_index"] = include_batch_in_index;
        var _op = tf.OpDefLib._apply_op_helper("MaxPoolGradWithArgmax", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "ksize", _op.get_attr("ksize"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "include_batch_in_index", _op._get_attr_bool("include_batch_in_index"), "Targmax", _op._get_attr_type("Targmax"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("MaxPoolGradWithArgmax", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor max_pool_grad_with_argmax_eager_fallback(Tensor input, Tensor grad, Tensor argmax, int[] ksize, int[] strides, string padding, bool include_batch_in_index, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, grad, argmax };
        object[] _attrs = new object[] { "ksize", ksize, "strides", strides, "padding", padding, "include_batch_in_index", include_batch_in_index, "Targmax", argmax.dtype, "T", input.dtype };
        var _result = _execute.execute("MaxPoolGradWithArgmax", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("MaxPoolGradWithArgmax", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Performs max pooling on the input.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="ksize"></param>
    /// <param name="strides"></param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <param name="data_format">
    /// 
    /// Specify the data format of the input and output data. With the
    /// default format "NHWC", the data is stored in the order of:
    ///     [batch, in_height, in_width, in_channels].
    /// Alternatively, the format could be "NCHW", the data storage order of:
    ///     [batch, in_channels, in_height, in_width].
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor max_pool_v2(Tensor input, Tensor ksize, Tensor strides, string padding, string data_format = "NHWC", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "MaxPoolV2", name) { args = new object[] { input, ksize, strides }, attrs = new Dictionary<string, object>() { ["padding"] = padding, ["data_format"] = data_format } });
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
                return max_pool_v2_eager_fallback(input, ksize, strides, padding: padding, data_format: data_format, name: name, ctx: _ctx);
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
        keywords["ksize"] = ksize;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["data_format"] = data_format;
        var _op = tf.OpDefLib._apply_op_helper("MaxPoolV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "padding", _op.get_attr("padding"), "data_format", _op.get_attr("data_format") };
            _execute.record_gradient("MaxPoolV2", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor max_pool_v2_eager_fallback(Tensor input, Tensor ksize, Tensor strides, string padding, string data_format, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, ksize, strides };
        object[] _attrs = new object[] { "T", input.dtype, "padding", padding, "data_format", data_format };
        var _result = _execute.execute("MaxPoolV2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("MaxPoolV2", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Performs max pooling on the input and outputs both max values and indices.
    /// </summary>
    /// <remarks>
    /// 
    /// The indices in `argmax` are flattened, so that a maximum value at position
    /// `[b, y, x, c]` becomes flattened index:
    /// `(y * width + x) * channels + c` if `include_batch_in_index` is False;
    /// `((b * height + y) * width + x) * channels + c` if `include_batch_in_index` is True.
    /// 
    /// The indices returned are always in `[0, height) x [0, width)` before flattening,
    /// even if padding is involved and the mathematically correct answer is outside
    /// (either negative or too large).  This is a bug, but fixing it is difficult to do
    /// in a safe backwards compatible way, especially due to flattening.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="ksize">
    /// 
    /// The size of the window for each dimension of the input tensor.
    /// 
    /// </param>
    /// <param name="strides">
    /// 
    /// The stride of the sliding window for each dimension of the
    /// input tensor.
    /// 
    /// </param>
    /// <param name="Targmax"></param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <param name="include_batch_in_index">
    /// 
    /// Whether to include batch dimension in flattened index of `argmax`.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] max_pool_with_argmax(Tensor input, int[] ksize, int[] strides, string padding, TF_DataType Targmax = TF_DataType.TF_INT64, bool include_batch_in_index = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "MaxPoolWithArgmax", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { ["ksize"] = ksize, ["strides"] = strides, ["Targmax"] = Targmax, ["padding"] = padding, ["include_batch_in_index"] = include_batch_in_index } });
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
                return max_pool_with_argmax_eager_fallback(input, ksize: ksize, strides: strides, Targmax: Targmax, padding: padding, include_batch_in_index: include_batch_in_index, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["ksize"] = ksize;
        keywords["strides"] = strides;
        keywords["Targmax"] = Targmax;
        keywords["padding"] = padding;
        keywords["include_batch_in_index"] = include_batch_in_index;
        var _op = tf.OpDefLib._apply_op_helper("MaxPoolWithArgmax", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "ksize", _op.get_attr("ksize"), "strides", _op.get_attr("strides"), "Targmax", _op._get_attr_type("Targmax"), "padding", _op.get_attr("padding"), "include_batch_in_index", _op._get_attr_bool("include_batch_in_index"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("MaxPoolWithArgmax", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] max_pool_with_argmax_eager_fallback(Tensor input, int[] ksize, int[] strides, TF_DataType Targmax, string padding, bool include_batch_in_index, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "ksize", ksize, "strides", strides, "Targmax", Targmax, "padding", padding, "include_batch_in_index", include_batch_in_index, "T", input.dtype };
        var _result = _execute.execute("MaxPoolWithArgmax", 2, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("MaxPoolWithArgmax", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Finds values of the `n`-th order statistic for the last dimension.
    /// </summary>
    /// <remarks>
    /// 
    /// If the input is a vector (rank-1), finds the entries which is the nth-smallest
    /// value in the vector and outputs their values as scalar tensor.
    /// 
    /// For matrices (resp. higher rank input), computes the entries which is the
    /// nth-smallest value in each row (resp. vector along the last dimension). Thus,
    /// 
    ///     values.shape = input.shape[:-1]
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="n"></param>
    /// <param name="reverse">
    /// 
    /// When set to True, find the nth-largest value in the vector and vice
    /// versa.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor nth_element(Tensor input, Tensor n, bool reverse = false, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "NthElement", name) { args = new object[] { input, n }, attrs = new Dictionary<string, object>() { ["reverse"] = reverse } });
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
                return nth_element_eager_fallback(input, n, reverse: reverse, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["n"] = n;
        keywords["reverse"] = reverse;
        var _op = tf.OpDefLib._apply_op_helper("NthElement", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "reverse", _op._get_attr_bool("reverse"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("NthElement", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor nth_element_eager_fallback(Tensor input, Tensor n, bool reverse, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, n };
        object[] _attrs = new object[] { "reverse", reverse, "T", input.dtype };
        var _result = _execute.execute("NthElement", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("NthElement", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Produces the average pool of the input tensor for quantized types.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="min_input"></param>
    /// <param name="max_input"></param>
    /// <param name="ksize">
    /// 
    /// The size of the window for each dimension of the input tensor.
    /// The length must be 4 to match the number of dimensions of the input.
    /// 
    /// </param>
    /// <param name="strides">
    /// 
    /// The stride of the sliding window for each dimension of the input
    /// tensor.  The length must be 4 to match the number of dimensions of the input.
    /// 
    /// </param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] quantized_avg_pool(Tensor input, Tensor min_input, Tensor max_input, int[] ksize, int[] strides, string padding, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedAvgPool", name) { args = new object[] { input, min_input, max_input }, attrs = new Dictionary<string, object>() { ["ksize"] = ksize, ["strides"] = strides, ["padding"] = padding } });
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
                return quantized_avg_pool_eager_fallback(input, min_input, max_input, ksize: ksize, strides: strides, padding: padding, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["min_input"] = min_input;
        keywords["max_input"] = max_input;
        keywords["ksize"] = ksize;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedAvgPool", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "ksize", _op.get_attr("ksize"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding") };
            _execute.record_gradient("QuantizedAvgPool", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_avg_pool_eager_fallback(Tensor input, Tensor min_input, Tensor max_input, int[] ksize, int[] strides, string padding, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, min_input, max_input };
        object[] _attrs = new object[] { "T", input.dtype, "ksize", ksize, "strides", strides, "padding", padding };
        var _result = _execute.execute("QuantizedAvgPool", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedAvgPool", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Quantized Batch normalization.
    /// </summary>
    /// <remarks>
    /// 
    /// This op is deprecated and will be removed in the future. Prefer
    /// `tf.nn.batch_normalization`.
    /// 
    /// </remarks>
    /// <param name="t"></param>
    /// <param name="t_min"></param>
    /// <param name="t_max"></param>
    /// <param name="m"></param>
    /// <param name="m_min"></param>
    /// <param name="m_max"></param>
    /// <param name="v"></param>
    /// <param name="v_min"></param>
    /// <param name="v_max"></param>
    /// <param name="beta"></param>
    /// <param name="beta_min"></param>
    /// <param name="beta_max"></param>
    /// <param name="gamma"></param>
    /// <param name="gamma_min"></param>
    /// <param name="gamma_max"></param>
    /// <param name="out_type"></param>
    /// <param name="variance_epsilon">
    /// 
    /// A small float number to avoid dividing by 0.
    /// 
    /// </param>
    /// <param name="scale_after_normalization">
    /// 
    /// A bool indicating whether the resulted tensor
    /// needs to be multiplied with gamma.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] quantized_batch_norm_with_global_normalization(Tensor t, Tensor t_min, Tensor t_max, Tensor m, Tensor m_min, Tensor m_max, Tensor v, Tensor v_min, Tensor v_max, Tensor beta, Tensor beta_min, Tensor beta_max, Tensor gamma, Tensor gamma_min, Tensor gamma_max, TF_DataType out_type, float variance_epsilon, bool scale_after_normalization, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedBatchNormWithGlobalNormalization", name) { args = new object[] { t, t_min, t_max, m, m_min, m_max, v, v_min, v_max, beta, beta_min, beta_max, gamma, gamma_min, gamma_max }, attrs = new Dictionary<string, object>() { ["out_type"] = out_type, ["variance_epsilon"] = variance_epsilon, ["scale_after_normalization"] = scale_after_normalization } });
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
                return quantized_batch_norm_with_global_normalization_eager_fallback(t, t_min, t_max, m, m_min, m_max, v, v_min, v_max, beta, beta_min, beta_max, gamma, gamma_min, gamma_max, out_type: out_type, variance_epsilon: variance_epsilon, scale_after_normalization: scale_after_normalization, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["t"] = t;
        keywords["t_min"] = t_min;
        keywords["t_max"] = t_max;
        keywords["m"] = m;
        keywords["m_min"] = m_min;
        keywords["m_max"] = m_max;
        keywords["v"] = v;
        keywords["v_min"] = v_min;
        keywords["v_max"] = v_max;
        keywords["beta"] = beta;
        keywords["beta_min"] = beta_min;
        keywords["beta_max"] = beta_max;
        keywords["gamma"] = gamma;
        keywords["gamma_min"] = gamma_min;
        keywords["gamma_max"] = gamma_max;
        keywords["out_type"] = out_type;
        keywords["variance_epsilon"] = variance_epsilon;
        keywords["scale_after_normalization"] = scale_after_normalization;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedBatchNormWithGlobalNormalization", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tinput", _op._get_attr_type("Tinput"), "out_type", _op._get_attr_type("out_type"), "variance_epsilon", _op.get_attr("variance_epsilon"), "scale_after_normalization", _op._get_attr_bool("scale_after_normalization") };
            _execute.record_gradient("QuantizedBatchNormWithGlobalNormalization", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_batch_norm_with_global_normalization_eager_fallback(Tensor t, Tensor t_min, Tensor t_max, Tensor m, Tensor m_min, Tensor m_max, Tensor v, Tensor v_min, Tensor v_max, Tensor beta, Tensor beta_min, Tensor beta_max, Tensor gamma, Tensor gamma_min, Tensor gamma_max, TF_DataType out_type, float variance_epsilon, bool scale_after_normalization, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { t, t_min, t_max, m, m_min, m_max, v, v_min, v_max, beta, beta_min, beta_max, gamma, gamma_min, gamma_max };
        object[] _attrs = new object[] { "Tinput", t.dtype, "out_type", out_type, "variance_epsilon", variance_epsilon, "scale_after_normalization", scale_after_normalization };
        var _result = _execute.execute("QuantizedBatchNormWithGlobalNormalization", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedBatchNormWithGlobalNormalization", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Adds Tensor 'bias' to Tensor 'input' for Quantized types.
    /// </summary>
    /// <remarks>
    /// 
    /// Broadcasts the values of bias on dimensions 0..N-2 of 'input'.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="bias"></param>
    /// <param name="min_input"></param>
    /// <param name="max_input"></param>
    /// <param name="min_bias"></param>
    /// <param name="max_bias"></param>
    /// <param name="out_type"></param>
    /// <returns></returns>
    public static Tensor[] quantized_bias_add(Tensor input, Tensor bias, Tensor min_input, Tensor max_input, Tensor min_bias, Tensor max_bias, TF_DataType out_type, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedBiasAdd", name) { args = new object[] { input, bias, min_input, max_input, min_bias, max_bias }, attrs = new Dictionary<string, object>() { ["out_type"] = out_type } });
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
                return quantized_bias_add_eager_fallback(input, bias, min_input, max_input, min_bias, max_bias, out_type: out_type, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["bias"] = bias;
        keywords["min_input"] = min_input;
        keywords["max_input"] = max_input;
        keywords["min_bias"] = min_bias;
        keywords["max_bias"] = max_bias;
        keywords["out_type"] = out_type;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedBiasAdd", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T1", _op._get_attr_type("T1"), "T2", _op._get_attr_type("T2"), "out_type", _op._get_attr_type("out_type") };
            _execute.record_gradient("QuantizedBiasAdd", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_bias_add_eager_fallback(Tensor input, Tensor bias, Tensor min_input, Tensor max_input, Tensor min_bias, Tensor max_bias, TF_DataType out_type, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, bias, min_input, max_input, min_bias, max_bias };
        object[] _attrs = new object[] { "T1", input.dtype, "T2", bias.dtype, "out_type", out_type };
        var _result = _execute.execute("QuantizedBiasAdd", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedBiasAdd", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Computes a 2D convolution given quantized 4D input and filter tensors.
    /// </summary>
    /// <remarks>
    /// 
    /// The inputs are quantized tensors where the lowest value represents the real
    /// number of the associated minimum, and the highest represents the maximum.
    /// This means that you can only interpret the quantized output in the same way, by
    /// taking the returned minimum and maximum values into account.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="filter"></param>
    /// <param name="min_input"></param>
    /// <param name="max_input"></param>
    /// <param name="min_filter"></param>
    /// <param name="max_filter"></param>
    /// <param name="out_type"></param>
    /// <param name="strides">
    /// 
    /// The stride of the sliding window for each dimension of the input
    /// tensor.
    /// 
    /// </param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <param name="dilations">
    /// 
    /// 1-D tensor of length 4.  The dilation factor for each dimension of
    /// `input`. If set to k > 1, there will be k-1 skipped cells between each
    /// filter element on that dimension. The dimension order is determined by the
    /// value of `data_format`, see above for details. Dilations in the batch and
    /// depth dimensions must be 1.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] quantized_conv2d(Tensor input, Tensor filter, Tensor min_input, Tensor max_input, Tensor min_filter, Tensor max_filter, int[] strides, string padding, TF_DataType out_type = TF_DataType.TF_QINT32, int[] dilations = null, string? name = null)
    {
        var _ctx = tf.Context;
        if (dilations is null)
        {
            dilations = new int[] { 1, 1, 1, 1 };
        }
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedConv2D", name) { args = new object[] { input, filter, min_input, max_input, min_filter, max_filter }, attrs = new Dictionary<string, object>() { ["out_type"] = out_type, ["strides"] = strides, ["padding"] = padding, ["dilations"] = dilations } });
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
                return quantized_conv2d_eager_fallback(input, filter, min_input, max_input, min_filter, max_filter, out_type: out_type, strides: strides, padding: padding, dilations: dilations, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["filter"] = filter;
        keywords["min_input"] = min_input;
        keywords["max_input"] = max_input;
        keywords["min_filter"] = min_filter;
        keywords["max_filter"] = max_filter;
        keywords["out_type"] = out_type;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["dilations"] = dilations;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedConv2D", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tinput", _op._get_attr_type("Tinput"), "Tfilter", _op._get_attr_type("Tfilter"), "out_type", _op._get_attr_type("out_type"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "dilations", _op.get_attr("dilations") };
            _execute.record_gradient("QuantizedConv2D", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_conv2d_eager_fallback(Tensor input, Tensor filter, Tensor min_input, Tensor max_input, Tensor min_filter, Tensor max_filter, TF_DataType out_type, int[] strides, string padding, int[] dilations, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, filter, min_input, max_input, min_filter, max_filter };
        object[] _attrs = new object[] { "Tinput", input.dtype, "Tfilter", filter.dtype, "out_type", out_type, "strides", strides, "padding", padding, "dilations", dilations };
        var _result = _execute.execute("QuantizedConv2D", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedConv2D", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="input"></param>
    /// <param name="filter"></param>
    /// <param name="min_input"></param>
    /// <param name="max_input"></param>
    /// <param name="min_filter"></param>
    /// <param name="max_filter"></param>
    /// <param name="out_type"></param>
    /// <param name="strides"></param>
    /// <param name="padding"></param>
    /// <param name="dilations"></param>
    /// <param name="padding_list"></param>
    /// <returns></returns>
    public static Tensor[] quantized_conv2d_and_relu(Tensor input, Tensor filter, Tensor min_input, Tensor max_input, Tensor min_filter, Tensor max_filter, int[] strides, string padding, TF_DataType out_type = TF_DataType.TF_QINT32, int[] dilations = null, int[] padding_list = null, string? name = null)
    {
        var _ctx = tf.Context;
        if (dilations is null)
        {
            dilations = new int[] { 1, 1, 1, 1 };
        }
        if (padding_list is null)
        {
            padding_list = new int[] { };
        }
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedConv2DAndRelu", name) { args = new object[] { input, filter, min_input, max_input, min_filter, max_filter }, attrs = new Dictionary<string, object>() { ["out_type"] = out_type, ["strides"] = strides, ["padding"] = padding, ["dilations"] = dilations, ["padding_list"] = padding_list } });
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
                return quantized_conv2d_and_relu_eager_fallback(input, filter, min_input, max_input, min_filter, max_filter, out_type: out_type, strides: strides, padding: padding, dilations: dilations, padding_list: padding_list, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["filter"] = filter;
        keywords["min_input"] = min_input;
        keywords["max_input"] = max_input;
        keywords["min_filter"] = min_filter;
        keywords["max_filter"] = max_filter;
        keywords["out_type"] = out_type;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["dilations"] = dilations;
        keywords["padding_list"] = padding_list;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedConv2DAndRelu", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tinput", _op._get_attr_type("Tinput"), "Tfilter", _op._get_attr_type("Tfilter"), "out_type", _op._get_attr_type("out_type"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "dilations", _op.get_attr("dilations"), "padding_list", _op.get_attr("padding_list") };
            _execute.record_gradient("QuantizedConv2DAndRelu", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_conv2d_and_relu_eager_fallback(Tensor input, Tensor filter, Tensor min_input, Tensor max_input, Tensor min_filter, Tensor max_filter, TF_DataType out_type, int[] strides, string padding, int[] dilations, int[] padding_list, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, filter, min_input, max_input, min_filter, max_filter };
        object[] _attrs = new object[] { "Tinput", input.dtype, "Tfilter", filter.dtype, "out_type", out_type, "strides", strides, "padding", padding, "dilations", dilations, "padding_list", padding_list };
        var _result = _execute.execute("QuantizedConv2DAndRelu", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedConv2DAndRelu", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="input"></param>
    /// <param name="filter"></param>
    /// <param name="min_input"></param>
    /// <param name="max_input"></param>
    /// <param name="min_filter"></param>
    /// <param name="max_filter"></param>
    /// <param name="min_freezed_output"></param>
    /// <param name="max_freezed_output"></param>
    /// <param name="out_type"></param>
    /// <param name="strides"></param>
    /// <param name="padding"></param>
    /// <param name="dilations"></param>
    /// <param name="padding_list"></param>
    /// <returns></returns>
    public static Tensor[] quantized_conv2d_and_relu_and_requantize(Tensor input, Tensor filter, Tensor min_input, Tensor max_input, Tensor min_filter, Tensor max_filter, Tensor min_freezed_output, Tensor max_freezed_output, int[] strides, string padding, TF_DataType out_type = TF_DataType.TF_QUINT8, int[] dilations = null, int[] padding_list = null, string? name = null)
    {
        var _ctx = tf.Context;
        if (dilations is null)
        {
            dilations = new int[] { 1, 1, 1, 1 };
        }
        if (padding_list is null)
        {
            padding_list = new int[] { };
        }
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedConv2DAndReluAndRequantize", name) { args = new object[] { input, filter, min_input, max_input, min_filter, max_filter, min_freezed_output, max_freezed_output }, attrs = new Dictionary<string, object>() { ["out_type"] = out_type, ["strides"] = strides, ["padding"] = padding, ["dilations"] = dilations, ["padding_list"] = padding_list } });
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
                return quantized_conv2d_and_relu_and_requantize_eager_fallback(input, filter, min_input, max_input, min_filter, max_filter, min_freezed_output, max_freezed_output, out_type: out_type, strides: strides, padding: padding, dilations: dilations, padding_list: padding_list, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["filter"] = filter;
        keywords["min_input"] = min_input;
        keywords["max_input"] = max_input;
        keywords["min_filter"] = min_filter;
        keywords["max_filter"] = max_filter;
        keywords["min_freezed_output"] = min_freezed_output;
        keywords["max_freezed_output"] = max_freezed_output;
        keywords["out_type"] = out_type;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["dilations"] = dilations;
        keywords["padding_list"] = padding_list;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedConv2DAndReluAndRequantize", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tinput", _op._get_attr_type("Tinput"), "Tfilter", _op._get_attr_type("Tfilter"), "out_type", _op._get_attr_type("out_type"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "dilations", _op.get_attr("dilations"), "padding_list", _op.get_attr("padding_list") };
            _execute.record_gradient("QuantizedConv2DAndReluAndRequantize", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_conv2d_and_relu_and_requantize_eager_fallback(Tensor input, Tensor filter, Tensor min_input, Tensor max_input, Tensor min_filter, Tensor max_filter, Tensor min_freezed_output, Tensor max_freezed_output, TF_DataType out_type, int[] strides, string padding, int[] dilations, int[] padding_list, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, filter, min_input, max_input, min_filter, max_filter, min_freezed_output, max_freezed_output };
        object[] _attrs = new object[] { "Tinput", input.dtype, "Tfilter", filter.dtype, "out_type", out_type, "strides", strides, "padding", padding, "dilations", dilations, "padding_list", padding_list };
        var _result = _execute.execute("QuantizedConv2DAndReluAndRequantize", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedConv2DAndReluAndRequantize", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="input"></param>
    /// <param name="filter"></param>
    /// <param name="min_input"></param>
    /// <param name="max_input"></param>
    /// <param name="min_filter"></param>
    /// <param name="max_filter"></param>
    /// <param name="min_freezed_output"></param>
    /// <param name="max_freezed_output"></param>
    /// <param name="out_type"></param>
    /// <param name="strides"></param>
    /// <param name="padding"></param>
    /// <param name="dilations"></param>
    /// <param name="padding_list"></param>
    /// <returns></returns>
    public static Tensor[] quantized_conv2d_and_requantize(Tensor input, Tensor filter, Tensor min_input, Tensor max_input, Tensor min_filter, Tensor max_filter, Tensor min_freezed_output, Tensor max_freezed_output, int[] strides, string padding, TF_DataType out_type = TF_DataType.TF_QINT8, int[] dilations = null, int[] padding_list = null, string? name = null)
    {
        var _ctx = tf.Context;
        if (dilations is null)
        {
            dilations = new int[] { 1, 1, 1, 1 };
        }
        if (padding_list is null)
        {
            padding_list = new int[] { };
        }
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedConv2DAndRequantize", name) { args = new object[] { input, filter, min_input, max_input, min_filter, max_filter, min_freezed_output, max_freezed_output }, attrs = new Dictionary<string, object>() { ["out_type"] = out_type, ["strides"] = strides, ["padding"] = padding, ["dilations"] = dilations, ["padding_list"] = padding_list } });
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
                return quantized_conv2d_and_requantize_eager_fallback(input, filter, min_input, max_input, min_filter, max_filter, min_freezed_output, max_freezed_output, out_type: out_type, strides: strides, padding: padding, dilations: dilations, padding_list: padding_list, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["filter"] = filter;
        keywords["min_input"] = min_input;
        keywords["max_input"] = max_input;
        keywords["min_filter"] = min_filter;
        keywords["max_filter"] = max_filter;
        keywords["min_freezed_output"] = min_freezed_output;
        keywords["max_freezed_output"] = max_freezed_output;
        keywords["out_type"] = out_type;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["dilations"] = dilations;
        keywords["padding_list"] = padding_list;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedConv2DAndRequantize", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tinput", _op._get_attr_type("Tinput"), "Tfilter", _op._get_attr_type("Tfilter"), "out_type", _op._get_attr_type("out_type"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "dilations", _op.get_attr("dilations"), "padding_list", _op.get_attr("padding_list") };
            _execute.record_gradient("QuantizedConv2DAndRequantize", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_conv2d_and_requantize_eager_fallback(Tensor input, Tensor filter, Tensor min_input, Tensor max_input, Tensor min_filter, Tensor max_filter, Tensor min_freezed_output, Tensor max_freezed_output, TF_DataType out_type, int[] strides, string padding, int[] dilations, int[] padding_list, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, filter, min_input, max_input, min_filter, max_filter, min_freezed_output, max_freezed_output };
        object[] _attrs = new object[] { "Tinput", input.dtype, "Tfilter", filter.dtype, "out_type", out_type, "strides", strides, "padding", padding, "dilations", dilations, "padding_list", padding_list };
        var _result = _execute.execute("QuantizedConv2DAndRequantize", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedConv2DAndRequantize", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Computes QuantizedConv2D per channel.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="filter"></param>
    /// <param name="min_input"></param>
    /// <param name="max_input"></param>
    /// <param name="min_filter"></param>
    /// <param name="max_filter"></param>
    /// <param name="out_type">
    /// 
    /// The quantized type of output tensor that needs to be converted.
    /// 
    /// </param>
    /// <param name="strides">
    /// list of stride values.
    /// </param>
    /// <param name="padding"></param>
    /// <param name="dilations">
    /// list of dilation values.
    /// </param>
    /// <returns></returns>
    public static Tensor[] quantized_conv2d_per_channel(Tensor input, Tensor filter, Tensor min_input, Tensor max_input, Tensor min_filter, Tensor max_filter, int[] strides, string padding, TF_DataType out_type = TF_DataType.TF_QINT32, int[] dilations = null, string? name = null)
    {
        var _ctx = tf.Context;
        if (dilations is null)
        {
            dilations = new int[] { 1, 1, 1, 1 };
        }
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedConv2DPerChannel", name) { args = new object[] { input, filter, min_input, max_input, min_filter, max_filter }, attrs = new Dictionary<string, object>() { ["out_type"] = out_type, ["strides"] = strides, ["padding"] = padding, ["dilations"] = dilations } });
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
                return quantized_conv2d_per_channel_eager_fallback(input, filter, min_input, max_input, min_filter, max_filter, out_type: out_type, strides: strides, padding: padding, dilations: dilations, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["filter"] = filter;
        keywords["min_input"] = min_input;
        keywords["max_input"] = max_input;
        keywords["min_filter"] = min_filter;
        keywords["max_filter"] = max_filter;
        keywords["out_type"] = out_type;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["dilations"] = dilations;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedConv2DPerChannel", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tinput", _op._get_attr_type("Tinput"), "Tfilter", _op._get_attr_type("Tfilter"), "out_type", _op._get_attr_type("out_type"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "dilations", _op.get_attr("dilations") };
            _execute.record_gradient("QuantizedConv2DPerChannel", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_conv2d_per_channel_eager_fallback(Tensor input, Tensor filter, Tensor min_input, Tensor max_input, Tensor min_filter, Tensor max_filter, TF_DataType out_type, int[] strides, string padding, int[] dilations, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, filter, min_input, max_input, min_filter, max_filter };
        object[] _attrs = new object[] { "Tinput", input.dtype, "Tfilter", filter.dtype, "out_type", out_type, "strides", strides, "padding", padding, "dilations", dilations };
        var _result = _execute.execute("QuantizedConv2DPerChannel", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedConv2DPerChannel", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="input"></param>
    /// <param name="filter"></param>
    /// <param name="bias"></param>
    /// <param name="min_input"></param>
    /// <param name="max_input"></param>
    /// <param name="min_filter"></param>
    /// <param name="max_filter"></param>
    /// <param name="out_type"></param>
    /// <param name="strides"></param>
    /// <param name="padding"></param>
    /// <param name="dilations"></param>
    /// <param name="padding_list"></param>
    /// <returns></returns>
    public static Tensor[] quantized_conv2d_with_bias(Tensor input, Tensor filter, Tensor bias, Tensor min_input, Tensor max_input, Tensor min_filter, Tensor max_filter, int[] strides, string padding, TF_DataType out_type = TF_DataType.TF_QINT32, int[] dilations = null, int[] padding_list = null, string? name = null)
    {
        var _ctx = tf.Context;
        if (dilations is null)
        {
            dilations = new int[] { 1, 1, 1, 1 };
        }
        if (padding_list is null)
        {
            padding_list = new int[] { };
        }
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedConv2DWithBias", name) { args = new object[] { input, filter, bias, min_input, max_input, min_filter, max_filter }, attrs = new Dictionary<string, object>() { ["out_type"] = out_type, ["strides"] = strides, ["padding"] = padding, ["dilations"] = dilations, ["padding_list"] = padding_list } });
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
                return quantized_conv2d_with_bias_eager_fallback(input, filter, bias, min_input, max_input, min_filter, max_filter, out_type: out_type, strides: strides, padding: padding, dilations: dilations, padding_list: padding_list, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["filter"] = filter;
        keywords["bias"] = bias;
        keywords["min_input"] = min_input;
        keywords["max_input"] = max_input;
        keywords["min_filter"] = min_filter;
        keywords["max_filter"] = max_filter;
        keywords["out_type"] = out_type;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["dilations"] = dilations;
        keywords["padding_list"] = padding_list;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedConv2DWithBias", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tinput", _op._get_attr_type("Tinput"), "Tfilter", _op._get_attr_type("Tfilter"), "out_type", _op._get_attr_type("out_type"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "dilations", _op.get_attr("dilations"), "padding_list", _op.get_attr("padding_list") };
            _execute.record_gradient("QuantizedConv2DWithBias", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_conv2d_with_bias_eager_fallback(Tensor input, Tensor filter, Tensor bias, Tensor min_input, Tensor max_input, Tensor min_filter, Tensor max_filter, TF_DataType out_type, int[] strides, string padding, int[] dilations, int[] padding_list, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, filter, bias, min_input, max_input, min_filter, max_filter };
        object[] _attrs = new object[] { "Tinput", input.dtype, "Tfilter", filter.dtype, "out_type", out_type, "strides", strides, "padding", padding, "dilations", dilations, "padding_list", padding_list };
        var _result = _execute.execute("QuantizedConv2DWithBias", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedConv2DWithBias", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="input"></param>
    /// <param name="filter"></param>
    /// <param name="bias"></param>
    /// <param name="min_input"></param>
    /// <param name="max_input"></param>
    /// <param name="min_filter"></param>
    /// <param name="max_filter"></param>
    /// <param name="out_type"></param>
    /// <param name="strides"></param>
    /// <param name="padding"></param>
    /// <param name="dilations"></param>
    /// <param name="padding_list"></param>
    /// <returns></returns>
    public static Tensor[] quantized_conv2d_with_bias_and_relu(Tensor input, Tensor filter, Tensor bias, Tensor min_input, Tensor max_input, Tensor min_filter, Tensor max_filter, int[] strides, string padding, TF_DataType out_type = TF_DataType.TF_QINT32, int[] dilations = null, int[] padding_list = null, string? name = null)
    {
        var _ctx = tf.Context;
        if (dilations is null)
        {
            dilations = new int[] { 1, 1, 1, 1 };
        }
        if (padding_list is null)
        {
            padding_list = new int[] { };
        }
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedConv2DWithBiasAndRelu", name) { args = new object[] { input, filter, bias, min_input, max_input, min_filter, max_filter }, attrs = new Dictionary<string, object>() { ["out_type"] = out_type, ["strides"] = strides, ["padding"] = padding, ["dilations"] = dilations, ["padding_list"] = padding_list } });
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
                return quantized_conv2d_with_bias_and_relu_eager_fallback(input, filter, bias, min_input, max_input, min_filter, max_filter, out_type: out_type, strides: strides, padding: padding, dilations: dilations, padding_list: padding_list, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["filter"] = filter;
        keywords["bias"] = bias;
        keywords["min_input"] = min_input;
        keywords["max_input"] = max_input;
        keywords["min_filter"] = min_filter;
        keywords["max_filter"] = max_filter;
        keywords["out_type"] = out_type;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["dilations"] = dilations;
        keywords["padding_list"] = padding_list;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedConv2DWithBiasAndRelu", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tinput", _op._get_attr_type("Tinput"), "Tfilter", _op._get_attr_type("Tfilter"), "out_type", _op._get_attr_type("out_type"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "dilations", _op.get_attr("dilations"), "padding_list", _op.get_attr("padding_list") };
            _execute.record_gradient("QuantizedConv2DWithBiasAndRelu", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_conv2d_with_bias_and_relu_eager_fallback(Tensor input, Tensor filter, Tensor bias, Tensor min_input, Tensor max_input, Tensor min_filter, Tensor max_filter, TF_DataType out_type, int[] strides, string padding, int[] dilations, int[] padding_list, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, filter, bias, min_input, max_input, min_filter, max_filter };
        object[] _attrs = new object[] { "Tinput", input.dtype, "Tfilter", filter.dtype, "out_type", out_type, "strides", strides, "padding", padding, "dilations", dilations, "padding_list", padding_list };
        var _result = _execute.execute("QuantizedConv2DWithBiasAndRelu", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedConv2DWithBiasAndRelu", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="input"></param>
    /// <param name="filter"></param>
    /// <param name="bias"></param>
    /// <param name="min_input"></param>
    /// <param name="max_input"></param>
    /// <param name="min_filter"></param>
    /// <param name="max_filter"></param>
    /// <param name="min_freezed_output"></param>
    /// <param name="max_freezed_output"></param>
    /// <param name="out_type"></param>
    /// <param name="strides"></param>
    /// <param name="padding"></param>
    /// <param name="dilations"></param>
    /// <param name="padding_list"></param>
    /// <returns></returns>
    public static Tensor[] quantized_conv2d_with_bias_and_relu_and_requantize(Tensor input, Tensor filter, Tensor bias, Tensor min_input, Tensor max_input, Tensor min_filter, Tensor max_filter, Tensor min_freezed_output, Tensor max_freezed_output, int[] strides, string padding, TF_DataType out_type = TF_DataType.TF_QUINT8, int[] dilations = null, int[] padding_list = null, string? name = null)
    {
        var _ctx = tf.Context;
        if (dilations is null)
        {
            dilations = new int[] { 1, 1, 1, 1 };
        }
        if (padding_list is null)
        {
            padding_list = new int[] { };
        }
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedConv2DWithBiasAndReluAndRequantize", name) { args = new object[] { input, filter, bias, min_input, max_input, min_filter, max_filter, min_freezed_output, max_freezed_output }, attrs = new Dictionary<string, object>() { ["out_type"] = out_type, ["strides"] = strides, ["padding"] = padding, ["dilations"] = dilations, ["padding_list"] = padding_list } });
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
                return quantized_conv2d_with_bias_and_relu_and_requantize_eager_fallback(input, filter, bias, min_input, max_input, min_filter, max_filter, min_freezed_output, max_freezed_output, out_type: out_type, strides: strides, padding: padding, dilations: dilations, padding_list: padding_list, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["filter"] = filter;
        keywords["bias"] = bias;
        keywords["min_input"] = min_input;
        keywords["max_input"] = max_input;
        keywords["min_filter"] = min_filter;
        keywords["max_filter"] = max_filter;
        keywords["min_freezed_output"] = min_freezed_output;
        keywords["max_freezed_output"] = max_freezed_output;
        keywords["out_type"] = out_type;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["dilations"] = dilations;
        keywords["padding_list"] = padding_list;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedConv2DWithBiasAndReluAndRequantize", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tinput", _op._get_attr_type("Tinput"), "Tfilter", _op._get_attr_type("Tfilter"), "Tbias", _op._get_attr_type("Tbias"), "out_type", _op._get_attr_type("out_type"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "dilations", _op.get_attr("dilations"), "padding_list", _op.get_attr("padding_list") };
            _execute.record_gradient("QuantizedConv2DWithBiasAndReluAndRequantize", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_conv2d_with_bias_and_relu_and_requantize_eager_fallback(Tensor input, Tensor filter, Tensor bias, Tensor min_input, Tensor max_input, Tensor min_filter, Tensor max_filter, Tensor min_freezed_output, Tensor max_freezed_output, TF_DataType out_type, int[] strides, string padding, int[] dilations, int[] padding_list, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, filter, bias, min_input, max_input, min_filter, max_filter, min_freezed_output, max_freezed_output };
        object[] _attrs = new object[] { "Tinput", input.dtype, "Tfilter", filter.dtype, "Tbias", bias.dtype, "out_type", out_type, "strides", strides, "padding", padding, "dilations", dilations, "padding_list", padding_list };
        var _result = _execute.execute("QuantizedConv2DWithBiasAndReluAndRequantize", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedConv2DWithBiasAndReluAndRequantize", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="input"></param>
    /// <param name="filter"></param>
    /// <param name="bias"></param>
    /// <param name="min_input"></param>
    /// <param name="max_input"></param>
    /// <param name="min_filter"></param>
    /// <param name="max_filter"></param>
    /// <param name="min_freezed_output"></param>
    /// <param name="max_freezed_output"></param>
    /// <param name="out_type"></param>
    /// <param name="strides"></param>
    /// <param name="padding"></param>
    /// <param name="dilations"></param>
    /// <param name="padding_list"></param>
    /// <returns></returns>
    public static Tensor[] quantized_conv2d_with_bias_and_requantize(Tensor input, Tensor filter, Tensor bias, Tensor min_input, Tensor max_input, Tensor min_filter, Tensor max_filter, Tensor min_freezed_output, Tensor max_freezed_output, int[] strides, string padding, TF_DataType out_type = TF_DataType.TF_QINT8, int[] dilations = null, int[] padding_list = null, string? name = null)
    {
        var _ctx = tf.Context;
        if (dilations is null)
        {
            dilations = new int[] { 1, 1, 1, 1 };
        }
        if (padding_list is null)
        {
            padding_list = new int[] { };
        }
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedConv2DWithBiasAndRequantize", name) { args = new object[] { input, filter, bias, min_input, max_input, min_filter, max_filter, min_freezed_output, max_freezed_output }, attrs = new Dictionary<string, object>() { ["out_type"] = out_type, ["strides"] = strides, ["padding"] = padding, ["dilations"] = dilations, ["padding_list"] = padding_list } });
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
                return quantized_conv2d_with_bias_and_requantize_eager_fallback(input, filter, bias, min_input, max_input, min_filter, max_filter, min_freezed_output, max_freezed_output, out_type: out_type, strides: strides, padding: padding, dilations: dilations, padding_list: padding_list, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["filter"] = filter;
        keywords["bias"] = bias;
        keywords["min_input"] = min_input;
        keywords["max_input"] = max_input;
        keywords["min_filter"] = min_filter;
        keywords["max_filter"] = max_filter;
        keywords["min_freezed_output"] = min_freezed_output;
        keywords["max_freezed_output"] = max_freezed_output;
        keywords["out_type"] = out_type;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["dilations"] = dilations;
        keywords["padding_list"] = padding_list;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedConv2DWithBiasAndRequantize", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tinput", _op._get_attr_type("Tinput"), "Tfilter", _op._get_attr_type("Tfilter"), "Tbias", _op._get_attr_type("Tbias"), "out_type", _op._get_attr_type("out_type"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "dilations", _op.get_attr("dilations"), "padding_list", _op.get_attr("padding_list") };
            _execute.record_gradient("QuantizedConv2DWithBiasAndRequantize", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_conv2d_with_bias_and_requantize_eager_fallback(Tensor input, Tensor filter, Tensor bias, Tensor min_input, Tensor max_input, Tensor min_filter, Tensor max_filter, Tensor min_freezed_output, Tensor max_freezed_output, TF_DataType out_type, int[] strides, string padding, int[] dilations, int[] padding_list, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, filter, bias, min_input, max_input, min_filter, max_filter, min_freezed_output, max_freezed_output };
        object[] _attrs = new object[] { "Tinput", input.dtype, "Tfilter", filter.dtype, "Tbias", bias.dtype, "out_type", out_type, "strides", strides, "padding", padding, "dilations", dilations, "padding_list", padding_list };
        var _result = _execute.execute("QuantizedConv2DWithBiasAndRequantize", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedConv2DWithBiasAndRequantize", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="input"></param>
    /// <param name="filter"></param>
    /// <param name="bias"></param>
    /// <param name="min_input"></param>
    /// <param name="max_input"></param>
    /// <param name="min_filter"></param>
    /// <param name="max_filter"></param>
    /// <param name="min_freezed_output"></param>
    /// <param name="max_freezed_output"></param>
    /// <param name="summand"></param>
    /// <param name="min_summand"></param>
    /// <param name="max_summand"></param>
    /// <param name="out_type"></param>
    /// <param name="strides"></param>
    /// <param name="padding"></param>
    /// <param name="dilations"></param>
    /// <param name="padding_list"></param>
    /// <returns></returns>
    public static Tensor[] quantized_conv2d_with_bias_signed_sum_and_relu_and_requantize(Tensor input, Tensor filter, Tensor bias, Tensor min_input, Tensor max_input, Tensor min_filter, Tensor max_filter, Tensor min_freezed_output, Tensor max_freezed_output, Tensor summand, Tensor min_summand, Tensor max_summand, int[] strides, string padding, TF_DataType out_type = TF_DataType.TF_QUINT8, int[] dilations = null, int[] padding_list = null, string? name = null)
    {
        var _ctx = tf.Context;
        if (dilations is null)
        {
            dilations = new int[] { 1, 1, 1, 1 };
        }
        if (padding_list is null)
        {
            padding_list = new int[] { };
        }
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedConv2DWithBiasSignedSumAndReluAndRequantize", name) { args = new object[] { input, filter, bias, min_input, max_input, min_filter, max_filter, min_freezed_output, max_freezed_output, summand, min_summand, max_summand }, attrs = new Dictionary<string, object>() { ["out_type"] = out_type, ["strides"] = strides, ["padding"] = padding, ["dilations"] = dilations, ["padding_list"] = padding_list } });
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
                return quantized_conv2d_with_bias_signed_sum_and_relu_and_requantize_eager_fallback(input, filter, bias, min_input, max_input, min_filter, max_filter, min_freezed_output, max_freezed_output, summand, min_summand, max_summand, out_type: out_type, strides: strides, padding: padding, dilations: dilations, padding_list: padding_list, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["filter"] = filter;
        keywords["bias"] = bias;
        keywords["min_input"] = min_input;
        keywords["max_input"] = max_input;
        keywords["min_filter"] = min_filter;
        keywords["max_filter"] = max_filter;
        keywords["min_freezed_output"] = min_freezed_output;
        keywords["max_freezed_output"] = max_freezed_output;
        keywords["summand"] = summand;
        keywords["min_summand"] = min_summand;
        keywords["max_summand"] = max_summand;
        keywords["out_type"] = out_type;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["dilations"] = dilations;
        keywords["padding_list"] = padding_list;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedConv2DWithBiasSignedSumAndReluAndRequantize", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tinput", _op._get_attr_type("Tinput"), "Tfilter", _op._get_attr_type("Tfilter"), "Tbias", _op._get_attr_type("Tbias"), "Tsummand", _op._get_attr_type("Tsummand"), "out_type", _op._get_attr_type("out_type"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "dilations", _op.get_attr("dilations"), "padding_list", _op.get_attr("padding_list") };
            _execute.record_gradient("QuantizedConv2DWithBiasSignedSumAndReluAndRequantize", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_conv2d_with_bias_signed_sum_and_relu_and_requantize_eager_fallback(Tensor input, Tensor filter, Tensor bias, Tensor min_input, Tensor max_input, Tensor min_filter, Tensor max_filter, Tensor min_freezed_output, Tensor max_freezed_output, Tensor summand, Tensor min_summand, Tensor max_summand, TF_DataType out_type, int[] strides, string padding, int[] dilations, int[] padding_list, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, filter, bias, min_input, max_input, min_filter, max_filter, min_freezed_output, max_freezed_output, summand, min_summand, max_summand };
        object[] _attrs = new object[] { "Tinput", input.dtype, "Tfilter", filter.dtype, "Tbias", bias.dtype, "Tsummand", summand.dtype, "out_type", out_type, "strides", strides, "padding", padding, "dilations", dilations, "padding_list", padding_list };
        var _result = _execute.execute("QuantizedConv2DWithBiasSignedSumAndReluAndRequantize", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedConv2DWithBiasSignedSumAndReluAndRequantize", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="input"></param>
    /// <param name="filter"></param>
    /// <param name="bias"></param>
    /// <param name="min_input"></param>
    /// <param name="max_input"></param>
    /// <param name="min_filter"></param>
    /// <param name="max_filter"></param>
    /// <param name="summand"></param>
    /// <param name="out_type"></param>
    /// <param name="strides"></param>
    /// <param name="padding"></param>
    /// <param name="dilations"></param>
    /// <param name="padding_list"></param>
    /// <returns></returns>
    public static Tensor[] quantized_conv2d_with_bias_sum_and_relu(Tensor input, Tensor filter, Tensor bias, Tensor min_input, Tensor max_input, Tensor min_filter, Tensor max_filter, Tensor summand, int[] strides, string padding, TF_DataType out_type = TF_DataType.TF_QINT32, int[] dilations = null, int[] padding_list = null, string? name = null)
    {
        var _ctx = tf.Context;
        if (dilations is null)
        {
            dilations = new int[] { 1, 1, 1, 1 };
        }
        if (padding_list is null)
        {
            padding_list = new int[] { };
        }
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedConv2DWithBiasSumAndRelu", name) { args = new object[] { input, filter, bias, min_input, max_input, min_filter, max_filter, summand }, attrs = new Dictionary<string, object>() { ["out_type"] = out_type, ["strides"] = strides, ["padding"] = padding, ["dilations"] = dilations, ["padding_list"] = padding_list } });
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
                return quantized_conv2d_with_bias_sum_and_relu_eager_fallback(input, filter, bias, min_input, max_input, min_filter, max_filter, summand, out_type: out_type, strides: strides, padding: padding, dilations: dilations, padding_list: padding_list, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["filter"] = filter;
        keywords["bias"] = bias;
        keywords["min_input"] = min_input;
        keywords["max_input"] = max_input;
        keywords["min_filter"] = min_filter;
        keywords["max_filter"] = max_filter;
        keywords["summand"] = summand;
        keywords["out_type"] = out_type;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["dilations"] = dilations;
        keywords["padding_list"] = padding_list;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedConv2DWithBiasSumAndRelu", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tinput", _op._get_attr_type("Tinput"), "Tfilter", _op._get_attr_type("Tfilter"), "out_type", _op._get_attr_type("out_type"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "dilations", _op.get_attr("dilations"), "padding_list", _op.get_attr("padding_list") };
            _execute.record_gradient("QuantizedConv2DWithBiasSumAndRelu", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_conv2d_with_bias_sum_and_relu_eager_fallback(Tensor input, Tensor filter, Tensor bias, Tensor min_input, Tensor max_input, Tensor min_filter, Tensor max_filter, Tensor summand, TF_DataType out_type, int[] strides, string padding, int[] dilations, int[] padding_list, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, filter, bias, min_input, max_input, min_filter, max_filter, summand };
        object[] _attrs = new object[] { "Tinput", input.dtype, "Tfilter", filter.dtype, "out_type", out_type, "strides", strides, "padding", padding, "dilations", dilations, "padding_list", padding_list };
        var _result = _execute.execute("QuantizedConv2DWithBiasSumAndRelu", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedConv2DWithBiasSumAndRelu", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="input"></param>
    /// <param name="filter"></param>
    /// <param name="bias"></param>
    /// <param name="min_input"></param>
    /// <param name="max_input"></param>
    /// <param name="min_filter"></param>
    /// <param name="max_filter"></param>
    /// <param name="min_freezed_output"></param>
    /// <param name="max_freezed_output"></param>
    /// <param name="summand"></param>
    /// <param name="min_summand"></param>
    /// <param name="max_summand"></param>
    /// <param name="out_type"></param>
    /// <param name="strides"></param>
    /// <param name="padding"></param>
    /// <param name="dilations"></param>
    /// <param name="padding_list"></param>
    /// <returns></returns>
    public static Tensor[] quantized_conv2d_with_bias_sum_and_relu_and_requantize(Tensor input, Tensor filter, Tensor bias, Tensor min_input, Tensor max_input, Tensor min_filter, Tensor max_filter, Tensor min_freezed_output, Tensor max_freezed_output, Tensor summand, Tensor min_summand, Tensor max_summand, int[] strides, string padding, TF_DataType out_type = TF_DataType.TF_QUINT8, int[] dilations = null, int[] padding_list = null, string? name = null)
    {
        var _ctx = tf.Context;
        if (dilations is null)
        {
            dilations = new int[] { 1, 1, 1, 1 };
        }
        if (padding_list is null)
        {
            padding_list = new int[] { };
        }
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedConv2DWithBiasSumAndReluAndRequantize", name) { args = new object[] { input, filter, bias, min_input, max_input, min_filter, max_filter, min_freezed_output, max_freezed_output, summand, min_summand, max_summand }, attrs = new Dictionary<string, object>() { ["out_type"] = out_type, ["strides"] = strides, ["padding"] = padding, ["dilations"] = dilations, ["padding_list"] = padding_list } });
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
                return quantized_conv2d_with_bias_sum_and_relu_and_requantize_eager_fallback(input, filter, bias, min_input, max_input, min_filter, max_filter, min_freezed_output, max_freezed_output, summand, min_summand, max_summand, out_type: out_type, strides: strides, padding: padding, dilations: dilations, padding_list: padding_list, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["filter"] = filter;
        keywords["bias"] = bias;
        keywords["min_input"] = min_input;
        keywords["max_input"] = max_input;
        keywords["min_filter"] = min_filter;
        keywords["max_filter"] = max_filter;
        keywords["min_freezed_output"] = min_freezed_output;
        keywords["max_freezed_output"] = max_freezed_output;
        keywords["summand"] = summand;
        keywords["min_summand"] = min_summand;
        keywords["max_summand"] = max_summand;
        keywords["out_type"] = out_type;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["dilations"] = dilations;
        keywords["padding_list"] = padding_list;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedConv2DWithBiasSumAndReluAndRequantize", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tinput", _op._get_attr_type("Tinput"), "Tfilter", _op._get_attr_type("Tfilter"), "Tbias", _op._get_attr_type("Tbias"), "Tsummand", _op._get_attr_type("Tsummand"), "out_type", _op._get_attr_type("out_type"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "dilations", _op.get_attr("dilations"), "padding_list", _op.get_attr("padding_list") };
            _execute.record_gradient("QuantizedConv2DWithBiasSumAndReluAndRequantize", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_conv2d_with_bias_sum_and_relu_and_requantize_eager_fallback(Tensor input, Tensor filter, Tensor bias, Tensor min_input, Tensor max_input, Tensor min_filter, Tensor max_filter, Tensor min_freezed_output, Tensor max_freezed_output, Tensor summand, Tensor min_summand, Tensor max_summand, TF_DataType out_type, int[] strides, string padding, int[] dilations, int[] padding_list, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, filter, bias, min_input, max_input, min_filter, max_filter, min_freezed_output, max_freezed_output, summand, min_summand, max_summand };
        object[] _attrs = new object[] { "Tinput", input.dtype, "Tfilter", filter.dtype, "Tbias", bias.dtype, "Tsummand", summand.dtype, "out_type", out_type, "strides", strides, "padding", padding, "dilations", dilations, "padding_list", padding_list };
        var _result = _execute.execute("QuantizedConv2DWithBiasSumAndReluAndRequantize", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedConv2DWithBiasSumAndReluAndRequantize", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Computes quantized depthwise Conv2D.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="filter"></param>
    /// <param name="min_input"></param>
    /// <param name="max_input"></param>
    /// <param name="min_filter"></param>
    /// <param name="max_filter"></param>
    /// <param name="out_type">
    /// The type of the output.
    /// </param>
    /// <param name="strides">
    /// List of stride values.
    /// </param>
    /// <param name="padding"></param>
    /// <param name="dilations">
    /// List of dilation values.
    /// </param>
    /// <returns></returns>
    public static Tensor[] quantized_depthwise_conv2d(Tensor input, Tensor filter, Tensor min_input, Tensor max_input, Tensor min_filter, Tensor max_filter, int[] strides, string padding, TF_DataType out_type = TF_DataType.TF_QINT32, int[] dilations = null, string? name = null)
    {
        var _ctx = tf.Context;
        if (dilations is null)
        {
            dilations = new int[] { 1, 1, 1, 1 };
        }
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedDepthwiseConv2D", name) { args = new object[] { input, filter, min_input, max_input, min_filter, max_filter }, attrs = new Dictionary<string, object>() { ["out_type"] = out_type, ["strides"] = strides, ["padding"] = padding, ["dilations"] = dilations } });
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
                return quantized_depthwise_conv2d_eager_fallback(input, filter, min_input, max_input, min_filter, max_filter, out_type: out_type, strides: strides, padding: padding, dilations: dilations, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["filter"] = filter;
        keywords["min_input"] = min_input;
        keywords["max_input"] = max_input;
        keywords["min_filter"] = min_filter;
        keywords["max_filter"] = max_filter;
        keywords["out_type"] = out_type;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["dilations"] = dilations;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedDepthwiseConv2D", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tinput", _op._get_attr_type("Tinput"), "Tfilter", _op._get_attr_type("Tfilter"), "out_type", _op._get_attr_type("out_type"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "dilations", _op.get_attr("dilations") };
            _execute.record_gradient("QuantizedDepthwiseConv2D", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_depthwise_conv2d_eager_fallback(Tensor input, Tensor filter, Tensor min_input, Tensor max_input, Tensor min_filter, Tensor max_filter, TF_DataType out_type, int[] strides, string padding, int[] dilations, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, filter, min_input, max_input, min_filter, max_filter };
        object[] _attrs = new object[] { "Tinput", input.dtype, "Tfilter", filter.dtype, "out_type", out_type, "strides", strides, "padding", padding, "dilations", dilations };
        var _result = _execute.execute("QuantizedDepthwiseConv2D", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedDepthwiseConv2D", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Computes quantized depthwise Conv2D with Bias.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="filter"></param>
    /// <param name="bias"></param>
    /// <param name="min_input"></param>
    /// <param name="max_input"></param>
    /// <param name="min_filter"></param>
    /// <param name="max_filter"></param>
    /// <param name="out_type">
    /// The type of the output.
    /// </param>
    /// <param name="strides">
    /// List of stride values.
    /// </param>
    /// <param name="padding"></param>
    /// <param name="dilations">
    /// List of dilation values.
    /// </param>
    /// <returns></returns>
    public static Tensor[] quantized_depthwise_conv2d_with_bias(Tensor input, Tensor filter, Tensor bias, Tensor min_input, Tensor max_input, Tensor min_filter, Tensor max_filter, int[] strides, string padding, TF_DataType out_type = TF_DataType.TF_QINT32, int[] dilations = null, string? name = null)
    {
        var _ctx = tf.Context;
        if (dilations is null)
        {
            dilations = new int[] { 1, 1, 1, 1 };
        }
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedDepthwiseConv2DWithBias", name) { args = new object[] { input, filter, bias, min_input, max_input, min_filter, max_filter }, attrs = new Dictionary<string, object>() { ["out_type"] = out_type, ["strides"] = strides, ["padding"] = padding, ["dilations"] = dilations } });
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
                return quantized_depthwise_conv2d_with_bias_eager_fallback(input, filter, bias, min_input, max_input, min_filter, max_filter, out_type: out_type, strides: strides, padding: padding, dilations: dilations, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["filter"] = filter;
        keywords["bias"] = bias;
        keywords["min_input"] = min_input;
        keywords["max_input"] = max_input;
        keywords["min_filter"] = min_filter;
        keywords["max_filter"] = max_filter;
        keywords["out_type"] = out_type;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["dilations"] = dilations;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedDepthwiseConv2DWithBias", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tinput", _op._get_attr_type("Tinput"), "Tfilter", _op._get_attr_type("Tfilter"), "out_type", _op._get_attr_type("out_type"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "dilations", _op.get_attr("dilations") };
            _execute.record_gradient("QuantizedDepthwiseConv2DWithBias", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_depthwise_conv2d_with_bias_eager_fallback(Tensor input, Tensor filter, Tensor bias, Tensor min_input, Tensor max_input, Tensor min_filter, Tensor max_filter, TF_DataType out_type, int[] strides, string padding, int[] dilations, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, filter, bias, min_input, max_input, min_filter, max_filter };
        object[] _attrs = new object[] { "Tinput", input.dtype, "Tfilter", filter.dtype, "out_type", out_type, "strides", strides, "padding", padding, "dilations", dilations };
        var _result = _execute.execute("QuantizedDepthwiseConv2DWithBias", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedDepthwiseConv2DWithBias", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Computes quantized depthwise Conv2D with Bias and Relu.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="filter"></param>
    /// <param name="bias"></param>
    /// <param name="min_input"></param>
    /// <param name="max_input"></param>
    /// <param name="min_filter"></param>
    /// <param name="max_filter"></param>
    /// <param name="out_type">
    /// The type of the output.
    /// </param>
    /// <param name="strides">
    /// List of stride values.
    /// </param>
    /// <param name="padding"></param>
    /// <param name="dilations">
    /// List of dilation values.
    /// </param>
    /// <param name="padding_list"></param>
    /// <returns></returns>
    public static Tensor[] quantized_depthwise_conv2d_with_bias_and_relu(Tensor input, Tensor filter, Tensor bias, Tensor min_input, Tensor max_input, Tensor min_filter, Tensor max_filter, int[] strides, string padding, TF_DataType out_type = TF_DataType.TF_QINT32, int[] dilations = null, int[] padding_list = null, string? name = null)
    {
        var _ctx = tf.Context;
        if (dilations is null)
        {
            dilations = new int[] { 1, 1, 1, 1 };
        }
        if (padding_list is null)
        {
            padding_list = new int[] { };
        }
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedDepthwiseConv2DWithBiasAndRelu", name) { args = new object[] { input, filter, bias, min_input, max_input, min_filter, max_filter }, attrs = new Dictionary<string, object>() { ["out_type"] = out_type, ["strides"] = strides, ["padding"] = padding, ["dilations"] = dilations, ["padding_list"] = padding_list } });
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
                return quantized_depthwise_conv2d_with_bias_and_relu_eager_fallback(input, filter, bias, min_input, max_input, min_filter, max_filter, out_type: out_type, strides: strides, padding: padding, dilations: dilations, padding_list: padding_list, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["filter"] = filter;
        keywords["bias"] = bias;
        keywords["min_input"] = min_input;
        keywords["max_input"] = max_input;
        keywords["min_filter"] = min_filter;
        keywords["max_filter"] = max_filter;
        keywords["out_type"] = out_type;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["dilations"] = dilations;
        keywords["padding_list"] = padding_list;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedDepthwiseConv2DWithBiasAndRelu", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tinput", _op._get_attr_type("Tinput"), "Tfilter", _op._get_attr_type("Tfilter"), "out_type", _op._get_attr_type("out_type"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "dilations", _op.get_attr("dilations"), "padding_list", _op.get_attr("padding_list") };
            _execute.record_gradient("QuantizedDepthwiseConv2DWithBiasAndRelu", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_depthwise_conv2d_with_bias_and_relu_eager_fallback(Tensor input, Tensor filter, Tensor bias, Tensor min_input, Tensor max_input, Tensor min_filter, Tensor max_filter, TF_DataType out_type, int[] strides, string padding, int[] dilations, int[] padding_list, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, filter, bias, min_input, max_input, min_filter, max_filter };
        object[] _attrs = new object[] { "Tinput", input.dtype, "Tfilter", filter.dtype, "out_type", out_type, "strides", strides, "padding", padding, "dilations", dilations, "padding_list", padding_list };
        var _result = _execute.execute("QuantizedDepthwiseConv2DWithBiasAndRelu", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedDepthwiseConv2DWithBiasAndRelu", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Computes quantized depthwise Conv2D with Bias, Relu and Requantize.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="filter"></param>
    /// <param name="bias"></param>
    /// <param name="min_input"></param>
    /// <param name="max_input"></param>
    /// <param name="min_filter"></param>
    /// <param name="max_filter"></param>
    /// <param name="min_freezed_output"></param>
    /// <param name="max_freezed_output"></param>
    /// <param name="out_type">
    /// The type of the output.
    /// </param>
    /// <param name="strides">
    /// List of stride values.
    /// </param>
    /// <param name="padding"></param>
    /// <param name="dilations">
    /// List of dilation values.
    /// </param>
    /// <param name="padding_list"></param>
    /// <returns></returns>
    public static Tensor[] quantized_depthwise_conv2d_with_bias_and_relu_and_requantize(Tensor input, Tensor filter, Tensor bias, Tensor min_input, Tensor max_input, Tensor min_filter, Tensor max_filter, Tensor min_freezed_output, Tensor max_freezed_output, int[] strides, string padding, TF_DataType out_type = TF_DataType.TF_QUINT8, int[] dilations = null, int[] padding_list = null, string? name = null)
    {
        var _ctx = tf.Context;
        if (dilations is null)
        {
            dilations = new int[] { 1, 1, 1, 1 };
        }
        if (padding_list is null)
        {
            padding_list = new int[] { };
        }
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize", name) { args = new object[] { input, filter, bias, min_input, max_input, min_filter, max_filter, min_freezed_output, max_freezed_output }, attrs = new Dictionary<string, object>() { ["out_type"] = out_type, ["strides"] = strides, ["padding"] = padding, ["dilations"] = dilations, ["padding_list"] = padding_list } });
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
                return quantized_depthwise_conv2d_with_bias_and_relu_and_requantize_eager_fallback(input, filter, bias, min_input, max_input, min_filter, max_filter, min_freezed_output, max_freezed_output, out_type: out_type, strides: strides, padding: padding, dilations: dilations, padding_list: padding_list, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["filter"] = filter;
        keywords["bias"] = bias;
        keywords["min_input"] = min_input;
        keywords["max_input"] = max_input;
        keywords["min_filter"] = min_filter;
        keywords["max_filter"] = max_filter;
        keywords["min_freezed_output"] = min_freezed_output;
        keywords["max_freezed_output"] = max_freezed_output;
        keywords["out_type"] = out_type;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        keywords["dilations"] = dilations;
        keywords["padding_list"] = padding_list;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tinput", _op._get_attr_type("Tinput"), "Tfilter", _op._get_attr_type("Tfilter"), "Tbias", _op._get_attr_type("Tbias"), "out_type", _op._get_attr_type("out_type"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding"), "dilations", _op.get_attr("dilations"), "padding_list", _op.get_attr("padding_list") };
            _execute.record_gradient("QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_depthwise_conv2d_with_bias_and_relu_and_requantize_eager_fallback(Tensor input, Tensor filter, Tensor bias, Tensor min_input, Tensor max_input, Tensor min_filter, Tensor max_filter, Tensor min_freezed_output, Tensor max_freezed_output, TF_DataType out_type, int[] strides, string padding, int[] dilations, int[] padding_list, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, filter, bias, min_input, max_input, min_filter, max_filter, min_freezed_output, max_freezed_output };
        object[] _attrs = new object[] { "Tinput", input.dtype, "Tfilter", filter.dtype, "Tbias", bias.dtype, "out_type", out_type, "strides", strides, "padding", padding, "dilations", dilations, "padding_list", padding_list };
        var _result = _execute.execute("QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// ~~%~~Performs a quantized matrix multiplication of `a` by the matrix `b` with bias~~%~~add.~~%~~
    /// </summary>
    /// <remarks>
    /// 
    /// The inputs must be two-dimensional matrices and 1D bias vector. And the inner
    /// dimension of `a` (after being transposed if `transpose_a` is non-zero) must
    /// match the outer dimension of `b` (after being transposed if `transposed_b` is
    /// non-zero). Then do broadcast add operation with bias values on the matrix
    /// multiplication result. The bias size must match inner dimension of `b`.
    /// 
    /// </remarks>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <param name="bias"></param>
    /// <param name="min_a"></param>
    /// <param name="max_a"></param>
    /// <param name="min_b"></param>
    /// <param name="max_b"></param>
    /// <param name="Toutput"></param>
    /// <param name="transpose_a">
    /// If true, `a` is transposed before multiplication.
    /// </param>
    /// <param name="transpose_b">
    /// If true, `b` is transposed before multiplication.
    /// </param>
    /// <param name="input_quant_mode">
    /// 
    /// Input data quantization mode. Either MIN_FIRST(default) or SCALED.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] quantized_mat_mul_with_bias(Tensor a, Tensor b, Tensor bias, Tensor min_a, Tensor max_a, Tensor min_b, Tensor max_b, TF_DataType Toutput = TF_DataType.TF_QINT32, bool transpose_a = false, bool transpose_b = false, string input_quant_mode = "MIN_FIRST", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedMatMulWithBias", name) { args = new object[] { a, b, bias, min_a, max_a, min_b, max_b }, attrs = new Dictionary<string, object>() { ["Toutput"] = Toutput, ["transpose_a"] = transpose_a, ["transpose_b"] = transpose_b, ["input_quant_mode"] = input_quant_mode } });
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
                return quantized_mat_mul_with_bias_eager_fallback(a, b, bias, min_a, max_a, min_b, max_b, Toutput: Toutput, transpose_a: transpose_a, transpose_b: transpose_b, input_quant_mode: input_quant_mode, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        if (input_quant_mode is null)
        {
            input_quant_mode = "MIN_FIRST";
        }
        Dictionary<string, object> keywords = new();
        keywords["a"] = a;
        keywords["b"] = b;
        keywords["bias"] = bias;
        keywords["min_a"] = min_a;
        keywords["max_a"] = max_a;
        keywords["min_b"] = min_b;
        keywords["max_b"] = max_b;
        keywords["Toutput"] = Toutput;
        keywords["transpose_a"] = transpose_a;
        keywords["transpose_b"] = transpose_b;
        keywords["input_quant_mode"] = input_quant_mode;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedMatMulWithBias", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T1", _op._get_attr_type("T1"), "T2", _op._get_attr_type("T2"), "Tbias", _op._get_attr_type("Tbias"), "Toutput", _op._get_attr_type("Toutput"), "transpose_a", _op._get_attr_bool("transpose_a"), "transpose_b", _op._get_attr_bool("transpose_b"), "input_quant_mode", _op.get_attr("input_quant_mode") };
            _execute.record_gradient("QuantizedMatMulWithBias", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_mat_mul_with_bias_eager_fallback(Tensor a, Tensor b, Tensor bias, Tensor min_a, Tensor max_a, Tensor min_b, Tensor max_b, TF_DataType Toutput, bool transpose_a, bool transpose_b, string input_quant_mode, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { a, b, bias, min_a, max_a, min_b, max_b };
        object[] _attrs = new object[] { "T1", a.dtype, "T2", b.dtype, "Tbias", bias.dtype, "Toutput", Toutput, "transpose_a", transpose_a, "transpose_b", transpose_b, "input_quant_mode", input_quant_mode };
        var _result = _execute.execute("QuantizedMatMulWithBias", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedMatMulWithBias", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <param name="bias"></param>
    /// <param name="min_a"></param>
    /// <param name="max_a"></param>
    /// <param name="min_b"></param>
    /// <param name="max_b"></param>
    /// <param name="min_freezed_output"></param>
    /// <param name="max_freezed_output"></param>
    /// <param name="Toutput"></param>
    /// <param name="transpose_a"></param>
    /// <param name="transpose_b"></param>
    /// <param name="input_quant_mode"></param>
    /// <returns></returns>
    public static Tensor quantized_mat_mul_with_bias_and_dequantize(Tensor a, Tensor b, Tensor bias, Tensor min_a, Tensor max_a, Tensor min_b, Tensor max_b, Tensor min_freezed_output, Tensor max_freezed_output, TF_DataType Toutput, bool transpose_a = false, bool transpose_b = false, string input_quant_mode = "MIN_FIRST", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedMatMulWithBiasAndDequantize", name) { args = new object[] { a, b, bias, min_a, max_a, min_b, max_b, min_freezed_output, max_freezed_output }, attrs = new Dictionary<string, object>() { ["Toutput"] = Toutput, ["transpose_a"] = transpose_a, ["transpose_b"] = transpose_b, ["input_quant_mode"] = input_quant_mode } });
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
                return quantized_mat_mul_with_bias_and_dequantize_eager_fallback(a, b, bias, min_a, max_a, min_b, max_b, min_freezed_output, max_freezed_output, Toutput: Toutput, transpose_a: transpose_a, transpose_b: transpose_b, input_quant_mode: input_quant_mode, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        if (input_quant_mode is null)
        {
            input_quant_mode = "MIN_FIRST";
        }
        Dictionary<string, object> keywords = new();
        keywords["a"] = a;
        keywords["b"] = b;
        keywords["bias"] = bias;
        keywords["min_a"] = min_a;
        keywords["max_a"] = max_a;
        keywords["min_b"] = min_b;
        keywords["max_b"] = max_b;
        keywords["min_freezed_output"] = min_freezed_output;
        keywords["max_freezed_output"] = max_freezed_output;
        keywords["Toutput"] = Toutput;
        keywords["transpose_a"] = transpose_a;
        keywords["transpose_b"] = transpose_b;
        keywords["input_quant_mode"] = input_quant_mode;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedMatMulWithBiasAndDequantize", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T1", _op._get_attr_type("T1"), "T2", _op._get_attr_type("T2"), "Tbias", _op._get_attr_type("Tbias"), "Toutput", _op._get_attr_type("Toutput"), "transpose_a", _op._get_attr_bool("transpose_a"), "transpose_b", _op._get_attr_bool("transpose_b"), "input_quant_mode", _op.get_attr("input_quant_mode") };
            _execute.record_gradient("QuantizedMatMulWithBiasAndDequantize", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor quantized_mat_mul_with_bias_and_dequantize_eager_fallback(Tensor a, Tensor b, Tensor bias, Tensor min_a, Tensor max_a, Tensor min_b, Tensor max_b, Tensor min_freezed_output, Tensor max_freezed_output, TF_DataType Toutput, bool transpose_a, bool transpose_b, string input_quant_mode, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { a, b, bias, min_a, max_a, min_b, max_b, min_freezed_output, max_freezed_output };
        object[] _attrs = new object[] { "T1", a.dtype, "T2", b.dtype, "Tbias", bias.dtype, "Toutput", Toutput, "transpose_a", transpose_a, "transpose_b", transpose_b, "input_quant_mode", input_quant_mode };
        var _result = _execute.execute("QuantizedMatMulWithBiasAndDequantize", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedMatMulWithBiasAndDequantize", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// ~~%~~Perform a quantized matrix multiplication of  `a` by the matrix `b` with bias~~%~~add and relu fusion.~~%~~
    /// </summary>
    /// <remarks>
    /// 
    /// The inputs must be two-dimensional matrices and 1D bias vector. And the inner
    /// dimension of `a` (after being transposed if `transpose_a` is non-zero) must
    /// match the outer dimension of `b` (after being transposed if `transposed_b` is
    /// non-zero). Then do broadcast add operation with bias values on the matrix
    /// multiplication result. The bias size must match inner dimension of `b`. Then do
    /// relu activation to get non-negative result.
    /// 
    /// </remarks>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <param name="bias"></param>
    /// <param name="min_a"></param>
    /// <param name="max_a"></param>
    /// <param name="min_b"></param>
    /// <param name="max_b"></param>
    /// <param name="Toutput"></param>
    /// <param name="transpose_a">
    /// If true, `a` is transposed before multiplication.
    /// </param>
    /// <param name="transpose_b">
    /// If true, `b` is transposed before multiplication.
    /// </param>
    /// <param name="input_quant_mode">
    /// 
    /// Input data quantization mode. Either MIN_FIRST(default) or SCALED.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] quantized_mat_mul_with_bias_and_relu(Tensor a, Tensor b, Tensor bias, Tensor min_a, Tensor max_a, Tensor min_b, Tensor max_b, TF_DataType Toutput = TF_DataType.TF_QINT32, bool transpose_a = false, bool transpose_b = false, string input_quant_mode = "MIN_FIRST", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedMatMulWithBiasAndRelu", name) { args = new object[] { a, b, bias, min_a, max_a, min_b, max_b }, attrs = new Dictionary<string, object>() { ["Toutput"] = Toutput, ["transpose_a"] = transpose_a, ["transpose_b"] = transpose_b, ["input_quant_mode"] = input_quant_mode } });
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
                return quantized_mat_mul_with_bias_and_relu_eager_fallback(a, b, bias, min_a, max_a, min_b, max_b, Toutput: Toutput, transpose_a: transpose_a, transpose_b: transpose_b, input_quant_mode: input_quant_mode, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        if (input_quant_mode is null)
        {
            input_quant_mode = "MIN_FIRST";
        }
        Dictionary<string, object> keywords = new();
        keywords["a"] = a;
        keywords["b"] = b;
        keywords["bias"] = bias;
        keywords["min_a"] = min_a;
        keywords["max_a"] = max_a;
        keywords["min_b"] = min_b;
        keywords["max_b"] = max_b;
        keywords["Toutput"] = Toutput;
        keywords["transpose_a"] = transpose_a;
        keywords["transpose_b"] = transpose_b;
        keywords["input_quant_mode"] = input_quant_mode;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedMatMulWithBiasAndRelu", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T1", _op._get_attr_type("T1"), "T2", _op._get_attr_type("T2"), "Toutput", _op._get_attr_type("Toutput"), "transpose_a", _op._get_attr_bool("transpose_a"), "transpose_b", _op._get_attr_bool("transpose_b"), "input_quant_mode", _op.get_attr("input_quant_mode") };
            _execute.record_gradient("QuantizedMatMulWithBiasAndRelu", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_mat_mul_with_bias_and_relu_eager_fallback(Tensor a, Tensor b, Tensor bias, Tensor min_a, Tensor max_a, Tensor min_b, Tensor max_b, TF_DataType Toutput, bool transpose_a, bool transpose_b, string input_quant_mode, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { a, b, bias, min_a, max_a, min_b, max_b };
        object[] _attrs = new object[] { "T1", a.dtype, "T2", b.dtype, "Toutput", Toutput, "transpose_a", transpose_a, "transpose_b", transpose_b, "input_quant_mode", input_quant_mode };
        var _result = _execute.execute("QuantizedMatMulWithBiasAndRelu", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedMatMulWithBiasAndRelu", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// ~~%~~Perform a quantized matrix multiplication of  `a` by the matrix `b` with bias~~%~~add and relu and requantize fusion.~~%~~
    /// </summary>
    /// <remarks>
    /// 
    /// The inputs must be two-dimensional matrices and 1D bias vector. And the inner
    /// dimension of `a` (after being transposed if `transpose_a` is non-zero) must
    /// match the outer dimension of `b` (after being transposed if `transposed_b` is
    /// non-zero). Then do broadcast add operation with bias values on the matrix
    /// multiplication result. The bias size must match inner dimension of `b`.  Then do
    /// relu activation to get non-negative result. Then do requantize operation to get
    /// final uint8 result.
    /// 
    /// </remarks>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <param name="bias"></param>
    /// <param name="min_a"></param>
    /// <param name="max_a"></param>
    /// <param name="min_b"></param>
    /// <param name="max_b"></param>
    /// <param name="min_freezed_output"></param>
    /// <param name="max_freezed_output"></param>
    /// <param name="Toutput"></param>
    /// <param name="transpose_a">
    /// If true, `a` is transposed before multiplication.
    /// </param>
    /// <param name="transpose_b">
    /// If true, `b` is transposed before multiplication.
    /// </param>
    /// <param name="input_quant_mode">
    /// 
    /// Input data quantization mode. Either MIN_FIRST(default) or SCALED.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] quantized_mat_mul_with_bias_and_relu_and_requantize(Tensor a, Tensor b, Tensor bias, Tensor min_a, Tensor max_a, Tensor min_b, Tensor max_b, Tensor min_freezed_output, Tensor max_freezed_output, TF_DataType Toutput = TF_DataType.TF_QUINT8, bool transpose_a = false, bool transpose_b = false, string input_quant_mode = "MIN_FIRST", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedMatMulWithBiasAndReluAndRequantize", name) { args = new object[] { a, b, bias, min_a, max_a, min_b, max_b, min_freezed_output, max_freezed_output }, attrs = new Dictionary<string, object>() { ["Toutput"] = Toutput, ["transpose_a"] = transpose_a, ["transpose_b"] = transpose_b, ["input_quant_mode"] = input_quant_mode } });
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
                return quantized_mat_mul_with_bias_and_relu_and_requantize_eager_fallback(a, b, bias, min_a, max_a, min_b, max_b, min_freezed_output, max_freezed_output, Toutput: Toutput, transpose_a: transpose_a, transpose_b: transpose_b, input_quant_mode: input_quant_mode, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        if (input_quant_mode is null)
        {
            input_quant_mode = "MIN_FIRST";
        }
        Dictionary<string, object> keywords = new();
        keywords["a"] = a;
        keywords["b"] = b;
        keywords["bias"] = bias;
        keywords["min_a"] = min_a;
        keywords["max_a"] = max_a;
        keywords["min_b"] = min_b;
        keywords["max_b"] = max_b;
        keywords["min_freezed_output"] = min_freezed_output;
        keywords["max_freezed_output"] = max_freezed_output;
        keywords["Toutput"] = Toutput;
        keywords["transpose_a"] = transpose_a;
        keywords["transpose_b"] = transpose_b;
        keywords["input_quant_mode"] = input_quant_mode;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedMatMulWithBiasAndReluAndRequantize", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T1", _op._get_attr_type("T1"), "T2", _op._get_attr_type("T2"), "Tbias", _op._get_attr_type("Tbias"), "Toutput", _op._get_attr_type("Toutput"), "transpose_a", _op._get_attr_bool("transpose_a"), "transpose_b", _op._get_attr_bool("transpose_b"), "input_quant_mode", _op.get_attr("input_quant_mode") };
            _execute.record_gradient("QuantizedMatMulWithBiasAndReluAndRequantize", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_mat_mul_with_bias_and_relu_and_requantize_eager_fallback(Tensor a, Tensor b, Tensor bias, Tensor min_a, Tensor max_a, Tensor min_b, Tensor max_b, Tensor min_freezed_output, Tensor max_freezed_output, TF_DataType Toutput, bool transpose_a, bool transpose_b, string input_quant_mode, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { a, b, bias, min_a, max_a, min_b, max_b, min_freezed_output, max_freezed_output };
        object[] _attrs = new object[] { "T1", a.dtype, "T2", b.dtype, "Tbias", bias.dtype, "Toutput", Toutput, "transpose_a", transpose_a, "transpose_b", transpose_b, "input_quant_mode", input_quant_mode };
        var _result = _execute.execute("QuantizedMatMulWithBiasAndReluAndRequantize", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedMatMulWithBiasAndReluAndRequantize", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <param name="bias"></param>
    /// <param name="min_a"></param>
    /// <param name="max_a"></param>
    /// <param name="min_b"></param>
    /// <param name="max_b"></param>
    /// <param name="min_freezed_output"></param>
    /// <param name="max_freezed_output"></param>
    /// <param name="Toutput"></param>
    /// <param name="transpose_a"></param>
    /// <param name="transpose_b"></param>
    /// <param name="input_quant_mode"></param>
    /// <returns></returns>
    public static Tensor[] quantized_mat_mul_with_bias_and_requantize(Tensor a, Tensor b, Tensor bias, Tensor min_a, Tensor max_a, Tensor min_b, Tensor max_b, Tensor min_freezed_output, Tensor max_freezed_output, TF_DataType Toutput = TF_DataType.TF_QUINT8, bool transpose_a = false, bool transpose_b = false, string input_quant_mode = "MIN_FIRST", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedMatMulWithBiasAndRequantize", name) { args = new object[] { a, b, bias, min_a, max_a, min_b, max_b, min_freezed_output, max_freezed_output }, attrs = new Dictionary<string, object>() { ["Toutput"] = Toutput, ["transpose_a"] = transpose_a, ["transpose_b"] = transpose_b, ["input_quant_mode"] = input_quant_mode } });
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
                return quantized_mat_mul_with_bias_and_requantize_eager_fallback(a, b, bias, min_a, max_a, min_b, max_b, min_freezed_output, max_freezed_output, Toutput: Toutput, transpose_a: transpose_a, transpose_b: transpose_b, input_quant_mode: input_quant_mode, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        if (input_quant_mode is null)
        {
            input_quant_mode = "MIN_FIRST";
        }
        Dictionary<string, object> keywords = new();
        keywords["a"] = a;
        keywords["b"] = b;
        keywords["bias"] = bias;
        keywords["min_a"] = min_a;
        keywords["max_a"] = max_a;
        keywords["min_b"] = min_b;
        keywords["max_b"] = max_b;
        keywords["min_freezed_output"] = min_freezed_output;
        keywords["max_freezed_output"] = max_freezed_output;
        keywords["Toutput"] = Toutput;
        keywords["transpose_a"] = transpose_a;
        keywords["transpose_b"] = transpose_b;
        keywords["input_quant_mode"] = input_quant_mode;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedMatMulWithBiasAndRequantize", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T1", _op._get_attr_type("T1"), "T2", _op._get_attr_type("T2"), "Tbias", _op._get_attr_type("Tbias"), "Toutput", _op._get_attr_type("Toutput"), "transpose_a", _op._get_attr_bool("transpose_a"), "transpose_b", _op._get_attr_bool("transpose_b"), "input_quant_mode", _op.get_attr("input_quant_mode") };
            _execute.record_gradient("QuantizedMatMulWithBiasAndRequantize", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_mat_mul_with_bias_and_requantize_eager_fallback(Tensor a, Tensor b, Tensor bias, Tensor min_a, Tensor max_a, Tensor min_b, Tensor max_b, Tensor min_freezed_output, Tensor max_freezed_output, TF_DataType Toutput, bool transpose_a, bool transpose_b, string input_quant_mode, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { a, b, bias, min_a, max_a, min_b, max_b, min_freezed_output, max_freezed_output };
        object[] _attrs = new object[] { "T1", a.dtype, "T2", b.dtype, "Tbias", bias.dtype, "Toutput", Toutput, "transpose_a", transpose_a, "transpose_b", transpose_b, "input_quant_mode", input_quant_mode };
        var _result = _execute.execute("QuantizedMatMulWithBiasAndRequantize", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedMatMulWithBiasAndRequantize", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Produces the max pool of the input tensor for quantized types.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="min_input"></param>
    /// <param name="max_input"></param>
    /// <param name="ksize">
    /// 
    /// The size of the window for each dimension of the input tensor.
    /// The length must be 4 to match the number of dimensions of the input.
    /// 
    /// </param>
    /// <param name="strides">
    /// 
    /// The stride of the sliding window for each dimension of the input
    /// tensor. The length must be 4 to match the number of dimensions of the input.
    /// 
    /// </param>
    /// <param name="padding">
    /// 
    /// The type of padding algorithm to use.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] quantized_max_pool(Tensor input, Tensor min_input, Tensor max_input, int[] ksize, int[] strides, string padding, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedMaxPool", name) { args = new object[] { input, min_input, max_input }, attrs = new Dictionary<string, object>() { ["ksize"] = ksize, ["strides"] = strides, ["padding"] = padding } });
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
                return quantized_max_pool_eager_fallback(input, min_input, max_input, ksize: ksize, strides: strides, padding: padding, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["min_input"] = min_input;
        keywords["max_input"] = max_input;
        keywords["ksize"] = ksize;
        keywords["strides"] = strides;
        keywords["padding"] = padding;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedMaxPool", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "ksize", _op.get_attr("ksize"), "strides", _op.get_attr("strides"), "padding", _op.get_attr("padding") };
            _execute.record_gradient("QuantizedMaxPool", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_max_pool_eager_fallback(Tensor input, Tensor min_input, Tensor max_input, int[] ksize, int[] strides, string padding, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, min_input, max_input };
        object[] _attrs = new object[] { "T", input.dtype, "ksize", ksize, "strides", strides, "padding", padding };
        var _result = _execute.execute("QuantizedMaxPool", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedMaxPool", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Computes Quantized Rectified Linear: `max(features, 0)`
    /// </summary>
    /// <param name="features"></param>
    /// <param name="min_features"></param>
    /// <param name="max_features"></param>
    /// <param name="out_type"></param>
    /// <returns></returns>
    public static Tensor[] quantized_relu(Tensor features, Tensor min_features, Tensor max_features, TF_DataType out_type = TF_DataType.TF_QUINT8, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedRelu", name) { args = new object[] { features, min_features, max_features }, attrs = new Dictionary<string, object>() { ["out_type"] = out_type } });
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
                return quantized_relu_eager_fallback(features, min_features, max_features, out_type: out_type, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["features"] = features;
        keywords["min_features"] = min_features;
        keywords["max_features"] = max_features;
        keywords["out_type"] = out_type;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedRelu", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tinput", _op._get_attr_type("Tinput"), "out_type", _op._get_attr_type("out_type") };
            _execute.record_gradient("QuantizedRelu", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_relu_eager_fallback(Tensor features, Tensor min_features, Tensor max_features, TF_DataType out_type, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { features, min_features, max_features };
        object[] _attrs = new object[] { "Tinput", features.dtype, "out_type", out_type };
        var _result = _execute.execute("QuantizedRelu", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedRelu", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Computes Quantized Rectified Linear 6: `min(max(features, 0), 6)`
    /// </summary>
    /// <param name="features"></param>
    /// <param name="min_features"></param>
    /// <param name="max_features"></param>
    /// <param name="out_type"></param>
    /// <returns></returns>
    public static Tensor[] quantized_relu6(Tensor features, Tensor min_features, Tensor max_features, TF_DataType out_type = TF_DataType.TF_QUINT8, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedRelu6", name) { args = new object[] { features, min_features, max_features }, attrs = new Dictionary<string, object>() { ["out_type"] = out_type } });
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
                return quantized_relu6_eager_fallback(features, min_features, max_features, out_type: out_type, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["features"] = features;
        keywords["min_features"] = min_features;
        keywords["max_features"] = max_features;
        keywords["out_type"] = out_type;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedRelu6", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tinput", _op._get_attr_type("Tinput"), "out_type", _op._get_attr_type("out_type") };
            _execute.record_gradient("QuantizedRelu6", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_relu6_eager_fallback(Tensor features, Tensor min_features, Tensor max_features, TF_DataType out_type, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { features, min_features, max_features };
        object[] _attrs = new object[] { "Tinput", features.dtype, "out_type", out_type };
        var _result = _execute.execute("QuantizedRelu6", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedRelu6", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Computes Quantized Rectified Linear X: `min(max(features, 0), max_value)`
    /// </summary>
    /// <param name="features"></param>
    /// <param name="max_value"></param>
    /// <param name="min_features"></param>
    /// <param name="max_features"></param>
    /// <param name="out_type"></param>
    /// <returns></returns>
    public static Tensor[] quantized_relu_x(Tensor features, Tensor max_value, Tensor min_features, Tensor max_features, TF_DataType out_type = TF_DataType.TF_QUINT8, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "QuantizedReluX", name) { args = new object[] { features, max_value, min_features, max_features }, attrs = new Dictionary<string, object>() { ["out_type"] = out_type } });
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
                return quantized_relu_x_eager_fallback(features, max_value, min_features, max_features, out_type: out_type, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["features"] = features;
        keywords["max_value"] = max_value;
        keywords["min_features"] = min_features;
        keywords["max_features"] = max_features;
        keywords["out_type"] = out_type;
        var _op = tf.OpDefLib._apply_op_helper("QuantizedReluX", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tinput", _op._get_attr_type("Tinput"), "out_type", _op._get_attr_type("out_type") };
            _execute.record_gradient("QuantizedReluX", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] quantized_relu_x_eager_fallback(Tensor features, Tensor max_value, Tensor min_features, Tensor max_features, TF_DataType out_type, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { features, max_value, min_features, max_features };
        object[] _attrs = new object[] { "Tinput", features.dtype, "out_type", out_type };
        var _result = _execute.execute("QuantizedReluX", 3, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("QuantizedReluX", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Computes rectified linear: `max(features, 0)`.
    /// </summary>
    /// <remarks>
    /// 
    /// See: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    /// Example usage:
    /// >>> tf.nn.relu([-2., 0., 3.]).numpy()
    /// array([0., 0., 3.], dtype=float32)
    /// 
    /// </remarks>
    /// <param name="features"></param>
    /// <returns></returns>
    public static Tensor relu(Tensor features, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Relu", name) { args = new object[] { features }, attrs = new Dictionary<string, object>() { } });
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
                return relu_eager_fallback(features, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["features"] = features;
        var _op = tf.OpDefLib._apply_op_helper("Relu", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Relu", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor relu_eager_fallback(Tensor features, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { features };
        object[] _attrs = new object[] { "T", features.dtype };
        var _result = _execute.execute("Relu", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Relu", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes rectified linear 6: `min(max(features, 0), 6)`.
    /// </summary>
    /// <param name="features"></param>
    /// <returns></returns>
    public static Tensor relu6(Tensor features, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Relu6", name) { args = new object[] { features }, attrs = new Dictionary<string, object>() { } });
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
                return relu6_eager_fallback(features, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["features"] = features;
        var _op = tf.OpDefLib._apply_op_helper("Relu6", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Relu6", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor relu6_eager_fallback(Tensor features, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { features };
        object[] _attrs = new object[] { "T", features.dtype };
        var _result = _execute.execute("Relu6", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Relu6", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes rectified linear gradients for a Relu operation.
    /// </summary>
    /// <param name="gradients"></param>
    /// <param name="features"></param>
    /// <returns></returns>
    public static Tensor relu_grad(Tensor gradients, Tensor features, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ReluGrad", name) { args = new object[] { gradients, features }, attrs = new Dictionary<string, object>() { } });
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
                return relu_grad_eager_fallback(gradients, features, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["gradients"] = gradients;
        keywords["features"] = features;
        var _op = tf.OpDefLib._apply_op_helper("ReluGrad", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("ReluGrad", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor relu_grad_eager_fallback(Tensor gradients, Tensor features, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { gradients, features };
        object[] _attrs = new object[] { "T", gradients.dtype };
        var _result = _execute.execute("ReluGrad", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ReluGrad", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes scaled exponential linear: `scale * alpha * (exp(features) - 1)`
    /// </summary>
    /// <remarks>
    /// 
    /// if < 0, `scale * features` otherwise.
    /// 
    /// To be used together with
    /// `initializer = tf.variance_scaling_initializer(factor=1.0, mode='FAN_IN')`.
    /// For correct dropout, use `tf.contrib.nn.alpha_dropout`.
    /// 
    /// See [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    /// 
    /// </remarks>
    /// <param name="features"></param>
    /// <returns></returns>
    public static Tensor selu(Tensor features, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Selu", name) { args = new object[] { features }, attrs = new Dictionary<string, object>() { } });
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
                return selu_eager_fallback(features, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["features"] = features;
        var _op = tf.OpDefLib._apply_op_helper("Selu", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Selu", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor selu_eager_fallback(Tensor features, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { features };
        object[] _attrs = new object[] { "T", features.dtype };
        var _result = _execute.execute("Selu", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Selu", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes gradients for the scaled exponential linear (Selu) operation.
    /// </summary>
    /// <param name="gradients"></param>
    /// <param name="outputs"></param>
    /// <returns></returns>
    public static Tensor selu_grad(Tensor gradients, Tensor outputs, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SeluGrad", name) { args = new object[] { gradients, outputs }, attrs = new Dictionary<string, object>() { } });
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
                return selu_grad_eager_fallback(gradients, outputs, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["gradients"] = gradients;
        keywords["outputs"] = outputs;
        var _op = tf.OpDefLib._apply_op_helper("SeluGrad", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("SeluGrad", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor selu_grad_eager_fallback(Tensor gradients, Tensor outputs, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { gradients, outputs };
        object[] _attrs = new object[] { "T", gradients.dtype };
        var _result = _execute.execute("SeluGrad", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SeluGrad", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes softmax activations.
    /// </summary>
    /// <remarks>
    /// 
    /// For each batch `i` and class `j` we have
    /// 
    ///     $$softmax[i, j] = exp(logits[i, j]) / sum_j(exp(logits[i, j]))$$
    /// 
    /// </remarks>
    /// <param name="logits"></param>
    /// <returns></returns>
    public static Tensor softmax(Tensor logits, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Softmax", name) { args = new object[] { logits }, attrs = new Dictionary<string, object>() { } });
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
                return softmax_eager_fallback(logits, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["logits"] = logits;
        var _op = tf.OpDefLib._apply_op_helper("Softmax", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Softmax", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor softmax_eager_fallback(Tensor logits, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { logits };
        object[] _attrs = new object[] { "T", logits.dtype };
        var _result = _execute.execute("Softmax", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Softmax", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes softmax cross entropy cost and gradients to backpropagate.
    /// </summary>
    /// <remarks>
    /// 
    /// Inputs are the logits, not probabilities.
    /// 
    /// </remarks>
    /// <param name="features"></param>
    /// <param name="labels"></param>
    /// <returns></returns>
    public static Tensor[] softmax_cross_entropy_with_logits(Tensor features, Tensor labels, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SoftmaxCrossEntropyWithLogits", name) { args = new object[] { features, labels }, attrs = new Dictionary<string, object>() { } });
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
                return softmax_cross_entropy_with_logits_eager_fallback(features, labels, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["features"] = features;
        keywords["labels"] = labels;
        var _op = tf.OpDefLib._apply_op_helper("SoftmaxCrossEntropyWithLogits", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("SoftmaxCrossEntropyWithLogits", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] softmax_cross_entropy_with_logits_eager_fallback(Tensor features, Tensor labels, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { features, labels };
        object[] _attrs = new object[] { "T", features.dtype };
        var _result = _execute.execute("SoftmaxCrossEntropyWithLogits", 2, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SoftmaxCrossEntropyWithLogits", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="features"></param>
    /// <returns></returns>
    public static Tensor softplus(Tensor features, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Softplus", name) { args = new object[] { features }, attrs = new Dictionary<string, object>() { } });
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
                return softplus_eager_fallback(features, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["features"] = features;
        var _op = tf.OpDefLib._apply_op_helper("Softplus", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Softplus", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor softplus_eager_fallback(Tensor features, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { features };
        object[] _attrs = new object[] { "T", features.dtype };
        var _result = _execute.execute("Softplus", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Softplus", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes softplus gradients for a softplus operation.
    /// </summary>
    /// <param name="gradients"></param>
    /// <param name="features"></param>
    /// <returns></returns>
    public static Tensor softplus_grad(Tensor gradients, Tensor features, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SoftplusGrad", name) { args = new object[] { gradients, features }, attrs = new Dictionary<string, object>() { } });
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
                return softplus_grad_eager_fallback(gradients, features, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["gradients"] = gradients;
        keywords["features"] = features;
        var _op = tf.OpDefLib._apply_op_helper("SoftplusGrad", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("SoftplusGrad", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor softplus_grad_eager_fallback(Tensor gradients, Tensor features, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { gradients, features };
        object[] _attrs = new object[] { "T", gradients.dtype };
        var _result = _execute.execute("SoftplusGrad", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SoftplusGrad", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes softsign: `features / (abs(features) + 1)`.
    /// </summary>
    /// <param name="features"></param>
    /// <returns></returns>
    public static Tensor softsign(Tensor features, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Softsign", name) { args = new object[] { features }, attrs = new Dictionary<string, object>() { } });
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
                return softsign_eager_fallback(features, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["features"] = features;
        var _op = tf.OpDefLib._apply_op_helper("Softsign", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("Softsign", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor softsign_eager_fallback(Tensor features, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { features };
        object[] _attrs = new object[] { "T", features.dtype };
        var _result = _execute.execute("Softsign", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Softsign", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes softsign gradients for a softsign operation.
    /// </summary>
    /// <param name="gradients"></param>
    /// <param name="features"></param>
    /// <returns></returns>
    public static Tensor softsign_grad(Tensor gradients, Tensor features, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SoftsignGrad", name) { args = new object[] { gradients, features }, attrs = new Dictionary<string, object>() { } });
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
                return softsign_grad_eager_fallback(gradients, features, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["gradients"] = gradients;
        keywords["features"] = features;
        var _op = tf.OpDefLib._apply_op_helper("SoftsignGrad", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("SoftsignGrad", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor softsign_grad_eager_fallback(Tensor gradients, Tensor features, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { gradients, features };
        object[] _attrs = new object[] { "T", gradients.dtype };
        var _result = _execute.execute("SoftsignGrad", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SoftsignGrad", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Computes softmax cross entropy cost and gradients to backpropagate.
    /// </summary>
    /// <remarks>
    /// 
    /// Unlike `SoftmaxCrossEntropyWithLogits`, this operation does not accept
    /// a matrix of label probabilities, but rather a single label per row
    /// of features.  This label is considered to have probability 1.0 for the
    /// given row.
    /// 
    /// Inputs are the logits, not probabilities.
    /// 
    /// </remarks>
    /// <param name="features"></param>
    /// <param name="labels"></param>
    /// <returns></returns>
    public static Tensor[] sparse_softmax_cross_entropy_with_logits(Tensor features, Tensor labels, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SparseSoftmaxCrossEntropyWithLogits", name) { args = new object[] { features, labels }, attrs = new Dictionary<string, object>() { } });
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
                return sparse_softmax_cross_entropy_with_logits_eager_fallback(features, labels, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["features"] = features;
        keywords["labels"] = labels;
        var _op = tf.OpDefLib._apply_op_helper("SparseSoftmaxCrossEntropyWithLogits", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T"), "Tlabels", _op._get_attr_type("Tlabels") };
            _execute.record_gradient("SparseSoftmaxCrossEntropyWithLogits", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] sparse_softmax_cross_entropy_with_logits_eager_fallback(Tensor features, Tensor labels, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { features, labels };
        object[] _attrs = new object[] { "T", features.dtype, "Tlabels", labels.dtype };
        var _result = _execute.execute("SparseSoftmaxCrossEntropyWithLogits", 2, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SparseSoftmaxCrossEntropyWithLogits", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Finds values and indices of the `k` largest elements for the last dimension.
    /// </summary>
    /// <remarks>
    /// 
    /// If the input is a vector (rank-1), finds the `k` largest entries in the vector
    /// and outputs their values and indices as vectors.  Thus `values[j]` is the
    /// `j`-th largest entry in `input`, and its index is `indices[j]`.
    /// 
    /// For matrices (resp. higher rank input), computes the top `k` entries in each
    /// row (resp. vector along the last dimension).  Thus,
    /// 
    ///     values.shape = indices.shape = input.shape[:-1] + [k]
    /// 
    /// If two elements are equal, the lower-index element appears first.
    /// 
    /// If `k` varies dynamically, use `TopKV2` below.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="k">
    /// 
    /// Number of top elements to look for along the last dimension (along each
    /// row for matrices).
    /// 
    /// </param>
    /// <param name="sorted">
    /// 
    /// If true the resulting `k` elements will be sorted by the values in
    /// descending order.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] top_k(Tensor input, int k = 0, bool sorted = true, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TopK", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { ["k"] = k, ["sorted"] = sorted } });
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
                return top_k_eager_fallback(input, k: k, sorted: sorted, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["k"] = k;
        keywords["sorted"] = sorted;
        var _op = tf.OpDefLib._apply_op_helper("TopK", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "k", _op._get_attr_int("k"), "sorted", _op._get_attr_bool("sorted"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("TopK", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] top_k_eager_fallback(Tensor input, int k, bool sorted, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "k", k, "sorted", sorted, "T", input.dtype };
        var _result = _execute.execute("TopK", 2, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TopK", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Finds values and indices of the `k` largest elements for the last dimension.
    /// </summary>
    /// <remarks>
    /// 
    /// If the input is a vector (rank-1), finds the `k` largest entries in the vector
    /// and outputs their values and indices as vectors.  Thus `values[j]` is the
    /// `j`-th largest entry in `input`, and its index is `indices[j]`.
    /// 
    /// For matrices (resp. higher rank input), computes the top `k` entries in each
    /// row (resp. vector along the last dimension).  Thus,
    /// 
    ///     values.shape = indices.shape = input.shape[:-1] + [k]
    /// 
    /// If two elements are equal, the lower-index element appears first.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <param name="k"></param>
    /// <param name="sorted">
    /// 
    /// If true the resulting `k` elements will be sorted by the values in
    /// descending order.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] top_kv2(Tensor input, Tensor k, bool sorted = true, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TopKV2", name) { args = new object[] { input, k }, attrs = new Dictionary<string, object>() { ["sorted"] = sorted } });
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
                return top_kv2_eager_fallback(input, k, sorted: sorted, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["k"] = k;
        keywords["sorted"] = sorted;
        var _op = tf.OpDefLib._apply_op_helper("TopKV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "sorted", _op._get_attr_bool("sorted"), "T", _op._get_attr_type("T") };
            _execute.record_gradient("TopKV2", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] top_kv2_eager_fallback(Tensor input, Tensor k, bool sorted, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input, k };
        object[] _attrs = new object[] { "sorted", sorted, "T", input.dtype };
        var _result = _execute.execute("TopKV2", 2, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TopKV2", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
}
