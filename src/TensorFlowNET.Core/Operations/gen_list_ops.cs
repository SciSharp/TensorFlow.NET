/*Wrappers around TensorFlow ops. This file is MACHINE GENERATED! Do not edit.*/

using Tensorflow.Eager;
using Tensorflow.Contexts;
using Tensorflow.Exceptions;
using static Tensorflow.Binding;

namespace Tensorflow;

public static class gen_list_ops
{
    /// <summary>
    /// Creates and returns an empty tensor list.
    /// </summary>
    /// <remarks>
    /// 
    /// All list elements must be tensors of dtype element_dtype and shape compatible
    /// with element_shape.
    /// 
    /// handle: an empty tensor list.
    /// element_dtype: the type of elements in the list.
    /// element_shape: a shape compatible with that of elements in the list.
    /// 
    /// </remarks>
    /// <param name="element_shape"></param>
    /// <param name="max_num_elements"></param>
    /// <param name="element_dtype"></param>
    /// <returns></returns>
    public static Tensor empty_tensor_list(Tensor element_shape, Tensor max_num_elements, TF_DataType element_dtype, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "EmptyTensorList", name) { args = new object[] { element_shape, max_num_elements }, attrs = new Dictionary<string, object>() { ["element_dtype"] = element_dtype } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return empty_tensor_list_eager_fallback(element_shape, max_num_elements, element_dtype: element_dtype, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["element_shape"] = element_shape;
        keywords["max_num_elements"] = max_num_elements;
        keywords["element_dtype"] = element_dtype;
        var _op = tf.OpDefLib._apply_op_helper("EmptyTensorList", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "element_dtype", _op._get_attr_type("element_dtype"), "shape_type", _op._get_attr_type("shape_type") };
            _execute.record_gradient("EmptyTensorList", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor empty_tensor_list_eager_fallback(Tensor element_shape, Tensor max_num_elements, TF_DataType element_dtype, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { element_shape, max_num_elements };
        object[] _attrs = new object[] { "element_dtype", element_dtype, "shape_type", element_shape.dtype };
        var _result = _execute.execute("EmptyTensorList", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("EmptyTensorList", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Concats all tensors in the list along the 0th dimension.
    /// </summary>
    /// <remarks>
    /// 
    /// Requires that all tensors have the same shape except the first dimension.
    /// 
    /// input_handle: The input list.
    /// tensor: The concated result.
    /// lengths: Output tensor containing sizes of the 0th dimension of tensors in the list, used for computing the gradient.
    /// 
    /// 
    /// </remarks>
    /// <param name="input_handle"></param>
    /// <param name="element_dtype"></param>
    /// <param name="element_shape"></param>
    /// <returns></returns>
    public static Tensor[] tensor_list_concat(Tensor input_handle, TF_DataType element_dtype, Shape element_shape = null, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TensorListConcat", name) { args = new object[] { input_handle }, attrs = new Dictionary<string, object>() { ["element_dtype"] = element_dtype, ["element_shape"] = element_shape } });
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
                return tensor_list_concat_eager_fallback(input_handle, element_dtype: element_dtype, element_shape: element_shape, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input_handle"] = input_handle;
        keywords["element_dtype"] = element_dtype;
        keywords["element_shape"] = element_shape;
        var _op = tf.OpDefLib._apply_op_helper("TensorListConcat", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "element_dtype", _op._get_attr_type("element_dtype"), "element_shape", _op.get_attr("element_shape") };
            _execute.record_gradient("TensorListConcat", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] tensor_list_concat_eager_fallback(Tensor input_handle, TF_DataType element_dtype, Shape element_shape, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input_handle };
        object[] _attrs = new object[] { "element_dtype", element_dtype, "element_shape", element_shape };
        var _result = _execute.execute("TensorListConcat", 2, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TensorListConcat", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="input_a"></param>
    /// <param name="input_b"></param>
    /// <param name="element_dtype"></param>
    /// <returns></returns>
    public static Tensor tensor_list_concat_lists(Tensor input_a, Tensor input_b, TF_DataType element_dtype, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TensorListConcatLists", name) { args = new object[] { input_a, input_b }, attrs = new Dictionary<string, object>() { ["element_dtype"] = element_dtype } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return tensor_list_concat_lists_eager_fallback(input_a, input_b, element_dtype: element_dtype, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input_a"] = input_a;
        keywords["input_b"] = input_b;
        keywords["element_dtype"] = element_dtype;
        var _op = tf.OpDefLib._apply_op_helper("TensorListConcatLists", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "element_dtype", _op._get_attr_type("element_dtype") };
            _execute.record_gradient("TensorListConcatLists", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor tensor_list_concat_lists_eager_fallback(Tensor input_a, Tensor input_b, TF_DataType element_dtype, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input_a, input_b };
        object[] _attrs = new object[] { "element_dtype", element_dtype };
        var _result = _execute.execute("TensorListConcatLists", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TensorListConcatLists", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Concats all tensors in the list along the 0th dimension.
    /// </summary>
    /// <remarks>
    /// 
    /// Requires that all tensors have the same shape except the first dimension.
    /// 
    /// input_handle: The input list.
    /// element_shape: The shape of the uninitialized elements in the list. If the first
    ///   dimension is not -1, it is assumed that all list elements have the same
    ///   leading dim.
    /// leading_dims: The list of leading dims of uninitialized list elements. Used if
    ///   the leading dim of input_handle.element_shape or the element_shape input arg
    ///   is not already set.
    /// tensor: The concated result.
    /// lengths: Output tensor containing sizes of the 0th dimension of tensors in the list, used for computing the gradient.
    /// 
    /// 
    /// </remarks>
    /// <param name="input_handle"></param>
    /// <param name="element_shape"></param>
    /// <param name="leading_dims"></param>
    /// <param name="element_dtype"></param>
    /// <returns></returns>
    public static Tensor[] tensor_list_concat_v2(Tensor input_handle, Tensor element_shape, Tensor leading_dims, TF_DataType element_dtype, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TensorListConcatV2", name) { args = new object[] { input_handle, element_shape, leading_dims }, attrs = new Dictionary<string, object>() { ["element_dtype"] = element_dtype } });
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
                return tensor_list_concat_v2_eager_fallback(input_handle, element_shape, leading_dims, element_dtype: element_dtype, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input_handle"] = input_handle;
        keywords["element_shape"] = element_shape;
        keywords["leading_dims"] = leading_dims;
        keywords["element_dtype"] = element_dtype;
        var _op = tf.OpDefLib._apply_op_helper("TensorListConcatV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "element_dtype", _op._get_attr_type("element_dtype"), "shape_type", _op._get_attr_type("shape_type") };
            _execute.record_gradient("TensorListConcatV2", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] tensor_list_concat_v2_eager_fallback(Tensor input_handle, Tensor element_shape, Tensor leading_dims, TF_DataType element_dtype, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input_handle, element_shape, leading_dims };
        object[] _attrs = new object[] { "element_dtype", element_dtype, "shape_type", element_shape.dtype };
        var _result = _execute.execute("TensorListConcatV2", 2, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TensorListConcatV2", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// The shape of the elements of the given list, as a tensor.
    /// </summary>
    /// <remarks>
    /// 
    ///   input_handle: the list
    ///   element_shape: the shape of elements of the list
    /// 
    /// </remarks>
    /// <param name="input_handle"></param>
    /// <param name="shape_type"></param>
    /// <returns></returns>
    public static Tensor tensor_list_element_shape(Tensor input_handle, TF_DataType shape_type, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TensorListElementShape", name) { args = new object[] { input_handle }, attrs = new Dictionary<string, object>() { ["shape_type"] = shape_type } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return tensor_list_element_shape_eager_fallback(input_handle, shape_type: shape_type, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input_handle"] = input_handle;
        keywords["shape_type"] = shape_type;
        var _op = tf.OpDefLib._apply_op_helper("TensorListElementShape", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "shape_type", _op._get_attr_type("shape_type") };
            _execute.record_gradient("TensorListElementShape", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor tensor_list_element_shape_eager_fallback(Tensor input_handle, TF_DataType shape_type, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input_handle };
        object[] _attrs = new object[] { "shape_type", shape_type };
        var _result = _execute.execute("TensorListElementShape", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TensorListElementShape", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Creates a TensorList which, when stacked, has the value of `tensor`.
    /// </summary>
    /// <remarks>
    /// 
    /// Each tensor in the result list corresponds to one row of the input tensor.
    /// 
    /// tensor: The input tensor.
    /// output_handle: The list.
    /// 
    /// </remarks>
    /// <param name="tensor"></param>
    /// <param name="element_shape"></param>
    /// <returns></returns>
    public static Tensor tensor_list_from_tensor(Tensor tensor, Tensor element_shape, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TensorListFromTensor", name) { args = new object[] { tensor, element_shape }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return tensor_list_from_tensor_eager_fallback(tensor, element_shape, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["tensor"] = tensor;
        keywords["element_shape"] = element_shape;
        var _op = tf.OpDefLib._apply_op_helper("TensorListFromTensor", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "element_dtype", _op._get_attr_type("element_dtype"), "shape_type", _op._get_attr_type("shape_type") };
            _execute.record_gradient("TensorListFromTensor", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor tensor_list_from_tensor_eager_fallback(Tensor tensor, Tensor element_shape, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { tensor, element_shape };
        object[] _attrs = new object[] { "element_dtype", tensor.dtype, "shape_type", element_shape.dtype };
        var _result = _execute.execute("TensorListFromTensor", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TensorListFromTensor", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Creates a Tensor by indexing into the TensorList.
    /// </summary>
    /// <remarks>
    /// 
    /// Each row in the produced Tensor corresponds to the element in the TensorList
    /// specified by the given index (see `tf.gather`).
    /// 
    /// input_handle: The input tensor list.
    /// indices: The indices used to index into the list.
    /// values: The tensor.
    /// 
    /// </remarks>
    /// <param name="input_handle"></param>
    /// <param name="indices"></param>
    /// <param name="element_shape"></param>
    /// <param name="element_dtype"></param>
    /// <returns></returns>
    public static Tensor tensor_list_gather(Tensor input_handle, Tensor indices, Tensor element_shape, TF_DataType element_dtype, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TensorListGather", name) { args = new object[] { input_handle, indices, element_shape }, attrs = new Dictionary<string, object>() { ["element_dtype"] = element_dtype } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return tensor_list_gather_eager_fallback(input_handle, indices, element_shape, element_dtype: element_dtype, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input_handle"] = input_handle;
        keywords["indices"] = indices;
        keywords["element_shape"] = element_shape;
        keywords["element_dtype"] = element_dtype;
        var _op = tf.OpDefLib._apply_op_helper("TensorListGather", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "element_dtype", _op._get_attr_type("element_dtype") };
            _execute.record_gradient("TensorListGather", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor tensor_list_gather_eager_fallback(Tensor input_handle, Tensor indices, Tensor element_shape, TF_DataType element_dtype, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input_handle, indices, element_shape };
        object[] _attrs = new object[] { "element_dtype", element_dtype };
        var _result = _execute.execute("TensorListGather", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TensorListGather", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="input_handle"></param>
    /// <param name="index"></param>
    /// <param name="element_shape"></param>
    /// <param name="element_dtype"></param>
    /// <returns></returns>
    public static Tensor tensor_list_get_item(Tensor input_handle, Tensor index, Tensor element_shape, TF_DataType element_dtype, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TensorListGetItem", name) { args = new object[] { input_handle, index, element_shape }, attrs = new Dictionary<string, object>() { ["element_dtype"] = element_dtype } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return tensor_list_get_item_eager_fallback(input_handle, index, element_shape, element_dtype: element_dtype, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input_handle"] = input_handle;
        keywords["index"] = index;
        keywords["element_shape"] = element_shape;
        keywords["element_dtype"] = element_dtype;
        var _op = tf.OpDefLib._apply_op_helper("TensorListGetItem", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "element_dtype", _op._get_attr_type("element_dtype") };
            _execute.record_gradient("TensorListGetItem", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor tensor_list_get_item_eager_fallback(Tensor input_handle, Tensor index, Tensor element_shape, TF_DataType element_dtype, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input_handle, index, element_shape };
        object[] _attrs = new object[] { "element_dtype", element_dtype };
        var _result = _execute.execute("TensorListGetItem", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TensorListGetItem", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns the number of tensors in the input tensor list.
    /// </summary>
    /// <remarks>
    /// 
    /// input_handle: the input list
    /// length: the number of tensors in the list
    /// 
    /// </remarks>
    /// <param name="input_handle"></param>
    /// <returns></returns>
    public static Tensor tensor_list_length(Tensor input_handle, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TensorListLength", name) { args = new object[] { input_handle }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return tensor_list_length_eager_fallback(input_handle, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input_handle"] = input_handle;
        var _op = tf.OpDefLib._apply_op_helper("TensorListLength", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { };
            _execute.record_gradient("TensorListLength", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor tensor_list_length_eager_fallback(Tensor input_handle, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input_handle };
        object[] _attrs = new object[] { };
        var _result = _execute.execute("TensorListLength", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TensorListLength", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Returns the last element of the input list as well as a list with all but that element.
    /// </summary>
    /// <remarks>
    /// 
    /// Fails if the list is empty.
    /// 
    /// input_handle: the input list
    /// tensor: the withdrawn last element of the list
    /// element_dtype: the type of elements in the list
    /// element_shape: the shape of the output tensor
    /// 
    /// </remarks>
    /// <param name="input_handle"></param>
    /// <param name="element_shape"></param>
    /// <param name="element_dtype"></param>
    /// <returns></returns>
    public static Tensor[] tensor_list_pop_back(Tensor input_handle, Tensor element_shape, TF_DataType element_dtype, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TensorListPopBack", name) { args = new object[] { input_handle, element_shape }, attrs = new Dictionary<string, object>() { ["element_dtype"] = element_dtype } });
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
                return tensor_list_pop_back_eager_fallback(input_handle, element_shape, element_dtype: element_dtype, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input_handle"] = input_handle;
        keywords["element_shape"] = element_shape;
        keywords["element_dtype"] = element_dtype;
        var _op = tf.OpDefLib._apply_op_helper("TensorListPopBack", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "element_dtype", _op._get_attr_type("element_dtype") };
            _execute.record_gradient("TensorListPopBack", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] tensor_list_pop_back_eager_fallback(Tensor input_handle, Tensor element_shape, TF_DataType element_dtype, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input_handle, element_shape };
        object[] _attrs = new object[] { "element_dtype", element_dtype };
        var _result = _execute.execute("TensorListPopBack", 2, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TensorListPopBack", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Returns a list which has the passed-in `Tensor` as last element and the other elements of the given list in `input_handle`.
    /// </summary>
    /// <remarks>
    /// 
    /// tensor: The tensor to put on the list.
    /// input_handle: The old list.
    /// output_handle: A list with the elements of the old list followed by tensor.
    /// element_dtype: the type of elements in the list.
    /// element_shape: a shape compatible with that of elements in the list.
    /// 
    /// </remarks>
    /// <param name="input_handle"></param>
    /// <param name="tensor"></param>
    /// <returns></returns>
    public static Tensor tensor_list_push_back(Tensor input_handle, Tensor tensor, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TensorListPushBack", name) { args = new object[] { input_handle, tensor }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return tensor_list_push_back_eager_fallback(input_handle, tensor, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input_handle"] = input_handle;
        keywords["tensor"] = tensor;
        var _op = tf.OpDefLib._apply_op_helper("TensorListPushBack", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "element_dtype", _op._get_attr_type("element_dtype") };
            _execute.record_gradient("TensorListPushBack", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor tensor_list_push_back_eager_fallback(Tensor input_handle, Tensor tensor, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input_handle, tensor };
        object[] _attrs = new object[] { "element_dtype", tensor.dtype };
        var _result = _execute.execute("TensorListPushBack", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TensorListPushBack", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="input_handles"></param>
    /// <param name="tensor"></param>
    /// <returns></returns>
    public static Tensor tensor_list_push_back_batch(Tensor input_handles, Tensor tensor, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TensorListPushBackBatch", name) { args = new object[] { input_handles, tensor }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return tensor_list_push_back_batch_eager_fallback(input_handles, tensor, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input_handles"] = input_handles;
        keywords["tensor"] = tensor;
        var _op = tf.OpDefLib._apply_op_helper("TensorListPushBackBatch", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "element_dtype", _op._get_attr_type("element_dtype") };
            _execute.record_gradient("TensorListPushBackBatch", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor tensor_list_push_back_batch_eager_fallback(Tensor input_handles, Tensor tensor, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input_handles, tensor };
        object[] _attrs = new object[] { "element_dtype", tensor.dtype };
        var _result = _execute.execute("TensorListPushBackBatch", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TensorListPushBackBatch", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// List of the given size with empty elements.
    /// </summary>
    /// <remarks>
    /// 
    /// element_shape: the shape of the future elements of the list
    /// num_elements: the number of elements to reserve
    /// handle: the output list
    /// element_dtype: the desired type of elements in the list.
    /// 
    /// </remarks>
    /// <param name="element_shape"></param>
    /// <param name="num_elements"></param>
    /// <param name="element_dtype"></param>
    /// <returns></returns>
    public static Tensor tensor_list_reserve(Tensor element_shape, Tensor num_elements, TF_DataType element_dtype, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TensorListReserve", name) { args = new object[] { element_shape, num_elements }, attrs = new Dictionary<string, object>() { ["element_dtype"] = element_dtype } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return tensor_list_reserve_eager_fallback(element_shape, num_elements, element_dtype: element_dtype, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["element_shape"] = element_shape;
        keywords["num_elements"] = num_elements;
        keywords["element_dtype"] = element_dtype;
        var _op = tf.OpDefLib._apply_op_helper("TensorListReserve", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "element_dtype", _op._get_attr_type("element_dtype"), "shape_type", _op._get_attr_type("shape_type") };
            _execute.record_gradient("TensorListReserve", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor tensor_list_reserve_eager_fallback(Tensor element_shape, Tensor num_elements, TF_DataType element_dtype, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { element_shape, num_elements };
        object[] _attrs = new object[] { "element_dtype", element_dtype, "shape_type", element_shape.dtype };
        var _result = _execute.execute("TensorListReserve", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TensorListReserve", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Resizes the list.
    /// </summary>
    /// <remarks>
    /// 
    /// 
    /// input_handle: the input list
    /// size: size of the output list
    /// 
    /// 
    /// </remarks>
    /// <param name="input_handle"></param>
    /// <param name="size"></param>
    /// <returns></returns>
    public static Tensor tensor_list_resize(Tensor input_handle, Tensor size, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TensorListResize", name) { args = new object[] { input_handle, size }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return tensor_list_resize_eager_fallback(input_handle, size, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input_handle"] = input_handle;
        keywords["size"] = size;
        var _op = tf.OpDefLib._apply_op_helper("TensorListResize", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { };
            _execute.record_gradient("TensorListResize", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor tensor_list_resize_eager_fallback(Tensor input_handle, Tensor size, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input_handle, size };
        object[] _attrs = new object[] { };
        var _result = _execute.execute("TensorListResize", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TensorListResize", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Creates a TensorList by indexing into a Tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Each member of the TensorList corresponds to one row of the input tensor,
    /// specified by the given index (see `tf.gather`).
    /// 
    /// tensor: The input tensor.
    /// indices: The indices used to index into the list.
    /// element_shape: The shape of the elements in the list (can be less specified than
    ///   the shape of the tensor).
    /// output_handle: The TensorList.
    /// 
    /// </remarks>
    /// <param name="tensor"></param>
    /// <param name="indices"></param>
    /// <param name="element_shape"></param>
    /// <returns></returns>
    public static Tensor tensor_list_scatter(Tensor tensor, Tensor indices, Tensor element_shape, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TensorListScatter", name) { args = new object[] { tensor, indices, element_shape }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return tensor_list_scatter_eager_fallback(tensor, indices, element_shape, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["tensor"] = tensor;
        keywords["indices"] = indices;
        keywords["element_shape"] = element_shape;
        var _op = tf.OpDefLib._apply_op_helper("TensorListScatter", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "element_dtype", _op._get_attr_type("element_dtype"), "shape_type", _op._get_attr_type("shape_type") };
            _execute.record_gradient("TensorListScatter", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor tensor_list_scatter_eager_fallback(Tensor tensor, Tensor indices, Tensor element_shape, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { tensor, indices, element_shape };
        object[] _attrs = new object[] { "element_dtype", tensor.dtype, "shape_type", element_shape.dtype };
        var _result = _execute.execute("TensorListScatter", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TensorListScatter", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Scatters tensor at indices in an input list.
    /// </summary>
    /// <remarks>
    /// 
    /// Each member of the TensorList corresponds to one row of the input tensor,
    /// specified by the given index (see `tf.gather`).
    /// 
    /// input_handle: The list to scatter into.
    /// tensor: The input tensor.
    /// indices: The indices used to index into the list.
    /// output_handle: The TensorList.
    /// 
    /// </remarks>
    /// <param name="input_handle"></param>
    /// <param name="tensor"></param>
    /// <param name="indices"></param>
    /// <returns></returns>
    public static Tensor tensor_list_scatter_into_existing_list(Tensor input_handle, Tensor tensor, Tensor indices, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TensorListScatterIntoExistingList", name) { args = new object[] { input_handle, tensor, indices }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return tensor_list_scatter_into_existing_list_eager_fallback(input_handle, tensor, indices, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input_handle"] = input_handle;
        keywords["tensor"] = tensor;
        keywords["indices"] = indices;
        var _op = tf.OpDefLib._apply_op_helper("TensorListScatterIntoExistingList", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "element_dtype", _op._get_attr_type("element_dtype") };
            _execute.record_gradient("TensorListScatterIntoExistingList", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor tensor_list_scatter_into_existing_list_eager_fallback(Tensor input_handle, Tensor tensor, Tensor indices, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input_handle, tensor, indices };
        object[] _attrs = new object[] { "element_dtype", tensor.dtype };
        var _result = _execute.execute("TensorListScatterIntoExistingList", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TensorListScatterIntoExistingList", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Creates a TensorList by indexing into a Tensor.
    /// </summary>
    /// <remarks>
    /// 
    /// Each member of the TensorList corresponds to one row of the input tensor,
    /// specified by the given index (see `tf.gather`).
    /// 
    /// tensor: The input tensor.
    /// indices: The indices used to index into the list.
    /// element_shape: The shape of the elements in the list (can be less specified than
    ///   the shape of the tensor).
    /// num_elements: The size of the output list. Must be large enough to accommodate
    ///   the largest index in indices. If -1, the list is just large enough to include
    ///   the largest index in indices.
    /// output_handle: The TensorList.
    /// 
    /// </remarks>
    /// <param name="tensor"></param>
    /// <param name="indices"></param>
    /// <param name="element_shape"></param>
    /// <param name="num_elements"></param>
    /// <returns></returns>
    public static Tensor tensor_list_scatter_v2(Tensor tensor, Tensor indices, Tensor element_shape, Tensor num_elements, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TensorListScatterV2", name) { args = new object[] { tensor, indices, element_shape, num_elements }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return tensor_list_scatter_v2_eager_fallback(tensor, indices, element_shape, num_elements, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["tensor"] = tensor;
        keywords["indices"] = indices;
        keywords["element_shape"] = element_shape;
        keywords["num_elements"] = num_elements;
        var _op = tf.OpDefLib._apply_op_helper("TensorListScatterV2", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "element_dtype", _op._get_attr_type("element_dtype"), "shape_type", _op._get_attr_type("shape_type") };
            _execute.record_gradient("TensorListScatterV2", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor tensor_list_scatter_v2_eager_fallback(Tensor tensor, Tensor indices, Tensor element_shape, Tensor num_elements, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { tensor, indices, element_shape, num_elements };
        object[] _attrs = new object[] { "element_dtype", tensor.dtype, "shape_type", element_shape.dtype };
        var _result = _execute.execute("TensorListScatterV2", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TensorListScatterV2", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="input_handle"></param>
    /// <param name="index"></param>
    /// <param name="item"></param>
    /// <returns></returns>
    public static Tensor tensor_list_set_item(Tensor input_handle, Tensor index, Tensor item, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TensorListSetItem", name) { args = new object[] { input_handle, index, item }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return tensor_list_set_item_eager_fallback(input_handle, index, item, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input_handle"] = input_handle;
        keywords["index"] = index;
        keywords["item"] = item;
        var _op = tf.OpDefLib._apply_op_helper("TensorListSetItem", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "element_dtype", _op._get_attr_type("element_dtype") };
            _execute.record_gradient("TensorListSetItem", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor tensor_list_set_item_eager_fallback(Tensor input_handle, Tensor index, Tensor item, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input_handle, index, item };
        object[] _attrs = new object[] { "element_dtype", item.dtype };
        var _result = _execute.execute("TensorListSetItem", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TensorListSetItem", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Splits a tensor into a list.
    /// </summary>
    /// <remarks>
    /// 
    /// list[i] corresponds to lengths[i] tensors from the input tensor.
    /// The tensor must have rank at least 1 and contain exactly sum(lengths) elements.
    /// 
    /// tensor: The input tensor.
    /// element_shape: A shape compatible with that of elements in the tensor.
    /// lengths: Vector of sizes of the 0th dimension of tensors in the list.
    /// output_handle: The list.
    /// 
    /// </remarks>
    /// <param name="tensor"></param>
    /// <param name="element_shape"></param>
    /// <param name="lengths"></param>
    /// <returns></returns>
    public static Tensor tensor_list_split(Tensor tensor, Tensor element_shape, Tensor lengths, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TensorListSplit", name) { args = new object[] { tensor, element_shape, lengths }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return tensor_list_split_eager_fallback(tensor, element_shape, lengths, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["tensor"] = tensor;
        keywords["element_shape"] = element_shape;
        keywords["lengths"] = lengths;
        var _op = tf.OpDefLib._apply_op_helper("TensorListSplit", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "element_dtype", _op._get_attr_type("element_dtype"), "shape_type", _op._get_attr_type("shape_type") };
            _execute.record_gradient("TensorListSplit", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor tensor_list_split_eager_fallback(Tensor tensor, Tensor element_shape, Tensor lengths, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { tensor, element_shape, lengths };
        object[] _attrs = new object[] { "element_dtype", tensor.dtype, "shape_type", element_shape.dtype };
        var _result = _execute.execute("TensorListSplit", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TensorListSplit", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Stacks all tensors in the list.
    /// </summary>
    /// <remarks>
    /// 
    /// Requires that all tensors have the same shape.
    /// 
    /// input_handle: the input list
    /// tensor: the gathered result
    /// num_elements: optional. If not -1, the number of elements in the list.
    /// 
    /// 
    /// </remarks>
    /// <param name="input_handle"></param>
    /// <param name="element_shape"></param>
    /// <param name="element_dtype"></param>
    /// <param name="num_elements"></param>
    /// <returns></returns>
    public static Tensor tensor_list_stack(Tensor input_handle, Tensor element_shape, TF_DataType element_dtype, int num_elements = -1, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "TensorListStack", name) { args = new object[] { input_handle, element_shape }, attrs = new Dictionary<string, object>() { ["element_dtype"] = element_dtype, ["num_elements"] = num_elements } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return tensor_list_stack_eager_fallback(input_handle, element_shape, element_dtype: element_dtype, num_elements: num_elements, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input_handle"] = input_handle;
        keywords["element_shape"] = element_shape;
        keywords["element_dtype"] = element_dtype;
        keywords["num_elements"] = num_elements;
        var _op = tf.OpDefLib._apply_op_helper("TensorListStack", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "element_dtype", _op._get_attr_type("element_dtype"), "num_elements", _op._get_attr_int("num_elements") };
            _execute.record_gradient("TensorListStack", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor tensor_list_stack_eager_fallback(Tensor input_handle, Tensor element_shape, TF_DataType element_dtype, int num_elements, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input_handle, element_shape };
        object[] _attrs = new object[] { "element_dtype", element_dtype, "num_elements", num_elements };
        var _result = _execute.execute("TensorListStack", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("TensorListStack", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
}
