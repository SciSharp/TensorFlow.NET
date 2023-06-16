/*Wrappers around TensorFlow ops. This file is MACHINE GENERATED! Do not edit.*/

using Tensorflow.Eager;
using Tensorflow.Contexts;
using Tensorflow.Exceptions;
using static Tensorflow.Binding;

namespace Tensorflow;

public static class gen_functional_ops
{
    /// <summary>
    /// An n-way switch statement which calls a single branch function.
    /// </summary>
    /// <remarks>
    /// 
    ///     An n-way switch statement, implementing the following:
    ///     ```
    ///     switch (branch_index) {
    ///       case 0:
    ///         output = branches[0](input);
    ///         break;
    ///       case 1:
    ///         output = branches[1](input);
    ///         break;
    ///       ...
    ///       case [[nbranches-1]]:
    ///       default:
    ///         output = branches[nbranches-1](input);
    ///         break;
    ///     }
    ///     ```
    /// 
    /// </remarks>
    /// <param name="branch_index"></param>
    /// <param name="input"></param>
    /// <param name="Tout">
    /// A list of output types.
    /// </param>
    /// <param name="branches">
    /// 
    ///       A list of functions each of which takes 'inputs' and returns a list of
    ///       tensors, whose types are the same as what every other branch returns.
    /// 
    /// </param>
    /// <param name="output_shapes"></param>
    /// <returns></returns>
    public static Tensor[] _case(Tensor branch_index, Tensors input, TF_DataType[] Tout, object[] branches, Shape[] output_shapes, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "Case", name) { args = new object[] { branch_index, input }, attrs = new Dictionary<string, object>() { ["Tout"] = Tout, ["branches"] = branches, ["output_shapes"] = output_shapes } });
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
                return case_eager_fallback(branch_index, input, Tout: Tout, branches: branches, output_shapes: output_shapes, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["branch_index"] = branch_index;
        keywords["input"] = input;
        keywords["Tout"] = Tout;
        keywords["branches"] = branches;
        keywords["output_shapes"] = output_shapes;
        var _op = tf.OpDefLib._apply_op_helper("Case", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tin", _op.get_attr("Tin"), "Tout", _op.get_attr("Tout"), "branches", _op.get_attr("branches"), "output_shapes", _op.get_attr("output_shapes") };
            _execute.record_gradient("Case", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] case_eager_fallback(Tensor branch_index, Tensor input, TF_DataType[] Tout, object[] branches, Shape[] output_shapes, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { branch_index, input };
        object[] _attrs = new object[] { "branches", branches, "output_shapes", output_shapes };
        var _result = _execute.execute("Case", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("Case", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Return the index of device the op runs.
    /// </summary>
    /// <remarks>
    /// 
    /// Given a list of device names, this operation returns the index of the device
    /// this op runs. The length of the list is returned in two cases:
    /// (1) Device does not exist in the given device list.
    /// (2) It is in XLA compilation.
    /// 
    /// </remarks>
    /// <param name="device_names"></param>
    /// <returns></returns>
    public static Tensor device_index(string[] device_names, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "DeviceIndex", name) { args = new object[] { }, attrs = new Dictionary<string, object>() { ["device_names"] = device_names } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return device_index_eager_fallback(device_names: device_names, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["device_names"] = device_names;
        var _op = tf.OpDefLib._apply_op_helper("DeviceIndex", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "device_names", _op.get_attr("device_names") };
            _execute.record_gradient("DeviceIndex", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor device_index_eager_fallback(string[] device_names, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { };
        object[] _attrs = new object[] { "device_names", device_names };
        var _result = _execute.execute("DeviceIndex", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("DeviceIndex", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// ~~%~~  This op is used as a placeholder in If branch functions. It doesn't provide a~~%~~  valid output when run, so must either be removed (e.g. replaced with a~~%~~  function input) or guaranteed not to be used (e.g. if mirroring an~~%~~  intermediate output needed for the gradient computation of the other branch).~~%~~
    /// </summary>
    /// <param name="dtype">
    /// The type of the output.
    /// </param>
    /// <param name="shape">
    /// 
    ///     The purported shape of the output. This is only used for shape inference;
    ///     the output will not necessarily have this shape. Can be a partial shape.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor fake_param(TF_DataType dtype, Shape shape, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "FakeParam", name) { args = new object[] { }, attrs = new Dictionary<string, object>() { ["dtype"] = dtype, ["shape"] = shape } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return fake_param_eager_fallback(dtype: dtype, shape: shape, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["dtype"] = dtype;
        keywords["shape"] = shape;
        var _op = tf.OpDefLib._apply_op_helper("FakeParam", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "dtype", _op._get_attr_type("dtype"), "shape", _op.get_attr("shape") };
            _execute.record_gradient("FakeParam", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor fake_param_eager_fallback(TF_DataType dtype, Shape shape, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { };
        object[] _attrs = new object[] { "dtype", dtype, "shape", shape };
        var _result = _execute.execute("FakeParam", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("FakeParam", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// Applies a for loop.
    /// </summary>
    /// <remarks>
    /// 
    ///   ```python
    ///    output = input;
    ///    for i in range(start, limit, delta)
    ///      output = body(i, output);
    ///   ```
    /// 
    /// </remarks>
    /// <param name="start"></param>
    /// <param name="limit"></param>
    /// <param name="delta"></param>
    /// <param name="input"></param>
    /// <param name="body">
    /// 
    ///     A function that takes a list of tensors (int32, T) and returns another
    ///     list of tensors (T).
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] _for(Tensor start, Tensor limit, Tensor delta, Tensors input, object body, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "For", name) { args = new object[] { start, limit, delta, input }, attrs = new Dictionary<string, object>() { ["body"] = body } });
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
                return for_eager_fallback(start, limit, delta, input, body: body, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["start"] = start;
        keywords["limit"] = limit;
        keywords["delta"] = delta;
        keywords["input"] = input;
        keywords["body"] = body;
        var _op = tf.OpDefLib._apply_op_helper("For", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op.get_attr("T"), "body", _op.get_attr("body") };
            _execute.record_gradient("For", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] for_eager_fallback(Tensor start, Tensor limit, Tensor delta, Tensor input, object body, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { start, limit, delta, input };
        object[] _attrs = new object[] { "body", body };
        var _result = _execute.execute("For", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("For", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// output = cond ? then_branch(input) : else_branch(input)
    /// </summary>
    /// <param name="cond"></param>
    /// <param name="input"></param>
    /// <param name="Tout">
    /// A list of output types.
    /// </param>
    /// <param name="then_branch">
    /// 
    ///       A function that takes 'inputs' and returns a list of tensors, whose
    ///       types are the same as what else_branch returns.
    /// 
    /// </param>
    /// <param name="else_branch">
    /// 
    ///     A function that takes 'inputs' and returns a list of tensors, whose
    ///     types are the same as what then_branch returns.
    /// 
    /// </param>
    /// <param name="output_shapes"></param>
    /// <returns></returns>
    public static Tensor[] _if(Tensor cond, Tensors input, TF_DataType[] Tout, object then_branch, object else_branch, Shape[] output_shapes, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "If", name) { args = new object[] { cond, input }, attrs = new Dictionary<string, object>() { ["Tout"] = Tout, ["then_branch"] = then_branch, ["else_branch"] = else_branch, ["output_shapes"] = output_shapes } });
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
                return if_eager_fallback(cond, input, Tout: Tout, then_branch: then_branch, else_branch: else_branch, output_shapes: output_shapes, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["cond"] = cond;
        keywords["input"] = input;
        keywords["Tout"] = Tout;
        keywords["then_branch"] = then_branch;
        keywords["else_branch"] = else_branch;
        keywords["output_shapes"] = output_shapes;
        var _op = tf.OpDefLib._apply_op_helper("If", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tcond", _op._get_attr_type("Tcond"), "Tin", _op.get_attr("Tin"), "Tout", _op.get_attr("Tout"), "then_branch", _op.get_attr("then_branch"), "else_branch", _op.get_attr("else_branch"), "output_shapes", _op.get_attr("output_shapes") };
            _execute.record_gradient("If", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] if_eager_fallback(Tensor cond, Tensor input, TF_DataType[] Tout, object then_branch, object else_branch, Shape[] output_shapes, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { cond, input };
        object[] _attrs = new object[] { "Tcond", cond.dtype, "then_branch", then_branch, "else_branch", else_branch, "output_shapes", output_shapes };
        var _result = _execute.execute("If", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("If", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// returns `f(inputs)`, where `f`'s body is placed and partitioned.
    /// </summary>
    /// <remarks>
    /// 
    /// Asynchronously executes a function, potentially across multiple devices but
    /// within a single process. The kernel places and partitions a given function's
    /// underlying graph, and executes each of the partitioned subgraphs as a function.
    /// 
    /// </remarks>
    /// <param name="args"></param>
    /// <param name="Tout">
    /// A list of output types.
    /// </param>
    /// <param name="f">
    /// 
    ///       A function that takes 'args', a list of tensors, and returns 'output',
    ///       another list of tensors. Input and output types are specified by 'Tin'
    ///       and 'Tout'. The function body of f will be placed and partitioned across
    ///       devices, setting this op apart from the regular Call op.
    /// 
    /// </param>
    /// <param name="config"></param>
    /// <param name="config_proto"></param>
    /// <param name="executor_type"></param>
    /// <returns></returns>
    public static Tensor[] partitioned_call(Tensors args, TF_DataType[] Tout, object f, string config = "", string config_proto = "", string executor_type = "", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "PartitionedCall", name) { args = new object[] { args }, attrs = new Dictionary<string, object>() { ["Tout"] = Tout, ["f"] = f, ["config"] = config, ["config_proto"] = config_proto, ["executor_type"] = executor_type } });
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
                return partitioned_call_eager_fallback(args, Tout: Tout, f: f, config: config, config_proto: config_proto, executor_type: executor_type, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        if (config is null)
        {
            config = "";
        }
        if (config_proto is null)
        {
            config_proto = "";
        }
        if (executor_type is null)
        {
            executor_type = "";
        }
        Dictionary<string, object> keywords = new();
        keywords["args"] = args;
        keywords["Tout"] = Tout;
        keywords["f"] = f;
        keywords["config"] = config;
        keywords["config_proto"] = config_proto;
        keywords["executor_type"] = executor_type;
        var _op = tf.OpDefLib._apply_op_helper("PartitionedCall", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tin", _op.get_attr("Tin"), "Tout", _op.get_attr("Tout"), "f", _op.get_attr("f"), "config", _op.get_attr("config"), "config_proto", _op.get_attr("config_proto"), "executor_type", _op.get_attr("executor_type") };
            _execute.record_gradient("PartitionedCall", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] partitioned_call_eager_fallback(Tensor args, TF_DataType[] Tout, object f, string config, string config_proto, string executor_type, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { args };
        object[] _attrs = new object[] { "f", f, "config", config, "config_proto", config_proto, "executor_type", executor_type };
        var _result = _execute.execute("PartitionedCall", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("PartitionedCall", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Runs function `f` on a remote device indicated by `target`.
    /// </summary>
    /// <param name="target"></param>
    /// <param name="args"></param>
    /// <param name="Tout">
    /// 
    /// The type list for the return values.
    /// 
    /// </param>
    /// <param name="f">
    /// 
    /// The function to run remotely.
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] remote_call(Tensor target, Tensors args, TF_DataType[] Tout, object f, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "RemoteCall", name) { args = new object[] { target, args }, attrs = new Dictionary<string, object>() { ["Tout"] = Tout, ["f"] = f } });
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
                return remote_call_eager_fallback(target, args, Tout: Tout, f: f, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["target"] = target;
        keywords["args"] = args;
        keywords["Tout"] = Tout;
        keywords["f"] = f;
        var _op = tf.OpDefLib._apply_op_helper("RemoteCall", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tin", _op.get_attr("Tin"), "Tout", _op.get_attr("Tout"), "f", _op.get_attr("f") };
            _execute.record_gradient("RemoteCall", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] remote_call_eager_fallback(Tensor target, Tensor args, TF_DataType[] Tout, object f, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { target, args };
        object[] _attrs = new object[] { "f", f };
        var _result = _execute.execute("RemoteCall", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("RemoteCall", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// returns `f(inputs)`, where `f`'s body is placed and partitioned.
    /// </summary>
    /// <param name="args"></param>
    /// <param name="Tout">
    /// A list of output types.
    /// </param>
    /// <param name="f">
    /// 
    ///       A function that takes 'args', a list of tensors, and returns 'output',
    ///       another list of tensors. Input and output types are specified by 'Tin'
    ///       and 'Tout'. The function body of f will be placed and partitioned across
    ///       devices, setting this op apart from the regular Call op. This op is
    ///       stateful.
    /// 
    /// </param>
    /// <param name="config"></param>
    /// <param name="config_proto"></param>
    /// <param name="executor_type"></param>
    /// <returns></returns>
    public static Tensor[] stateful_partitioned_call(Tensors args, TF_DataType[] Tout, object f, string config = "", string config_proto = "", string executor_type = "", string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "StatefulPartitionedCall", name) { args = new object[] { args }, attrs = new Dictionary<string, object>() { ["Tout"] = Tout, ["f"] = f, ["config"] = config, ["config_proto"] = config_proto, ["executor_type"] = executor_type } });
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
                return stateful_partitioned_call_eager_fallback(args, Tout: Tout, f: f, config: config, config_proto: config_proto, executor_type: executor_type, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        if (config is null)
        {
            config = "";
        }
        if (config_proto is null)
        {
            config_proto = "";
        }
        if (executor_type is null)
        {
            executor_type = "";
        }
        Dictionary<string, object> keywords = new();
        keywords["args"] = args;
        keywords["Tout"] = Tout;
        keywords["f"] = f;
        keywords["config"] = config;
        keywords["config_proto"] = config_proto;
        keywords["executor_type"] = executor_type;
        var _op = tf.OpDefLib._apply_op_helper("StatefulPartitionedCall", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tin", _op.get_attr("Tin"), "Tout", _op.get_attr("Tout"), "f", _op.get_attr("f"), "config", _op.get_attr("config"), "config_proto", _op.get_attr("config_proto"), "executor_type", _op.get_attr("executor_type") };
            _execute.record_gradient("StatefulPartitionedCall", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] stateful_partitioned_call_eager_fallback(Tensor args, TF_DataType[] Tout, object f, string config, string config_proto, string executor_type, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { args };
        object[] _attrs = new object[] { "f", f, "config", config, "config_proto", config_proto, "executor_type", executor_type };
        var _result = _execute.execute("StatefulPartitionedCall", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("StatefulPartitionedCall", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// An n-way switch statement which calls a single branch function.
    /// </summary>
    /// <remarks>
    /// 
    ///     An n-way switch statement, implementing the following:
    ///     ```
    ///     switch (branch_index) {
    ///       case 0:
    ///         output = branches[0](input);
    ///         break;
    ///       case 1:
    ///         output = branches[1](input);
    ///         break;
    ///       ...
    ///       case [[nbranches-1]]:
    ///       default:
    ///         output = branches[nbranches-1](input);
    ///         break;
    ///     }
    ///     ```
    /// 
    ///     This should only be used when the none of branches has stateful ops.
    /// 
    /// </remarks>
    /// <param name="branch_index"></param>
    /// <param name="input"></param>
    /// <param name="Tout">
    /// A list of output types.
    /// </param>
    /// <param name="branches">
    /// 
    ///       A list of functions each of which takes 'inputs' and returns a list of
    ///       tensors, whose types are the same as what every other branch returns.
    /// 
    /// </param>
    /// <param name="output_shapes"></param>
    /// <returns></returns>
    public static Tensor[] stateless_case(Tensor branch_index, Tensors input, TF_DataType[] Tout, object[] branches, Shape[] output_shapes, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "StatelessCase", name) { args = new object[] { branch_index, input }, attrs = new Dictionary<string, object>() { ["Tout"] = Tout, ["branches"] = branches, ["output_shapes"] = output_shapes } });
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
                return stateless_case_eager_fallback(branch_index, input, Tout: Tout, branches: branches, output_shapes: output_shapes, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["branch_index"] = branch_index;
        keywords["input"] = input;
        keywords["Tout"] = Tout;
        keywords["branches"] = branches;
        keywords["output_shapes"] = output_shapes;
        var _op = tf.OpDefLib._apply_op_helper("StatelessCase", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tin", _op.get_attr("Tin"), "Tout", _op.get_attr("Tout"), "branches", _op.get_attr("branches"), "output_shapes", _op.get_attr("output_shapes") };
            _execute.record_gradient("StatelessCase", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] stateless_case_eager_fallback(Tensor branch_index, Tensor input, TF_DataType[] Tout, object[] branches, Shape[] output_shapes, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { branch_index, input };
        object[] _attrs = new object[] { "branches", branches, "output_shapes", output_shapes };
        var _result = _execute.execute("StatelessCase", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("StatelessCase", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// output = cond ? then_branch(input) : else_branch(input)
    /// </summary>
    /// <param name="cond"></param>
    /// <param name="input"></param>
    /// <param name="Tout">
    /// A list of output types.
    /// </param>
    /// <param name="then_branch">
    /// 
    ///       A function that takes 'inputs' and returns a list of tensors, whose
    ///       types are the same as what else_branch returns.
    /// 
    /// </param>
    /// <param name="else_branch">
    /// 
    ///     A function that takes 'inputs' and returns a list of tensors, whose
    ///     types are the same as what then_branch returns.
    /// 
    /// </param>
    /// <param name="output_shapes"></param>
    /// <returns></returns>
    public static Tensor[] stateless_if(Tensor cond, Tensors input, TF_DataType[] Tout, object then_branch, object else_branch, Shape[] output_shapes, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "StatelessIf", name) { args = new object[] { cond, input }, attrs = new Dictionary<string, object>() { ["Tout"] = Tout, ["then_branch"] = then_branch, ["else_branch"] = else_branch, ["output_shapes"] = output_shapes } });
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
                return stateless_if_eager_fallback(cond, input, Tout: Tout, then_branch: then_branch, else_branch: else_branch, output_shapes: output_shapes, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["cond"] = cond;
        keywords["input"] = input;
        keywords["Tout"] = Tout;
        keywords["then_branch"] = then_branch;
        keywords["else_branch"] = else_branch;
        keywords["output_shapes"] = output_shapes;
        var _op = tf.OpDefLib._apply_op_helper("StatelessIf", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tcond", _op._get_attr_type("Tcond"), "Tin", _op.get_attr("Tin"), "Tout", _op.get_attr("Tout"), "then_branch", _op.get_attr("then_branch"), "else_branch", _op.get_attr("else_branch"), "output_shapes", _op.get_attr("output_shapes") };
            _execute.record_gradient("StatelessIf", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] stateless_if_eager_fallback(Tensor cond, Tensor input, TF_DataType[] Tout, object then_branch, object else_branch, Shape[] output_shapes, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { cond, input };
        object[] _attrs = new object[] { "Tcond", cond.dtype, "then_branch", then_branch, "else_branch", else_branch, "output_shapes", output_shapes };
        var _result = _execute.execute("StatelessIf", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("StatelessIf", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// output = input; While (Cond(output)) { output = Body(output) }
    /// </summary>
    /// <param name="input"></param>
    /// <param name="cond">
    /// 
    ///       A function takes 'input' and returns a tensor.  If the tensor is
    ///       a scalar of non-boolean, the scalar is converted to a boolean
    ///       according to the following rule: if the scalar is a numerical
    ///       value, non-zero means True and zero means False; if the scalar is
    ///       a string, non-empty means True and empty means False. If the
    ///       tensor is not a scalar, non-emptiness means True and False
    ///       otherwise.
    /// 
    ///       This should only be used when the while condition and body functions
    ///       do not have stateful ops.
    /// 
    /// </param>
    /// <param name="body">
    /// 
    ///       A function that takes a list of tensors and returns another
    ///       list of tensors. Both lists have the same types as specified
    ///       by T.
    /// 
    /// </param>
    /// <param name="output_shapes"></param>
    /// <param name="parallel_iterations"></param>
    /// <returns></returns>
    public static Tensor[] stateless_while(Tensors input, object cond, object body, Shape[] output_shapes, int parallel_iterations = 10, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "StatelessWhile", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { ["cond"] = cond, ["body"] = body, ["output_shapes"] = output_shapes, ["parallel_iterations"] = parallel_iterations } });
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
                return stateless_while_eager_fallback(input, cond: cond, body: body, output_shapes: output_shapes, parallel_iterations: parallel_iterations, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["cond"] = cond;
        keywords["body"] = body;
        keywords["output_shapes"] = output_shapes;
        keywords["parallel_iterations"] = parallel_iterations;
        var _op = tf.OpDefLib._apply_op_helper("StatelessWhile", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op.get_attr("T"), "cond", _op.get_attr("cond"), "body", _op.get_attr("body"), "output_shapes", _op.get_attr("output_shapes"), "parallel_iterations", _op._get_attr_int("parallel_iterations") };
            _execute.record_gradient("StatelessWhile", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] stateless_while_eager_fallback(Tensor input, object cond, object body, Shape[] output_shapes, int parallel_iterations, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "cond", cond, "body", body, "output_shapes", output_shapes, "parallel_iterations", parallel_iterations };
        var _result = _execute.execute("StatelessWhile", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("StatelessWhile", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Computes the gradient function for function f via backpropagation.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="Tout">
    /// 
    /// the type list for the input list.
    /// 
    /// </param>
    /// <param name="f">
    /// 
    /// The function we want to compute the gradient for.
    /// 
    /// The function 'f' must be a numerical function which takes N inputs and
    /// produces M outputs. Its gradient function 'g', which is computed by
    /// this SymbolicGradient op is a function taking N + M inputs and
    /// produces N outputs.
    /// 
    /// I.e. if we have
    ///    (y1, y2, ..., y_M) = f(x1, x2, ..., x_N),
    /// then, g is
    ///    (dL/dx1, dL/dx2, ..., dL/dx_N) = g(x1, x2, ..., x_N,
    ///                                      dL/dy1, dL/dy2, ..., dL/dy_M),
    /// 
    /// where L is a scalar-value function of (x1, x2, ..., xN) (e.g., the
    /// loss function). dL/dx_i is the partial derivative of L with respect
    /// to x_i.
    /// 
    /// (Needs some math expert to say the comment above better.)
    /// 
    /// </param>
    /// <returns></returns>
    public static Tensor[] symbolic_gradient(Tensors input, TF_DataType[] Tout, object f, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "SymbolicGradient", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { ["Tout"] = Tout, ["f"] = f } });
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
                return symbolic_gradient_eager_fallback(input, Tout: Tout, f: f, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["Tout"] = Tout;
        keywords["f"] = f;
        var _op = tf.OpDefLib._apply_op_helper("SymbolicGradient", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "Tin", _op.get_attr("Tin"), "Tout", _op.get_attr("Tout"), "f", _op.get_attr("f") };
            _execute.record_gradient("SymbolicGradient", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] symbolic_gradient_eager_fallback(Tensor input, TF_DataType[] Tout, object f, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "f", f };
        var _result = _execute.execute("SymbolicGradient", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("SymbolicGradient", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
    /// <summary>
    /// Converts a tensor to a scalar predicate.
    /// </summary>
    /// <remarks>
    /// 
    /// Converts a tensor to a scalar predicate with the following rules:
    /// 
    /// - For 0D tensors, truthiness is determined by comparing against a "zero"
    ///   value. For numerical types it is the obvious zero. For strings it is the
    ///   empty string.
    /// 
    /// - For >0D tensors, truthiness is determined by looking at the number of
    ///   elements. If has zero elements, then the result is false. Otherwise the
    ///   result is true.
    /// 
    /// This matches the behavior of If and While for determining if a tensor counts
    /// as true/false for a branch condition.
    /// 
    /// </remarks>
    /// <param name="input"></param>
    /// <returns></returns>
    public static Tensor to_bool(Tensor input, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "ToBool", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { } });
                return _fast_path_result[0];
            }
            catch (NotOkStatusException ex)
            {
                throw ex;
            }
            catch (Exception)
            {
            }
            try
            {
                return to_bool_eager_fallback(input, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        var _op = tf.OpDefLib._apply_op_helper("ToBool", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op._get_attr_type("T") };
            _execute.record_gradient("ToBool", _op.inputs, _attrs, _result);
        }
        return _result[0];
    }

    public static Tensor to_bool_eager_fallback(Tensor input, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "T", input.dtype };
        var _result = _execute.execute("ToBool", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("ToBool", _inputs_flat, _attrs, _result);
        }
        return _result[0];
    }
    /// <summary>
    /// output = input; While (Cond(output)) { output = Body(output) }
    /// </summary>
    /// <param name="input"></param>
    /// <param name="cond">
    /// 
    ///       A function takes 'input' and returns a tensor.  If the tensor is
    ///       a scalar of non-boolean, the scalar is converted to a boolean
    ///       according to the following rule: if the scalar is a numerical
    ///       value, non-zero means True and zero means False; if the scalar is
    ///       a string, non-empty means True and empty means False. If the
    ///       tensor is not a scalar, non-emptiness means True and False
    ///       otherwise.
    /// 
    /// </param>
    /// <param name="body">
    /// 
    ///       A function that takes a list of tensors and returns another
    ///       list of tensors. Both lists have the same types as specified
    ///       by T.
    /// 
    /// </param>
    /// <param name="output_shapes"></param>
    /// <param name="parallel_iterations"></param>
    /// <returns></returns>
    public static Tensor[] _while(Tensors input, object cond, object body, Shape[] output_shapes, int parallel_iterations = 10, string? name = null)
    {
        var _ctx = tf.Context;
        if (_ctx.executing_eagerly())
        {
            try
            {
                var _fast_path_result = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo(_ctx, "While", name) { args = new object[] { input }, attrs = new Dictionary<string, object>() { ["cond"] = cond, ["body"] = body, ["output_shapes"] = output_shapes, ["parallel_iterations"] = parallel_iterations } });
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
                return while_eager_fallback(input, cond: cond, body: body, output_shapes: output_shapes, parallel_iterations: parallel_iterations, name: name, ctx: _ctx);
            }
            catch (Exception)
            {
            }
        }
        Dictionary<string, object> keywords = new();
        keywords["input"] = input;
        keywords["cond"] = cond;
        keywords["body"] = body;
        keywords["output_shapes"] = output_shapes;
        keywords["parallel_iterations"] = parallel_iterations;
        var _op = tf.OpDefLib._apply_op_helper("While", name, keywords);
        var _result = _op.outputs;
        if (_execute.must_record_gradient())
        {
            object[] _attrs = new object[] { "T", _op.get_attr("T"), "cond", _op.get_attr("cond"), "body", _op.get_attr("body"), "output_shapes", _op.get_attr("output_shapes"), "parallel_iterations", _op._get_attr_int("parallel_iterations") };
            _execute.record_gradient("While", _op.inputs, _attrs, _result);
        }
        return _result;
    }

    public static Tensor[] while_eager_fallback(Tensor input, object cond, object body, Shape[] output_shapes, int parallel_iterations, string name, Context ctx)
    {
        Tensor[] _inputs_flat = new Tensor[] { input };
        object[] _attrs = new object[] { "cond", cond, "body", body, "output_shapes", output_shapes, "parallel_iterations", parallel_iterations };
        var _result = _execute.execute("While", 1, inputs: _inputs_flat, attrs: _attrs, ctx: ctx, name: name);
        if (_execute.must_record_gradient())
        {
            _execute.record_gradient("While", _inputs_flat, _attrs, _result);
        }
        return _result;
    }
}
