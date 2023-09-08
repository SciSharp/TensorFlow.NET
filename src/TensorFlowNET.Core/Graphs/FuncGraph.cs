using Google.Protobuf;
using System;
using System.Buffers;
using System.Diagnostics;
using System.Linq;
using Tensorflow.Eager;
using Tensorflow.Exceptions;
using Tensorflow.Framework;
using Tensorflow.Framework.Models;
using Tensorflow.Functions;
using Tensorflow.NumPy;
using Tensorflow.Operations;
using Tensorflow.Util;
using static Tensorflow.Binding;

namespace Tensorflow.Graphs;

/// <summary>
/// Graph representing a function body.
/// </summary>
public class FuncGraph : Graph, IDisposable
{
    internal SafeFuncGraphHandle _func_graph_handle;
    internal HashSet<Tensor> _resource_tensor_inputs;
    internal HashSet<WeakReference<IVariableV1>> _watched_variables;
    internal IEnumerable<WeakReference<IVariableV1>> _weak_variables;
    internal object[] _structured_outputs;
    internal Dictionary<long, string> _output_names;
    public string FuncName => _graph_key;

    public Tensors Inputs { get; set; } = new Tensors();
    public Tensors Outputs { get; set; } = new Tensors();
    public Tensors FlatStructuredOutputs
    {
        get
        {
            List<Tensor> res = new();
            foreach(var obj in _structured_outputs)
            {
                if(obj is Tensor tensor)
                {
                    res.Add(tensor);
                }
                else if(obj is IEnumerable<Tensor> tensors)
                {
                    res.AddRange(tensors);
                }
                else
                {
                    throw new TypeError("The structured outputs member should be tensor or tensors.");
                }
            }
            return res;
        }
    }
    public string Name { get; set; }
    public IEnumerable<IVariableV1> Variables
    {
        get
        {
            return _weak_variables.Select(v =>
            {
                if (v.TryGetTarget(out var target))
                {
                    return target;
                }
                else
                {
                    throw new AssertionError("Called a function referencing variables which have been deleted. " +
                        "This likely means that function-local variables were created and " +
                        "not referenced elsewhere in the program. This is generally a " +
                        "mistake; consider storing variables in an object attribute on first call.");
                }
            });
        }
        internal set
        {
            _weak_variables = value.Select(x => new WeakReference<IVariableV1>(x));
        }
    }
    public IEnumerable<IVariableV1> TrainableVariables => Variables.Where(v => v.Trainable);
    public Dictionary<string, AttrValue> Attrs { get; set; }

    internal Dictionary<long, (Tensor, Tensor)> _captures
        = new Dictionary<long, (Tensor, Tensor)>();

    public Tensor[] external_captures
        => _captures.Select(x => x.Value.Item1).ToArray();
    public (Tensor, Tensor)[] captures
        => _captures.Values.Select(x => x).ToArray();

    public Tensor[] internal_captures
        => _captures.Select(x => x.Value.Item2).ToArray();

    public Tensor[] captured_inputs
        => external_captures;

    /// <summary>
    /// Construct a new FuncGraph.
    /// </summary>
    public FuncGraph(string name) : base()
    {
        outer_graph = ops.get_default_graph();
        while (outer_graph.building_function)
            outer_graph = outer_graph.OuterGraph;
        _graph_key = Name = name;
        building_function = true;
        _weak_variables = new List<WeakReference<IVariableV1>>();
        _resource_tensor_inputs = new HashSet<Tensor>();
        _watched_variables = new HashSet<WeakReference<IVariableV1>>();
    }

    public FuncGraph(SafeGraphHandle handle, string name, Dictionary<string, AttrValue> attrs) : base()
    {
        outer_graph = ops.get_default_graph();
        while (outer_graph.building_function)
            outer_graph = outer_graph.OuterGraph;
        _graph_key = Name = name;
        building_function = true;
        Attrs = attrs;
        // Will to test if FuncGraph has memory leak
        // c_api.TF_DeleteGraph(_handle);
        _handle = handle;
        _weak_variables = new List<WeakReference<IVariableV1>>();
        _resource_tensor_inputs = new HashSet<Tensor>();
        _watched_variables = new HashSet<WeakReference<IVariableV1>>();
    }

    public void replace_capture(Tensor tensor, Tensor placeholder)
    {
        _captures[tensor.Id] = (tensor, placeholder);
    }

    public unsafe void ToGraph(Operation[] opers,
        Tensor[] inputs, Tensor[] outputs,
        string[] output_names)
    {
        var status = new Status();
        if (output_names is null)
        {
            output_names = new string[0];
        };

        _func_graph_handle = c_api.TF_GraphToFunction(_handle,
            _graph_key,
            false,
            opers.Length,
            opers.Select(x => (IntPtr)x).ToArray(),
            inputs.Length,
            inputs.Select(x => new TF_Output(x.op, 0)).ToArray(),
            outputs.Length,
            outputs.Select(x => new TF_Output(x.op, 0)).ToArray(),
            output_names.Length != outputs.Length ? null :  output_names,
            IntPtr.Zero,
            null,
            status);
        status.Check(true);

        SetAttrs();

        // c_api.TF_GraphCopyFunction(outer_graph, _func_graph_handle, IntPtr.Zero, status.Handle);
        // status.Check(true);

        c_api.TFE_ContextAddFunction(tf.Context, _func_graph_handle, status);
        status.Check(true);

        _graph_key = c_api.StringPiece(c_api.TF_FunctionName(_func_graph_handle));

        Inputs = inputs;
        // mark_as_return
        Outputs = outputs;// .Select(x => array_ops.identity(x)).ToArray();
    }

    public override Operation create_op(string op_type, Tensor[] inputs, TF_DataType[] dtypes, TF_DataType[] input_types = null, string name = null, Dictionary<string, AttrValue> attrs = null, OpDef op_def = null, bool compute_device = true)
    {
        foreach(var (i, inp) in enumerate(inputs))
            inputs[i] = capture(inp);

        return base.create_op(op_type, inputs, dtypes, input_types, name, attrs, op_def, compute_device);
    }

    const int _EAGER_CONST_THRESHOLD = 128;
    public Tensor capture(Tensor tensor, string name = null, Shape shape = null)
    {
        if(tensor is EagerTensor or NDArray)
        {
            if (name == null)
                name = ops.uid().ToString();

            // Small EagerTensors are captured with Const ops
            if (dtypes.is_value_dtype(tensor.dtype) 
                && (tensor.rank == 0 || tensor.size < _EAGER_CONST_THRESHOLD))
                return capture_eager_tensor(tensor, name);

            // Large EagerTensors and resources are captured with Placeholder ops
            return _capture_helper(tensor, name, shape: shape);
        }

        if(tensor.graph != this)
        {
            if (name == null)
                name = tensor.op.name;
            var inner_graph = tensor.graph;
            while(inner_graph != null && inner_graph is FuncGraph inner_func_graph)
            {
                if (inner_graph == this)
                    throw new InaccessibleTensorError($"The tensor '{tensor.name}' cannot be accessed here: it is defined" +
                        " in another function or code block. Use return values," +
                        " explicit Python locals or TensorFlow collections to access" +
                        $" it. Defined in: {tensor.graph.graph_key}; accessed from: {graph_key}.");
                inner_graph = inner_func_graph.outer_graph;
            }
            return _capture_helper(tensor, name);
        }

        return tensor;
    }

    public void watch_variable(IVariableV1 v)
    {
        if (_resource_tensor_inputs.Contains(v.Handle))
        {
            return;
        }
        _watched_variables.Add(new WeakReference<IVariableV1>(v));
        //this = this.outer_graph;
    }

    Tensor capture_eager_tensor(Tensor tensor, string name)
    {
        Tensor graph_const = null;
        if (!_captures.ContainsKey(tensor.Id))
        {
            graph_const = tf_with(ops.control_dependencies(null), ctl
                => constant_op.constant(tensor.numpy(), dtype: tensor.dtype, shape: tensor.shape, name: name));
            add_capture(tensor, graph_const);
        }
        else
        {
            graph_const = _captures[tensor.Id].Item2;
        }

        BackwardFunction _backward_function_wrapper = (output_grads, unneeded_gradients) =>
        {
            return output_grads;
        };

        tf.Runner.RecordGradient("captured_value",
            new[] { graph_const }, null,
            new[] { tensor },
            getBackwardFunction: _backward_function_wrapper
            /*getForwardFunction: forward_function*/);

        return graph_const;
    }

    Tensor _capture_helper(Tensor tensor, string name, Shape shape = null)
    {
        Tensor placeholder = null;
        if (!_captures.ContainsKey(tensor.Id))
        {
            placeholder = _create_substitute_placeholder(tensor,
                name: name,
                dtype: tensor.dtype,
                shape: shape);
            add_capture(tensor, placeholder);
        }
        else
        {
            placeholder = _captures[tensor.Id].Item2;
        }

        BackwardFunction _backward_function_wrapper = (output_grads, unneeded_gradients) =>
        {
            return output_grads;
        };

        tf.Runner.RecordGradient("captured_value",
            new[] { placeholder }, null,
            new[] { tensor },
            getBackwardFunction: _backward_function_wrapper
            /*getForwardFunction: forward_function*/);

        return placeholder;
    }

    void add_capture(Tensor tensor, Tensor placeholder)
    {
        _captures.Add(tensor.Id, (tensor, placeholder));
        Inputs.Add(placeholder);
    }

    Tensor pop_capture(Tensor tensor)
    {
        if(_captures.TryGetValue(tensor.Id, out var capture))
        {
            _captures.Remove(tensor.Id);
            return capture.Item2;
        }
        else
        {
            return null;
        }
    }

    Tensor _create_substitute_placeholder(Tensor value, 
        string name = null, 
        TF_DataType dtype = TF_DataType.DtInvalid, 
        Shape shape = null)
    {
        if (shape is null)
            shape = value.shape;
        if (dtype == TF_DataType.DtInvalid)
            dtype = value.dtype;

        var placeholder = tf_with(ops.control_dependencies(null), ctl
            => array_ops.placeholder(dtype, shape: shape, name: name));
        // custom_gradient.copy_handle_data(value, placeholder)
        return placeholder;
    }

    void SetAttrs()
    {
        if (Attrs == null)
            return;

        foreach (var (_name, attr_value) in enumerate(Attrs))
        {
            var serialized = attr_value.ToByteArray();
            c_api.TF_FunctionSetAttrValueProto(_func_graph_handle, _name, serialized, serialized.Length, tf.Status);
            tf.Status.Check(true);
        }
    }

    public override Graph as_default()
    {
        tf.Context.graph_mode(isFunc: true);
        ops.set_default_graph(this);
        return this;
    }

    public override void Exit()
    {
        tf.Context.restore_mode();
        ops.pop_graph();
    }

    public void Dispose()
    {
        c_api.TFE_ContextRemoveFunction(tf.Context, _graph_key, tf.Status);
    }

    public static FuncGraph func_graph_from_func(string name, Func<object[], object[]> func, 
        object[] args, Dictionary<string, object> kwargs, TensorSpec[] signature = null, 
        FuncGraph func_graph = null, bool autograph = false, object autograph_options = null, 
        bool add_control_dependencies = true, string[] arg_names = null, 
        Tensor op_return_value = null, bool capture_by_value = false, 
        bool acd_record_initial_resource_uses = false)
    {
        if(func_graph is null)
        {
            func_graph = new FuncGraph(name);
        }

        // TODO(Rinne): deal with control dependencies.

        func_graph.as_default();
        var current_scope = variable_scope.get_variable_scope();
        var default_use_resource = current_scope.use_resource;
        current_scope.use_resource = true;

        if(signature is not null)
        {
            args = signature;
            kwargs = new Dictionary<string, object>();
        }
        var func_args = _get_defun_inputs_from_args(args, arg_names);
        var func_kwargs = _get_defun_inputs_from_kwargs(kwargs);

        if(func_kwargs is not null && func_kwargs.Count > 0)
        {
            throw new NotImplementedException("The keyword args has not been supported in `func_graph_from_func`.");
        }

        foreach(var arg in nest.flatten<object>(new object[] { func_args, func_kwargs }))
        {
            if(arg is Tensor tensor && tensor.dtype == dtypes.resource)
            {
                func_graph._resource_tensor_inputs.Add(tensor);
            }
            else if (arg is ResourceVariable variable)
            {
                func_graph._resource_tensor_inputs.Add(variable.Handle);
            }
        }

        // skip the assignment of `func_graph.structured_input_signature`.

        var flat_func_args = nest.flatten(func_args as object);
        var flat_func_kwargs = nest.flatten(func_kwargs as object);
        func_graph.Inputs = new Tensors(flat_func_args.concat(flat_func_kwargs)
            .Where(x => x is Tensor).Select(x => (Tensor)x).ToArray());

        //var func_args_before = nest.pack_sequence_as(func_args, flat_func_args, true);
        //var func_kwargs_before = nest.pack_sequence_as(func_kwargs, flat_func_kwargs, true);

        Tensor convert(object x)
        {
            if (x is null) return null;
            Tensor res = null;
            if(op_return_value is not null && x is Operation)
            {
                tf_with(ops.control_dependencies(new object[] { x }), _ =>
                {
                    res = array_ops.identity(op_return_value);
                });
            }
            else if(x is not TensorArray)
            {
                Debug.Assert(x is Tensor);
                res = ops.convert_to_tensor_or_composite(x as Tensor);
            }
            else
            {
                throw new NotImplementedException($"The `TensorArray` is not supported here currently.");
            }
            if (add_control_dependencies)
            {
                // TODO(Rinne): `x = deps_ctx.mark_as_return(x)`.
            }
            return res;
        }

        if (autograph)
        {
            throw new NotImplementedException("The autograph of `func_graph_from_func` has not been supported.");
        }

        var func_outputs = func(func_args);
        func_outputs = variable_utils.convert_variables_to_tensors(func_outputs);
        func_outputs = func_outputs.Select(x => convert(x)).ToArray();
        // TODO(Rinne): `check_func_mutation`.

        current_scope.use_resource = default_use_resource;

        var graph_variables = func_graph._watched_variables.ToList();
        HashSet<IVariableV1> arg_variables = new HashSet<IVariableV1>();
        List<Tensor> inputs = new();
        foreach(var arg in composite_tensor_utils.flatten_with_variables(func_args))
        {
            if(arg is BaseResourceVariable variable)
            {
                var resource_placeholder = func_graph.pop_capture(variable.Handle);
                if(resource_placeholder is null)
                {
                    continue;
                }
                Debug.Assert(variable is IVariableV1);
                arg_variables.Add(variable as IVariableV1);
                inputs.Add(resource_placeholder);
            }
            else if(arg is Tensor tensor)
            {
                inputs.Add(tensor);
            }
        }
        var variables = graph_variables.Select(v =>
        {
            if (v.TryGetTarget(out var target))
            {
                return target;
            }
            else
            {
                return null;
            }
        }).Where(v => v is not null && !arg_variables.Contains(v));
        func_graph.Inputs = inputs.Concat(func_graph.internal_captures).ToArray();
        func_graph._structured_outputs = func_outputs;
        func_graph.Outputs.AddRange(func_graph.FlatStructuredOutputs.Where(x => x is not null)
            .Select(x => func_graph.capture(x)));

        func_graph.Variables = variables;

        func_graph.Exit();

        if (add_control_dependencies)
        {
            // TODO(Rinne): implement it.
        }
        return func_graph;
    }

    private static object[] _get_defun_inputs_from_args(object[] args, string[] names)
    {
        return _get_defun_inputs(args, names, args) as object[];
    }

    private static Dictionary<string, object> _get_defun_inputs_from_kwargs(Dictionary<string, object> kwargs)
    {
        // TODO(Rinne): implement it.
        Debug.Assert(kwargs is null || kwargs.Count == 0);
        return kwargs;
        //string[] names;
        //object[] args;
        //if(kwargs is not null && kwargs.Count > 0)
        //{
        //    var sorted_kwargs = kwargs.OrderBy(x => x.Key);
        //    names = sorted_kwargs.Select(x => x.Key).ToArray();
        //    args = sorted_kwargs.Select(x => x.Value).ToArray();
        //}
        //else
        //{
        //    names = new string[0];
        //    args = new object[0];
        //}
        //return _get_defun_inputs(args, names, kwargs) as Dictionary<string, object>;
    }

    private static object _get_defun_inputs(object[] args, string[] names, object structured_args)
    {
        List<object> function_inputs = new();
        if(names is null)
        {
            names = new string[args.Length];
        }

        foreach(var (arg_value, name) in zip(args, names))
        {
            foreach(var val in composite_tensor_utils.flatten_with_variables_or_variable_specs(arg_value))
            {
                function_inputs.Add(_get_defun_input(val, name));
            }
        }
        return nest.pack_sequence_as(structured_args, nest.flatten<object>(function_inputs), true);
    }

    private static object _get_defun_input(object arg, string name)
    {
        var func_graph = ops.get_default_graph() as FuncGraph;
        Debug.Assert(func_graph is not null);
        if (arg is Tensor tensor)
        {
            Tensor placeholder;
            try
            {
                placeholder = GraphOnlyOps.graph_placeholder(tensor.dtype, tensor.shape, name);
            }
            catch (ValueError ex)
            {
                tf.Logger.Warning(ex.ToString());
                placeholder = GraphOnlyOps.graph_placeholder(tensor.dtype, tensor.shape);
            }
            handle_data_util.copy_handle_data(tensor, placeholder);
            if (name is not null)
            {
                placeholder.op._set_attr("_user_specified_name", new AttrValue()
                {
                    S = tf.compat.as_bytes(name)
                });
            }
            return placeholder;
        }
        else if (arg is TensorSpec spec)
        {
            string requested_name;
            if (!string.IsNullOrEmpty(spec.name))
            {
                requested_name = spec.name;
            }
            else
            {
                requested_name = name;
            }
            Tensor placeholder;
            try
            {
                placeholder = GraphOnlyOps.graph_placeholder(spec.dtype, spec.shape, requested_name);
            }
            catch (ValueError)
            {
                // TODO(Rinne): Add warning here.
                placeholder = GraphOnlyOps.graph_placeholder(spec.dtype, spec.shape);
            }
            if (name is not null)
            {
                placeholder.op._set_attr("_user_specified_name", new AttrValue()
                {
                    S = tf.compat.as_bytes(requested_name)
                });
            }
            return placeholder;
        }
        else if (arg is BaseResourceVariable variable)
        {
            var placeholder = func_graph.capture(variable.Handle, name);
            placeholder.op._set_attr("_user_specified_name", new AttrValue()
            {
                S = tf.compat.as_bytes(name)
            });
            return arg;
        }
        // TODO(Rinne): deal with `VariableSpec`.
        else
        {
            return arg;
        }
    }
}
