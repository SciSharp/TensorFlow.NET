using Google.Protobuf;
using Tensorflow.Eager;
using Tensorflow.Exceptions;
using static Tensorflow.Binding;

namespace Tensorflow.Graphs;

/// <summary>
/// Graph representing a function body.
/// </summary>
public class FuncGraph : Graph, IDisposable
{
    SafeFuncGraphHandle _func_graph_handle;
    public string FuncName => _graph_key;

    public Tensors Inputs { get; set; } = new Tensors();
    public Tensors Outputs { get; set; } = new Tensors();
    public string Name { get; set; }
    public Dictionary<string, string> Attrs { get; set; }

    Dictionary<long, (Tensor, Tensor)> _captures
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
    }

    public FuncGraph(SafeGraphHandle handle, string name, Dictionary<string, string> attrs) : base()
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
    }

    public void replace_capture(Tensor tensor, Tensor placeholder)
    {
        _captures[tensor.Id] = (tensor, placeholder);
    }

    public void ToGraph(Operation[] opers,
        Tensor[] inputs, Tensor[] outputs,
        string[] output_names)
    {
        var status = new Status();
        if (output_names != null && output_names.Length == 0)
        {
            output_names = null;
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
            output_names,
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
        if(tensor is EagerTensor)
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
            var serialized = new AttrValue
            {
                S = ByteString.CopyFromUtf8(attr_value)
            }.ToByteArray();
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
}
