using Google.Protobuf;
using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Eager;
using Tensorflow.Exceptions;
using static Tensorflow.Binding;

namespace Tensorflow.Graphs
{
    /// <summary>
    /// Graph representing a function body.
    /// </summary>
    public class FuncGraph : Graph
    {
        Graph outer_graph;
        public Graph OuterGraph => outer_graph;

        string func_name;

        // _handle == IntPtr.Zero ? string.Empty : c_api.StringPiece(c_api.TF_FunctionName(_handle));
        IntPtr func_handle;
        public string FuncName => func_name;

        public Tensors Inputs { get; set; }
        public Tensors Outputs { get; set; }
        public Dictionary<string, string> Attrs { get; set; }

        Dictionary<long, (Tensor, Tensor)> _captures = new Dictionary<long, (Tensor, Tensor)>();
        /// <summary>
        /// Construct a new FuncGraph.
        /// </summary>
        public FuncGraph(string name) : base()
        {
            outer_graph = ops.get_default_graph();
            func_name = name;

            tf.Context.graph_mode();
            as_default();
        }

        public FuncGraph(IntPtr handle, string name, Dictionary<string, string> attrs) : base()
        {
            outer_graph = ops.get_default_graph();
            func_name = name;
            Attrs = attrs;
            // Will to test if FuncGraph has memory leak
            // c_api.TF_DeleteGraph(_handle);
            _handle = handle;

            tf.Context.graph_mode();
            as_default();
        }

        public IntPtr ToGraph(Operation[] opers,
            Tensor[] inputs, Tensor[] outputs,
            string[] output_names)
        {
            using var status = new Status();
            func_handle = c_api.TF_GraphToFunction(_handle,
                func_name,
                false,
                opers.Length,
                opers.Select(x => (IntPtr)x).ToArray(),
                inputs.Length,
                inputs.Select(x => new TF_Output(x.op, 0)).ToArray(),
                outputs.Length,
                outputs.Select(x => new TF_Output(x.op, 0)).ToArray(),
                output_names == null || output_names.Length == 0 ? null : output_names,
                IntPtr.Zero,
                null,
                status.Handle);
            status.Check(true);

            SetAttrs();

            c_api.TF_GraphCopyFunction(outer_graph, func_handle, IntPtr.Zero, status.Handle);
            status.Check(true);

            c_api.TFE_ContextAddFunction(tf.Context.Handle, func_handle, status.Handle);
            status.Check(true);

            func_name = c_api.StringPiece(c_api.TF_FunctionName(func_handle));

            Inputs = inputs;
            // mark_as_return
            Outputs = outputs;// .Select(x => array_ops.identity(x)).ToArray();

            tf.Context.restore_mode();

            return func_handle;
        }

        public override Operation create_op(string op_type, Tensor[] inputs, TF_DataType[] dtypes, TF_DataType[] input_types = null, string name = null, Dictionary<string, AttrValue> attrs = null, OpDef op_def = null, bool compute_device = true)
        {
            foreach(var (i, inp) in enumerate(inputs))
                inputs[i] = capture(inp);

            return base.create_op(op_type, inputs, dtypes, input_types, name, attrs, op_def, compute_device);
        }

        Tensor capture(Tensor tensor, string name = null, TF_DataType shape = TF_DataType.DtInvalid)
        {
            if(tensor is EagerTensor)
            {
                throw new NotImplementedException("");
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

        Tensor _capture_helper(Tensor tensor, string name, TensorShape shape = null)
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
                placeholder = _captures[tensor.Id].Item1;
            }

            BackwardFunction _backward_function_wrapper = (output_grads, unneeded_gradients) =>
            {
                return output_grads;
            };

            tf.Runner.RecordGradient("captured_value",
                new[] { placeholder }, null,
                new[] { tensor },
                getBackwardFunction: () => _backward_function_wrapper
                /*getForwardFunction: forward_function*/);

            return placeholder;
        }

        void add_capture(Tensor tensor, Tensor placeholder)
        {
            _captures[tensor.Id] = (tensor, placeholder);
            if (Inputs == null)
                Inputs = new Tensors(placeholder);
            else
            {
                var inputs = Inputs.ToList();
                inputs.Add(placeholder);
                Inputs = new Tensors(inputs.ToArray());
            }
        }

        Tensor _create_substitute_placeholder(Tensor value, 
            string name = null, 
            TF_DataType dtype = TF_DataType.DtInvalid, 
            TensorShape shape = null)
        {
            if (shape is null)
                shape = value.shape;
            if (dtype == TF_DataType.DtInvalid)
                dtype = value.dtype;

            var placeholder = tf_with(ops.control_dependencies(null), ctl => array_ops.placeholder(dtype, shape: shape, name: name));
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
                c_api.TF_FunctionSetAttrValueProto(func_handle, _name, serialized, serialized.Length, tf.Status.Handle);
                tf.Status.Check(true);
            }
        }

        protected override void DisposeManagedResources()
        {
            base.DisposeManagedResources();
        }
    }
}
