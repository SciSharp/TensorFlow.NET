using System;
using System.Collections.Generic;
using System.Linq;
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

        public FuncGraph(IntPtr handle, string name)
        {
            outer_graph = ops.get_default_graph();
            func_name = name;

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

            c_api.TF_GraphCopyFunction(outer_graph, func_handle, IntPtr.Zero, status.Handle);
            status.Check(true);

            c_api.TFE_ContextAddFunction(tf.Context.Handle, func_handle, status.Handle);
            status.Check(true);

            func_name = c_api.StringPiece(c_api.TF_FunctionName(func_handle));

            Inputs = inputs;
            // mark_as_return
            Outputs = outputs.Select(x => array_ops.identity(x)).ToArray();

            tf.Context.restore_mode();

            return func_handle;
        }

        protected override void DisposeManagedResources()
        {
            base.DisposeManagedResources();
        }
    }
}
