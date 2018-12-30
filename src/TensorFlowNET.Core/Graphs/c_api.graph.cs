using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public static partial class c_api
    {
        /// <summary>
        /// Destroy an options object.  Graph will be deleted once no more
        /// TFSession's are referencing it.
        /// </summary>
        /// <param name="graph"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_DeleteGraph(IntPtr graph);

        [DllImport(TensorFlowLibName)]
        public static extern void TF_GraphGetOpDef(IntPtr graph, string op_name, IntPtr output_op_def, IntPtr status);

        /// <summary>
        /// Returns the shape of the Tensor referenced by `output` in `graph`
        /// into `dims`. `dims` must be an array large enough to hold `num_dims`
        /// entries (e.g., the return value of TF_GraphGetTensorNumDims).
        /// </summary>
        /// <param name="graph"></param>
        /// <param name="output"></param>
        /// <param name="dims"></param>
        /// <param name="num_dims"></param>
        /// <param name="status"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_GraphGetTensorShape(IntPtr graph, TF_Output output, long[] dims, int num_dims, IntPtr status);

        /// <summary>
        /// Sets the shape of the Tensor referenced by `output` in `graph` to
        /// the shape described by `dims` and `num_dims`.
        /// </summary>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_GraphSetTensorShape(IntPtr graph, TF_Output output, long[] dims, int num_dims, IntPtr status);

        /// <summary>
        /// Write out a serialized representation of `graph` (as a GraphDef protocol
        /// message) to `output_graph_def` (allocated by TF_NewBuffer()).
        /// </summary>
        /// <param name="graph"></param>
        /// <param name="output_graph_def"></param>
        /// <param name="status"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_GraphToGraphDef(IntPtr graph, IntPtr output_graph_def, IntPtr status);
        
        /// <summary>
        /// Returns the number of dimensions of the Tensor referenced by `output`
        /// in `graph`.
        /// 
        /// If the number of dimensions in the shape is unknown, returns -1.
        /// </summary>
        /// <param name="graph"></param>
        /// <param name="output"></param>
        /// <param name="status"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern int TF_GraphGetTensorNumDims(IntPtr graph, TF_Output output, IntPtr status);

        [DllImport(TensorFlowLibName)]
        public static unsafe extern IntPtr TF_NewGraph();
    }
}
