/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using System;
using System.Runtime.InteropServices;

namespace Tensorflow
{
    public partial class c_api
    {
        /// <summary>
        /// Destroy an options object.  Graph will be deleted once no more
        /// TFSession's are referencing it.
        /// </summary>
        /// <param name="graph"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_DeleteGraph(IntPtr graph);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="opts">TF_ImportGraphDefOptions*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_DeleteImportGraphDefOptions(IntPtr opts);

        /// <summary>
        /// Deletes a results object returned by TF_GraphImportGraphDefWithResults().
        /// </summary>
        /// <param name="results"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_DeleteImportGraphDefResults(IntPtr results);

        [DllImport(TensorFlowLibName)]
        public static extern string TF_GraphDebugString(IntPtr graph, out int len);

        [DllImport(TensorFlowLibName)]
        public static extern void TF_GraphGetOpDef(IntPtr graph, string op_name, SafeBufferHandle output_op_def, SafeStatusHandle status);

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
        public static extern void TF_GraphGetTensorShape(IntPtr graph, TF_Output output, long[] dims, int num_dims, SafeStatusHandle status);

        /// <summary>
        /// Import the graph serialized in `graph_def` into `graph`.
        /// Convenience function for when only return outputs are needed.
        ///
        /// `num_return_outputs` must be the number of return outputs added (i.e. the
        /// result of TF_ImportGraphDefOptionsNumReturnOutputs()).  If
        /// `num_return_outputs` is non-zero, `return_outputs` must be of length
        /// `num_return_outputs`. Otherwise it can be null.
        /// </summary>
        /// <param name="graph">TF_Graph* graph</param>
        /// <param name="graph_def">const TF_Buffer*</param>
        /// <param name="options">const TF_ImportGraphDefOptions*</param>
        /// <param name="return_outputs">TF_Output*</param>
        /// <param name="num_return_outputs">int</param>
        /// <param name="status">TF_Status*</param>
        [DllImport(TensorFlowLibName)]
        public static extern unsafe void TF_GraphImportGraphDefWithReturnOutputs(IntPtr graph, SafeBufferHandle graph_def, SafeImportGraphDefOptionsHandle options, IntPtr return_outputs, int num_return_outputs, SafeStatusHandle status);

        /// <summary>
        /// Import the graph serialized in `graph_def` into `graph`.  Returns nullptr and
        /// a bad status on error. Otherwise, returns a populated
        /// TF_ImportGraphDefResults instance. The returned instance must be deleted via
        /// TF_DeleteImportGraphDefResults().
        /// </summary>
        /// <param name="graph">TF_Graph*</param>
        /// <param name="graph_def">const TF_Buffer*</param>
        /// <param name="options">const TF_ImportGraphDefOptions*</param>
        /// <param name="status">TF_Status*</param>
        /// <returns>TF_ImportGraphDefResults*</returns>
        [DllImport(TensorFlowLibName)]
        public static extern SafeImportGraphDefResultsHandle TF_GraphImportGraphDefWithResults(IntPtr graph, SafeBufferHandle graph_def, SafeImportGraphDefOptionsHandle options, SafeStatusHandle status);

        /// <summary>
        /// Import the graph serialized in `graph_def` into `graph`.
        /// </summary>
        /// <param name="graph">TF_Graph*</param>
        /// <param name="graph_def">TF_Buffer*</param>
        /// <param name="options">TF_ImportGraphDefOptions*</param>
        /// <param name="status">TF_Status*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_GraphImportGraphDef(IntPtr graph, SafeBufferHandle graph_def, SafeImportGraphDefOptionsHandle options, SafeStatusHandle status);

        /// <summary>
        /// Iterate through the operations of a graph.
        /// </summary>
        /// <param name="graph"></param>
        /// <param name="pos"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_GraphNextOperation(IntPtr graph, ref uint pos);

        /// <summary>
        /// Returns the operation in the graph with `oper_name`. Returns nullptr if
        /// no operation found.
        /// </summary>
        /// <param name="graph"></param>
        /// <param name="oper_name"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_GraphOperationByName(IntPtr graph, string oper_name);

        /// <summary>
        /// Sets the shape of the Tensor referenced by `output` in `graph` to
        /// the shape described by `dims` and `num_dims`.
        /// </summary>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_GraphSetTensorShape(IntPtr graph, TF_Output output, long[] dims, int num_dims, SafeStatusHandle status);

        /// <summary>
        /// Write out a serialized representation of `graph` (as a GraphDef protocol
        /// message) to `output_graph_def` (allocated by TF_NewBuffer()).
        /// </summary>
        /// <param name="graph">TF_Graph*</param>
        /// <param name="output_graph_def">TF_Buffer*</param>
        /// <param name="status">TF_Status*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_GraphToGraphDef(IntPtr graph, SafeBufferHandle output_graph_def, SafeStatusHandle status);

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
        public static extern int TF_GraphGetTensorNumDims(IntPtr graph, TF_Output output, SafeStatusHandle status);

        /// <summary>
        /// Cause the imported graph to have a control dependency on `oper`. `oper`
        /// should exist in the graph being imported into.
        /// </summary>
        /// <param name="opts"></param>
        /// <param name="oper"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_ImportGraphDefOptionsAddControlDependency(SafeImportGraphDefOptionsHandle opts, IntPtr oper);

        /// <summary>
        /// Set any imported nodes with input `src_name:src_index` to have that input
        /// replaced with `dst`. `src_name` refers to a node in the graph to be imported,
        /// `dst` references a node already existing in the graph being imported into.
        /// `src_name` is copied and has no lifetime requirements.
        /// </summary>
        /// <param name="opts">TF_ImportGraphDefOptions*</param>
        /// <param name="src_name">const char*</param>
        /// <param name="src_index">int</param>
        /// <param name="dst">TF_Output</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_ImportGraphDefOptionsAddInputMapping(SafeImportGraphDefOptionsHandle opts, string src_name, int src_index, TF_Output dst);

        /// <summary>
        /// Add an operation in `graph_def` to be returned via the `return_opers` output
        /// parameter of TF_GraphImportGraphDef(). `oper_name` is copied and has no
        /// lifetime requirements.
        /// </summary>
        /// <param name="opts">TF_ImportGraphDefOptions* opts</param>
        /// <param name="oper_name">const char*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_ImportGraphDefOptionsAddReturnOperation(SafeImportGraphDefOptionsHandle opts, string oper_name);

        /// <summary>
        /// Add an output in `graph_def` to be returned via the `return_outputs` output
        /// parameter of TF_GraphImportGraphDef(). If the output is remapped via an input
        /// mapping, the corresponding existing tensor in `graph` will be returned.
        /// `oper_name` is copied and has no lifetime requirements.
        /// </summary>
        /// <param name="opts">TF_ImportGraphDefOptions*</param>
        /// <param name="oper_name">const char*</param>
        /// <param name="index">int</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_ImportGraphDefOptionsAddReturnOutput(SafeImportGraphDefOptionsHandle opts, string oper_name, int index);

        /// <summary>
        /// Returns the number of return operations added via
        /// TF_ImportGraphDefOptionsAddReturnOperation().
        /// </summary>
        /// <param name="opts"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern int TF_ImportGraphDefOptionsNumReturnOperations(SafeImportGraphDefOptionsHandle opts);

        /// <summary>
        /// Returns the number of return outputs added via
        /// TF_ImportGraphDefOptionsAddReturnOutput().
        /// </summary>
        /// <param name="opts">const TF_ImportGraphDefOptions*</param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern int TF_ImportGraphDefOptionsNumReturnOutputs(SafeImportGraphDefOptionsHandle opts);

        /// <summary>
        /// Set any imported nodes with control input `src_name` to have that input
        /// replaced with `dst`. `src_name` refers to a node in the graph to be imported,
        /// `dst` references an operation already existing in the graph being imported
        /// into. `src_name` is copied and has no lifetime requirements. 
        /// </summary>
        /// <param name="opts">TF_ImportGraphDefOptions*</param>
        /// <param name="src_name">const char*</param>
        /// <param name="dst">TF_Operation*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_ImportGraphDefOptionsRemapControlDependency(SafeImportGraphDefOptionsHandle opts, string src_name, IntPtr dst);

        /// <summary>
        /// Set the prefix to be prepended to the names of nodes in `graph_def` that will
        /// be imported into `graph`. `prefix` is copied and has no lifetime
        /// requirements.
        /// </summary>
        /// <param name="ops"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_ImportGraphDefOptionsSetPrefix(SafeImportGraphDefOptionsHandle ops, string prefix);

        /// <summary>
        /// Set whether to uniquify imported operation names. If true, imported operation
        /// names will be modified if their name already exists in the graph. If false,
        /// conflicting names will be treated as an error. Note that this option has no
        /// effect if a prefix is set, since the prefix will guarantee all names are
        /// unique. Defaults to false.
        /// </summary>
        /// <param name="ops">TF_ImportGraphDefOptions*</param>
        /// <param name="uniquify_prefix">unsigned char</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_ImportGraphDefOptionsSetUniquifyNames(SafeImportGraphDefOptionsHandle ops, char uniquify_prefix);

        /// <summary>
        /// Fetches the return operations requested via
        /// TF_ImportGraphDefOptionsAddReturnOperation(). The number of fetched
        /// operations is returned in `num_opers`. The array of return operations is
        /// returned in `opers`. `*opers` is owned by and has the lifetime of `results`.
        /// </summary>
        /// <param name="results">TF_ImportGraphDefResults*</param>
        /// <param name="num_opers">int*</param>
        /// <param name="opers">TF_Operation***</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_ImportGraphDefResultsReturnOperations(SafeImportGraphDefResultsHandle results, ref int num_opers, ref TF_Operation opers);

        /// <summary>
        /// Fetches the return outputs requested via
        /// TF_ImportGraphDefOptionsAddReturnOutput(). The number of fetched outputs is
        /// returned in `num_outputs`. The array of return outputs is returned in
        /// `outputs`. `*outputs` is owned by and has the lifetime of `results`.
        /// </summary>
        /// <param name="results">TF_ImportGraphDefResults* results</param>
        /// <param name="num_outputs">int*</param>
        /// <param name="outputs">TF_Output**</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_ImportGraphDefResultsReturnOutputs(SafeImportGraphDefResultsHandle results, ref int num_outputs, ref IntPtr outputs);

        /// <summary>
        /// This function creates a new TF_Session (which is created on success) using
        /// `session_options`, and then initializes state (restoring tensors and other
        /// assets) using `run_options`.
        /// </summary>
        /// <param name="session_options">const TF_SessionOptions*</param>
        /// <param name="run_options">const TF_Buffer*</param>
        /// <param name="export_dir">const char*</param>
        /// <param name="tags">const char* const*</param>
        /// <param name="tags_len">int</param>
        /// <param name="graph">TF_Graph*</param>
        /// <param name="meta_graph_def">TF_Buffer*</param>
        /// <param name="status">TF_Status*</param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_LoadSessionFromSavedModel(SafeSessionOptionsHandle session_options, IntPtr run_options,
            string export_dir, string[] tags, int tags_len,
            IntPtr graph, ref TF_Buffer meta_graph_def, SafeStatusHandle status);

        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_NewGraph();

        [DllImport(TensorFlowLibName)]
        public static extern SafeImportGraphDefOptionsHandle TF_NewImportGraphDefOptions();

        /// <summary>
        /// Set the shapes and types of the output's handle.
        /// </summary>
        /// <param name="graph">TF_Graph*</param>
        /// <param name="output">TF_Output</param>
        /// <param name="num_shapes_and_types">int</param>
        /// <param name="shapes">const int64_t**</param>
        /// <param name="ranks">const int*</param>
        /// <param name="types">const TF_DataType*</param>
        /// <param name="status">TF_Status*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_GraphSetOutputHandleShapesAndTypes(IntPtr graph, TF_Output output,
            int num_shapes_and_types, IntPtr[] shapes, int[] ranks, DataType[] types,
            SafeStatusHandle status);

        /// <summary>
        /// Updates 'dst' to consume 'new_src'.
        /// </summary>
        /// <param name="graph">TF_Graph*</param>
        /// <param name="new_src"></param>
        /// <param name="dst"></param>
        /// <param name="status">TF_Status*</param>
        [DllImport(TensorFlowLibName)]

        public static extern void TF_UpdateEdge(IntPtr graph, TF_Output new_src, TF_Input dst, SafeStatusHandle status);

        /// <summary>
        /// Attempts to evaluate `output`. This will only be possible if `output` doesn't
        /// depend on any graph inputs (this function is safe to call if this isn't the
        /// case though).
        /// </summary>
        /// <param name="graph"></param>
        /// <param name="output"></param>
        /// <param name="result"></param>
        /// <param name="status"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern bool TF_TryEvaluateConstant(IntPtr graph, TF_Output output, IntPtr[] result, SafeStatusHandle status);
    }
}
