using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public static partial class c_api
    {
        /// <summary>
        /// Get the OpList of all OpDefs defined in this address space.
        /// </summary>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_GetAllOpList();

        /// <summary>
        /// For inputs that take a single tensor.
        /// </summary>
        /// <param name="desc"></param>
        /// <param name="input"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_AddInput(IntPtr desc, TF_Output input);

        /// <summary>
        /// For inputs that take a list of tensors.
        /// inputs must point to TF_Output[num_inputs].
        /// </summary>
        /// <param name="desc"></param>
        /// <param name="inputs"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_AddInputList(IntPtr desc, TF_Output[] inputs, int num_inputs);

        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_FinishOperation(IntPtr desc, IntPtr status);

        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_NewOperation(IntPtr graph, string opType, string oper_name);

        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_OperationDevice(IntPtr oper);

        /// <summary>
        /// Sets `output_attr_value` to the binary-serialized AttrValue proto
        /// representation of the value of the `attr_name` attr of `oper`.
        /// </summary>
        /// <param name="oper"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern int TF_OperationGetAttrValueProto(IntPtr oper, string attr_name, IntPtr output_attr_value, IntPtr status);

        /// <summary>
        /// TF_Output producer = TF_OperationInput(consumer);
        /// There is an edge from producer.oper's output (given by
        /// producer.index) to consumer.oper's input (given by consumer.index).
        /// </summary>
        /// <param name="oper_in"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern TF_Output TF_OperationInput(TF_Input oper_in);

        [DllImport(TensorFlowLibName)]
        public static extern int TF_OperationInputListLength(IntPtr oper, string arg_name, IntPtr status);

        [DllImport(TensorFlowLibName)]
        public static extern TF_DataType TF_OperationInputType(TF_Input oper_in);

        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_OperationName(IntPtr oper);

        [DllImport(TensorFlowLibName)]
        public static extern int TF_OperationNumInputs(IntPtr oper);

        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_OperationOpType(IntPtr oper);

        /// <summary>
        /// Get the number of control inputs to an operation.
        /// </summary>
        /// <param name="oper"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern int TF_OperationNumControlInputs(IntPtr oper);

        /// <summary>
        /// Get the number of operations that have `*oper` as a control input.
        /// </summary>
        /// <param name="oper"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern int TF_OperationNumControlOutputs(IntPtr oper);

        [DllImport(TensorFlowLibName)]
        public static extern int TF_OperationNumOutputs(IntPtr oper);

        /// <summary>
        /// Get the number of current consumers of a specific output of an
        /// operation.  Note that this number can change when new operations
        /// are added to the graph.
        /// </summary>
        /// <param name="oper_out"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern int TF_OperationOutputNumConsumers(TF_Output oper_out);

        /// <summary>
        /// Get list of all current consumers of a specific output of an
        /// operation.
        /// </summary>
        /// <param name="oper_out"></param>
        /// <param name="consumers"></param>
        /// <param name="max_consumers"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern unsafe int TF_OperationOutputConsumers(TF_Output oper_out, TF_Input * consumers, int max_consumers);

        [DllImport(TensorFlowLibName)]
        public static extern TF_DataType TF_OperationOutputType(TF_Output oper_out);

        [DllImport(TensorFlowLibName)]
        public static extern void TF_OperationToNodeDef(IntPtr oper, IntPtr buffer, IntPtr status);

        [DllImport(TensorFlowLibName)]
        public static extern int TF_OperationOutputListLength(IntPtr oper, string arg_name, IntPtr status);

        [DllImport(TensorFlowLibName)]
        public static extern void TF_SetAttrValueProto(IntPtr desc, string attr_name, IntPtr proto, UIntPtr proto_len, IntPtr status);

        /// <summary>
        /// Set `num_dims` to -1 to represent "unknown rank".
        /// </summary>
        /// <param name="desc"></param>
        /// <param name="attr_name"></param>
        /// <param name="dims"></param>
        /// <param name="num_dims"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_SetAttrShape(IntPtr desc, string attr_name, long[] dims, int num_dims);

        [DllImport(TensorFlowLibName)]
        public static extern void TF_SetAttrTensor(IntPtr desc, string attr_name, IntPtr value, IntPtr status);

        [DllImport(TensorFlowLibName)]
        public static extern void TF_SetAttrType(IntPtr desc, string attr_name, TF_DataType value);
    }
}
