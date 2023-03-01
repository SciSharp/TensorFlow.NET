using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.InteropServices;

namespace Tensorflow
{
    public unsafe partial class c_api
    {
        [DllImport(TensorFlowLibName)]
        internal static extern IntPtr TF_NewCheckpointReader(string filename, SafeStatusHandle status);
        [DllImport(TensorFlowLibName)]
        internal static extern void TF_DeleteCheckpointReader(IntPtr reader);
        [DllImport(TensorFlowLibName)]
        internal static extern int TF_CheckpointReaderHasTensor(IntPtr reader, string name);
        [DllImport(TensorFlowLibName)]
        internal static extern string TF_CheckpointReaderGetVariable(IntPtr reader, int index);
        [DllImport(TensorFlowLibName)]
        internal static extern int TF_CheckpointReaderSize(IntPtr reader);
        [DllImport(TensorFlowLibName)]
        internal static extern TF_DataType TF_CheckpointReaderGetVariableDataType(IntPtr reader, string name);
        [DllImport(TensorFlowLibName)]
        internal static extern void TF_CheckpointReaderGetVariableShape(IntPtr reader, string name, long[] dims, int num_dims, SafeStatusHandle status);
        [DllImport(TensorFlowLibName)]
        internal static extern int TF_CheckpointReaderGetVariableNumDims(IntPtr reader, string name);
        [DllImport(TensorFlowLibName)]
        internal static extern SafeTensorHandle TF_CheckpointReaderGetTensor(IntPtr reader, string name, SafeStatusHandle status);
    }
}
