using System.Runtime.InteropServices;
using Tensorflow.Checkpoint;

namespace Tensorflow
{
    public unsafe partial class c_api
    {
        [DllImport(TensorFlowLibName)]
        internal static extern SafeCheckpointReaderHandle TF_NewCheckpointReader(string filename, SafeStatusHandle status);
        [DllImport(TensorFlowLibName)]
        internal static extern void TF_DeleteCheckpointReader(IntPtr reader);
        [DllImport(TensorFlowLibName)]
        internal static extern int TF_CheckpointReaderHasTensor(SafeCheckpointReaderHandle reader, string name);
        [DllImport(TensorFlowLibName)]
        internal static extern IntPtr TF_CheckpointReaderGetVariable(SafeCheckpointReaderHandle reader, int index);
        [DllImport(TensorFlowLibName)]
        internal static extern int TF_CheckpointReaderSize(SafeCheckpointReaderHandle reader);
        [DllImport(TensorFlowLibName)]
        internal static extern TF_DataType TF_CheckpointReaderGetVariableDataType(SafeCheckpointReaderHandle reader, string name);
        [DllImport(TensorFlowLibName)]
        internal static extern void TF_CheckpointReaderGetVariableShape(SafeCheckpointReaderHandle reader, string name, long[] dims, int num_dims, SafeStatusHandle status);
        [DllImport(TensorFlowLibName)]
        internal static extern int TF_CheckpointReaderGetVariableNumDims(SafeCheckpointReaderHandle reader, string name);
        [DllImport(TensorFlowLibName)]
        internal static extern SafeTensorHandle TF_CheckpointReaderGetTensor(SafeCheckpointReaderHandle reader, string name, SafeStatusHandle status);
    }
}
