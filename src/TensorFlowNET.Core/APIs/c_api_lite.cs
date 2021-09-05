using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using Tensorflow.Lite;

namespace Tensorflow
{
    public class c_api_lite
    {
        public const string TensorFlowLibName = "tensorflowlite_c";

        public static string StringPiece(IntPtr handle)
        {
            return handle == IntPtr.Zero ? String.Empty : Marshal.PtrToStringAnsi(handle);
        }

        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TfLiteVersion();

        [DllImport(TensorFlowLibName)]
        public static extern SafeTfLiteModelHandle TfLiteModelCreateFromFile(string model_path);

        [DllImport(TensorFlowLibName)]
        public static extern void TfLiteModelDelete(IntPtr model);

        [DllImport(TensorFlowLibName)]
        public static extern SafeTfLiteInterpreterOptionsHandle TfLiteInterpreterOptionsCreate();

        [DllImport(TensorFlowLibName)]
        public static extern void TfLiteInterpreterOptionsDelete(IntPtr options);

        [DllImport(TensorFlowLibName)]
        public static extern void TfLiteInterpreterOptionsSetNumThreads(SafeTfLiteInterpreterOptionsHandle options, int num_threads);

        [DllImport(TensorFlowLibName)]
        public static extern SafeTfLiteInterpreterHandle TfLiteInterpreterCreate(SafeTfLiteModelHandle model, SafeTfLiteInterpreterOptionsHandle optional_options);

        [DllImport(TensorFlowLibName)]
        public static extern void TfLiteInterpreterDelete(IntPtr interpreter);

        [DllImport(TensorFlowLibName)]
        public static extern TfLiteStatus TfLiteInterpreterAllocateTensors(SafeTfLiteInterpreterHandle interpreter);

        [DllImport(TensorFlowLibName)]
        public static extern int TfLiteInterpreterGetInputTensorCount(SafeTfLiteInterpreterHandle interpreter);

        [DllImport(TensorFlowLibName)]
        public static extern int TfLiteInterpreterGetOutputTensorCount(SafeTfLiteInterpreterHandle interpreter);

        [DllImport(TensorFlowLibName)]
        public static extern TfLiteStatus TfLiteInterpreterResizeInputTensor(SafeTfLiteInterpreterHandle interpreter, 
            int input_index, int[] input_dims, int input_dims_size);

        [DllImport(TensorFlowLibName)]
        public static extern TfLiteTensor TfLiteInterpreterGetInputTensor(SafeTfLiteInterpreterHandle interpreter, int input_index);

        [DllImport(TensorFlowLibName)]
        public static extern TfLiteDataType TfLiteTensorType(TfLiteTensor tensor);

        [DllImport(TensorFlowLibName)]
        public static extern int TfLiteTensorNumDims(TfLiteTensor tensor);

        [DllImport(TensorFlowLibName)]
        public static extern int TfLiteTensorDim(TfLiteTensor tensor, int dim_index);

        [DllImport(TensorFlowLibName)]
        public static extern int TfLiteTensorByteSize(TfLiteTensor tensor);

        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TfLiteTensorData(TfLiteTensor tensor);

        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TfLiteTensorName(TfLiteTensor tensor);

        [DllImport(TensorFlowLibName)]
        public static extern TfLiteQuantizationParams TfLiteTensorQuantizationParams(TfLiteTensor tensor);

        [DllImport(TensorFlowLibName)]
        public static extern TfLiteStatus TfLiteTensorCopyFromBuffer(TfLiteTensor tensor, IntPtr input_data, int input_data_size);

        [DllImport(TensorFlowLibName)]
        public static extern TfLiteStatus TfLiteInterpreterInvoke(SafeTfLiteInterpreterHandle interpreter);

        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TfLiteInterpreterGetOutputTensor(SafeTfLiteInterpreterHandle interpreter, int output_index);

        [DllImport(TensorFlowLibName)]
        public static extern TfLiteStatus TfLiteTensorCopyToBuffer(TfLiteTensor output_tensor, IntPtr output_data, int output_data_size);
    }
}
