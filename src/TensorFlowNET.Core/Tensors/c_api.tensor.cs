using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public static partial class c_api
    {
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_AllocateTensor(TF_DataType dtype, long[] dims, int num_dims, ulong len);

        /// <summary>
        /// returns the sizeof() for the underlying type corresponding to the given TF_DataType enum value.
        /// </summary>
        /// <param name="dt"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern ulong TF_DataTypeSize(TF_DataType dt);

        /// <summary>
        /// Destroy a tensor.
        /// </summary>
        /// <param name="tensor"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TF_DeleteTensor(IntPtr tensor);

        /// <summary>
        /// Return the length of the tensor in the "dim_index" dimension.
        /// REQUIRES: 0 <= dim_index < TF_NumDims(tensor)
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="dim_index"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern long TF_Dim(IntPtr tensor, int dim_index);

        /// <summary>
        /// Return a new tensor that holds the bytes data[0,len-1]
        /// </summary>
        /// <param name="dataType"></param>
        /// <param name="dims"></param>
        /// <param name="num_dims"></param>
        /// <param name="data"></param>
        /// <param name="len">num_bytes, ex: 6 * sizeof(float)</param>
        /// <param name="deallocator"></param>
        /// <param name="deallocator_arg"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_NewTensor(TF_DataType dataType, long[] dims, int num_dims, IntPtr data, ulong len, Deallocator deallocator, ref bool deallocator_arg);

        /// <summary>
        /// Return the number of dimensions that the tensor has.
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern int TF_NumDims(IntPtr tensor);

        /// <summary>
        /// Return the size of the underlying data in bytes.
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern ulong TF_TensorByteSize(IntPtr tensor);

        /// <summary>
        /// Return a pointer to the underlying data buffer.
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_TensorData(IntPtr tensor);

        /// <summary>
        /// Return the type of a tensor element.
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern TF_DataType TF_TensorType(IntPtr tensor);
    }
}
