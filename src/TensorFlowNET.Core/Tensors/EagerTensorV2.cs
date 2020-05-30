using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using Tensorflow.Eager;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class EagerTensorV2 : DisposableObject, ITensor
    {
        IntPtr tfe_tensor_handle;
        public IntPtr EagerTensorHandle { get; set; }
        public string Device => c_api.StringPiece(c_api.TFE_TensorHandleDeviceName(tfe_tensor_handle, status));

        static Status status = new Status();

        public EagerTensorV2(IntPtr handle)
        {
            EagerTensorHandle = handle;
            tfe_tensor_handle = c_api.TFE_EagerTensorHandle(handle);
            _handle = c_api.TFE_TensorHandleResolve(tfe_tensor_handle, status);
        }

        public unsafe EagerTensorV2(NDArray nd, string device_name = "")
        {
            if (nd.typecode == NPTypeCode.String)
                throw new NotImplementedException("Support for NDArray of type string not implemented yet");

            var arraySlice = nd.Unsafe.Storage.Shape.IsContiguous ? nd.GetData() : nd.CloneData();

            _handle = c_api.TF_NewTensor(nd.dtype.as_dtype(),
                    nd.shape.Select(i => (long)i).ToArray(),
                    nd.ndim,
                    new IntPtr(arraySlice.Address),
                    nd.size * nd.dtypesize,
                    deallocator: (IntPtr dataPtr, long len, IntPtr args) =>
                    {

                    }, IntPtr.Zero);

            tfe_tensor_handle = c_api.TFE_NewTensorHandle(_handle, status);
            EagerTensorHandle = c_api.TFE_NewEagerTensor();
        }

        /*public unsafe EagerTensorV2(float[,] value)
        {
            var dims = new long[] { value.Rank, value.Length / value.Rank };
            fixed (float* pointer = &value[0, 0])
            {
                // The address stored in pointerToFirst
                // is valid only inside this fixed statement block.
                tensorHandle = c_api.TF_NewTensor(TF_DataType.TF_FLOAT,
                    dims,
                    value.Rank,
                    new IntPtr(pointer),
                    value.Length * sizeof(float),
                    deallocator: (IntPtr dataPtr, long len, IntPtr args) =>
                    {

                    }, IntPtr.Zero);


                localTensorHandle = c_api.TFE_NewTensorHandle(tensorHandle, status);
                _handle = c_api.TFE_EagerTensorFromHandle(tf.context, localTensorHandle);
            }
        }*/

        protected override void DisposeUnmanagedResources(IntPtr handle)
        {
            c_api.TF_DeleteTensor(_handle);
            c_api.TFE_DeleteTensorHandle(tfe_tensor_handle);
            c_api.TFE_DeleteEagerTensor(EagerTensorHandle);
        }
    }
}
