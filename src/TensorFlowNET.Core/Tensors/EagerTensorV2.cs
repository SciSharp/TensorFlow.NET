using NumSharp;
using System;
using System.Linq;
using Tensorflow.Eager;
using Tensorflow.Util;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class EagerTensorV2 : DisposableObject, ITensor
    {
        SafeTensorHandleHandle EagerTensorHandle;
        public string Device
        {
            get
            {
                using var _ = EagerTensorHandle.Lease();
                return c_api.StringPiece(c_api.TFE_TensorHandleDeviceName(EagerTensorHandle, tf.status.Handle));
            }
        }

        public EagerTensorV2(IntPtr handle)
        {
            EagerTensorHandle = c_api.TFE_EagerTensorHandle(handle);
            _handle = c_api.TFE_TensorHandleResolve(EagerTensorHandle, tf.status.Handle);
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

            EagerTensorHandle = c_api.TFE_NewTensorHandle(_handle, tf.status.Handle);
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

        protected override void DisposeManagedResources()
        {
            EagerTensorHandle.Dispose();
        }

        protected override void DisposeUnmanagedResources(IntPtr handle)
        {
            c_api.TF_DeleteTensor(_handle);
        }
    }
}
