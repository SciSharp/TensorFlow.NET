using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using static Tensorflow.c_api;

namespace Tensorflow
{
    public partial class Tensor
    {
        /// <summary>
        /// if original buffer is free.
        /// </summary>
        private bool deallocator_called;

        public Tensor(IntPtr handle)
        {
            _handle = handle;
        }

        public Tensor(NDArray nd)
        {
            _handle = Allocate(nd);
        }

        public unsafe Tensor(byte[] buffer)
        {
            var size = c_api.TF_StringEncodedSize((UIntPtr)buffer.Length);
            _handle = TF_AllocateTensor(TF_DataType.TF_STRING, IntPtr.Zero, 0, (UIntPtr)((ulong)size + 8));

            IntPtr tensor = c_api.TF_TensorData(_handle);
            Marshal.WriteInt64(tensor, 0);
            fixed (byte* src = &buffer[0])
                c_api.TF_StringEncode(src, (UIntPtr)buffer.Length, (sbyte*)(tensor + sizeof(Int64)), size, status);

            status.Check(true);
        }

        private IntPtr Allocate(NDArray nd)
        {
            IntPtr dotHandle = IntPtr.Zero;
            ulong size = 0;

            if (nd.dtype.Name != "String")
            {
                dotHandle = Marshal.AllocHGlobal(nd.dtypesize * nd.size);
                size = (ulong)(nd.size * nd.dtypesize);
            }

            var dataType = ToTFDataType(nd.dtype);
            // shape
            var dims = nd.shape.Select(x => (long)x).ToArray();

            switch (nd.dtype.Name)
            {
                case "Int16":
                    Marshal.Copy(nd.ravel().Data<short>(), 0, dotHandle, nd.size);
                    break;
                case "Int32":
                    Marshal.Copy(nd.ravel().Data<int>(), 0, dotHandle, nd.size);
                    break;
                case "Single":
                    Marshal.Copy(nd.ravel().Data<float>(), 0, dotHandle, nd.size);
                    break;
                case "Double":
                    Marshal.Copy(nd.ravel().Data<double>(), 0, dotHandle, nd.size);
                    break;
                //case "Byte":
                    /*var bb = nd.Data<byte>();
                    var bytes = Marshal.AllocHGlobal(bb.Length);
                    Marshal.Copy(bb, 0, bytes, bb.Length);
                    ulong bytes_len = c_api.TF_StringEncodedSize((ulong)bb.Length);
                    var dataTypeByte = ToTFDataType(nd.dtype);
                    // shape
                    var dims2 = nd.shape.Select(x => (long)x).ToArray();

                    var tfHandle2 = c_api.TF_AllocateTensor(dataTypeByte,
                        dims2,
                        nd.ndim,
                        bytes_len + sizeof(Int64));

                    dotHandle = c_api.TF_TensorData(tfHandle2);
                    Marshal.WriteInt64(dotHandle, 0);
                    c_api.TF_StringEncode(bytes, (ulong)bb.Length, dotHandle + sizeof(Int64), bytes_len, status);
                    return tfHandle2;*/
                    break;
                //case "String":
                    /*string ss = nd.Data<string>()[0];
                    var str = Marshal.StringToHGlobalAnsi(ss);
                    ulong dst_len = c_api.TF_StringEncodedSize((ulong)ss.Length);
                    var dataType1 = ToTFDataType(nd.dtype);
                    // shape
                    var dims1 = nd.shape.Select(x => (long)x).ToArray();

                    var tfHandle1 = c_api.TF_AllocateTensor(dataType1,
                        dims1,
                        nd.ndim,
                        dst_len + sizeof(Int64));

                    dotHandle = c_api.TF_TensorData(tfHandle1);
                    Marshal.WriteInt64(dotHandle, 0);
                    c_api.TF_StringEncode(str, (ulong)ss.Length, dotHandle + sizeof(Int64), dst_len, status);
                    return tfHandle1;*/
                    break;
                default:
                    throw new NotImplementedException("Marshal.Copy failed.");
            }
            
            // Free the original buffer and set flag
            Deallocator deallocator = (IntPtr values, IntPtr len, ref bool closure) =>
            {
                Marshal.FreeHGlobal(dotHandle);
                closure = true;
            };

            var tfHandle = c_api.TF_NewTensor(dataType,
                dims,
                dims.Length,
                dotHandle,
                size,
                deallocator,
                ref deallocator_called);

            return tfHandle;
        }

        public Tensor(Operation op, int value_index, TF_DataType dtype)
        {
            this.op = op;
            this.value_index = value_index;
            this._dtype = dtype;
            _id = ops.uid();
        }
    }
}
