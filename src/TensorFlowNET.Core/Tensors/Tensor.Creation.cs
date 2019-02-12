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
                    Marshal.Copy(nd.Data<short>(), 0, dotHandle, nd.size);
                    break;
                case "Int32":
                    Marshal.Copy(nd.Data<int>(), 0, dotHandle, nd.size);
                    break;
                case "Single":
                    Marshal.Copy(nd.Data<float>(), 0, dotHandle, nd.size);
                    break;
                case "Double":
                    Marshal.Copy(nd.Data<double>(), 0, dotHandle, nd.size);
                    break;
                case "String":
                    var str = nd.Data<string>()[0];
                    ulong dst_len = c_api.TF_StringEncodedSize((ulong)str.Length);
                    var dataType1 = ToTFDataType(nd.dtype);
                    // shape
                    var dims1 = nd.shape.Select(x => (long)x).ToArray();

                    var tfHandle1 = c_api.TF_AllocateTensor(dataType1,
                        dims1,
                        nd.ndim,
                        dst_len + sizeof(Int64));

                    dotHandle = c_api.TF_TensorData(tfHandle1);
                    Marshal.WriteInt64(dotHandle, 0);
                    c_api.TF_StringEncode(str, (ulong)str.Length, dotHandle + sizeof(Int64), dst_len, status);
                    return tfHandle1;
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
                nd.ndim,
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
