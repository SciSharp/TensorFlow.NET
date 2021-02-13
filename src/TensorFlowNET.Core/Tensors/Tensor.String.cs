using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public partial class Tensor
    {
        const int TF_TSRING_SIZE = 24;

        public IntPtr StringTensor(string[] strings, TensorShape shape)
        {
            // convert string array to byte[][]
            var buffer = new byte[strings.Length][];
            for (var i = 0; i < strings.Length; i++)
                buffer[i] = Encoding.UTF8.GetBytes(strings[i]);

            return StringTensor(buffer, shape);
        }

        public IntPtr StringTensor(byte[][] buffer, TensorShape shape)
        {
            var handle = c_api.TF_AllocateTensor(TF_DataType.TF_STRING,
                shape.ndim == 0 ? null : shape.dims.Select(x => (long)x).ToArray(),
                shape.ndim,
                (ulong)shape.size * TF_TSRING_SIZE);

            var tstr = c_api.TF_TensorData(handle);
#if TRACK_TENSOR_LIFE
            print($"New TString 0x{handle.ToString("x16")} {AllocationType} Data: 0x{tstr.ToString("x16")}");
#endif
            for (int i = 0; i < buffer.Length; i++)
            {
                c_api.TF_StringInit(tstr);
                c_api.TF_StringCopy(tstr, buffer[i], buffer[i].Length);
                var data = c_api.TF_StringGetDataPointer(tstr);
                tstr += TF_TSRING_SIZE;
            }

            return handle;
        }

        public string[] StringData()
        {
            var buffer = StringBytes();

            var _str = new string[buffer.Length];
            for (int i = 0; i < _str.Length; i++)
                _str[i] = Encoding.UTF8.GetString(buffer[i]);

            return _str;
        }

        public byte[][] StringBytes()
        {
            if (dtype != TF_DataType.TF_STRING)
                throw new InvalidOperationException($"Unable to call StringData when dtype != TF_DataType.TF_STRING (dtype is {dtype})");

            //
            // TF_STRING tensors are encoded with a table of 8-byte offsets followed by TF_StringEncode-encoded bytes.
            // [offset1, offset2,...,offsetn, s1size, s1bytes, s2size, s2bytes,...,snsize,snbytes]
            //
            int size = 1;
            foreach (var s in TensorShape.dims)
                size *= s;

            var buffer = new byte[size][];
            var tstrings = TensorDataPointer;
            for (int i = 0; i < buffer.Length; i++)
            {
                var data = c_api.TF_StringGetDataPointer(tstrings);
                var len = c_api.TF_StringGetSize(tstrings);
                buffer[i] = new byte[len];
                // var capacity = c_api.TF_StringGetCapacity(tstrings);
                // var type = c_api.TF_StringGetType(tstrings);
                Marshal.Copy(data, buffer[i], 0, Convert.ToInt32(len));
                tstrings += TF_TSRING_SIZE;
            }
            return buffer;
        }
    }
}
