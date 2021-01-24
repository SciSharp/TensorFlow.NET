using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public partial class Tensor
    {
        public unsafe IntPtr StringTensor(string[] strings, TensorShape shape)
        {
            // convert string array to byte[][]
            var buffer = new byte[strings.Length][];
            for (var i = 0; i < strings.Length; i++)
                buffer[i] = Encoding.UTF8.GetBytes(strings[i]);

            return StringTensor(buffer, shape);
        }

        public unsafe IntPtr StringTensor(byte[][] buffer, TensorShape shape)
        {
            ulong size = 0;
            foreach (var b in buffer)
                size += c_api.TF_StringEncodedSize((ulong)b.Length);

            var src_size = size + (ulong)buffer.Length * sizeof(ulong);
            var handle = c_api.TF_AllocateTensor(TF_DataType.TF_STRING,
                shape.dims.Select(x => (long)x).ToArray(),
                shape.ndim,
                src_size);
            AllocationType = AllocationType.Tensorflow;

            IntPtr data_start = c_api.TF_TensorData(handle);
            IntPtr string_start = data_start + buffer.Length * sizeof(ulong);
            IntPtr limit = data_start + (int)src_size;
            ulong offset = 0;
            for (int i = 0; i < buffer.Length; i++)
            {
                Marshal.WriteInt64(data_start, i * sizeof(ulong), (long)offset);
                if (buffer[i].Length == 0)
                {
                    Marshal.WriteByte(string_start, 0);
                    break;
                }

                fixed (byte* src = &buffer[i][0])
                {
                    /*Marshal.WriteByte(string_start, Convert.ToByte(buffer[i].Length));
                    tf.memcpy((string_start + 1).ToPointer(), src, (ulong)buffer[i].Length);
                    string_start += buffer[i].Length + 1;
                    offset += buffer[i].Length + 1;*/

                    var written = c_api.TF_StringEncode(src, (ulong)buffer[i].Length, (byte*)string_start, (ulong)(limit.ToInt64() - string_start.ToInt64()), tf.Status.Handle);
                    tf.Status.Check(true);
                    string_start += (int)written;
                    offset += written;
                }
            }

            return handle;
        }

        /// <summary>
        ///     Extracts string array from current Tensor.
        /// </summary>
        /// <exception cref="InvalidOperationException">When <see cref="dtype"/> != TF_DataType.TF_STRING</exception>
        public unsafe string[] StringData()
        {
            var buffer = StringBytes();

            var _str = new string[buffer.Length];
            for (int i = 0; i < _str.Length; i++)
                _str[i] = Encoding.UTF8.GetString(buffer[i]);

            return _str;
        }

        public unsafe byte[][] StringBytes()
        {
            if (dtype != TF_DataType.TF_STRING)
                throw new InvalidOperationException($"Unable to call StringData when dtype != TF_DataType.TF_STRING (dtype is {dtype})");

            //
            // TF_STRING tensors are encoded with a table of 8-byte offsets followed by TF_StringEncode-encoded bytes.
            // [offset1, offset2,...,offsetn, s1size, s1bytes, s2size, s2bytes,...,snsize,snbytes]
            //
            long size = 1;
            foreach (var s in TensorShape.dims)
                size *= s;

            var buffer = new byte[size][];
            var data_start = c_api.TF_TensorData(_handle);
            var string_start = data_start + (int)(size * sizeof(ulong));
            for (int i = 0; i < buffer.Length; i++)
            {
                var len = *(byte*)string_start;
                buffer[i] = new byte[len];
                string_start += 1;
                Marshal.Copy(string_start, buffer[i], 0, len);
                string_start += len;
            }

            return buffer;
        }
    }
}
