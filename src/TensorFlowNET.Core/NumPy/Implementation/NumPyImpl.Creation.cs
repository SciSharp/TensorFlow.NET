using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Tensorflow.Util;
using Razorvine.Pickle;
using Tensorflow.NumPy.Pickle;
using static Tensorflow.Binding;

namespace Tensorflow.NumPy
{
    public partial class NumPyImpl
    {
        public NDArray eye(int N, int? M = null, int k = 0, TF_DataType dtype = TF_DataType.TF_DOUBLE)
        {
            if (!M.HasValue)
                M = N;

            var diag_len = min(N, M.Value);
            if (k > 0)
            {
                if (N >= M)
                    diag_len -= k;
                else if (N + k > M)
                    diag_len = M.Value - k;
            }
            else
            {
                if (M >= N)
                    diag_len += k;
                else if (M - k > N)
                    diag_len = N + k;
            }

            var diagonal_ = array_ops.ones(new Shape(diag_len), dtype: dtype);
            var tensor = array_ops.matrix_diag(diagonal: diagonal_, num_rows: N, num_cols: M.Value, k: k);
            return new NDArray(tensor);
        }

        public NDArray frombuffer(byte[] bytes, string dtype)
        {
            if (dtype == ">u4")
            {
                var size = bytes.Length / sizeof(uint);
                var ints = new int[size];
                for (var index = 0; index < size; index++)
                    ints[index] = bytes[0] * 256 + bytes[1] + bytes[2] * 256 + bytes[3];

                return new NDArray(ints, shape: new Shape(size));
            }

            throw new NotImplementedException("");
        }

        public NDArray frombuffer(byte[] bytes, Shape shape, TF_DataType dtype)
        {
            return new NDArray(bytes, shape, dtype);
        }

        public NDArray linspace<T>(T start, T stop, int num = 50, bool endpoint = true, bool retstep = false,
            TF_DataType dtype = TF_DataType.TF_DOUBLE, int axis = 0)
        {
            var start_tensor = array_ops.constant(start, dtype: dtype);
            var stop_tensor = array_ops.constant(stop, dtype: dtype);

            // var step_tensor = array_ops.constant(np.nan);
            Tensor result = null;

            if (endpoint)
            {
                result = math_ops.linspace(start_tensor, stop_tensor, num, axis: axis);
            }
            else
            {
                if (num > 1)
                {
                    var step = (stop_tensor - start_tensor) / num;
                    var new_stop = math_ops.cast(stop_tensor, step.dtype) - step;
                    start_tensor = math_ops.cast(start_tensor, new_stop.dtype);
                    result = math_ops.linspace(start_tensor, new_stop, num, axis: axis);
                }
                else
                    result = math_ops.linspace(start_tensor, stop_tensor, num, axis: axis);
            }

            return new NDArray(result);
        }

        Array ReadValueMatrix(BinaryReader reader, Array matrix, int bytes, Type type, int[] shape)
        {
            int total = 1;
            for (int i = 0; i < shape.Length; i++)
                total *= shape[i];
            
            var buffer = reader.ReadBytes(bytes * total);
            System.Buffer.BlockCopy(buffer, 0, matrix, 0, buffer.Length);

            return matrix;
        }

        Array ReadObjectMatrix(BinaryReader reader, Array matrix, int[] shape)
        {
            Stream deflateStream = reader.BaseStream;
            BufferedStream bufferedStream = new BufferedStream(deflateStream);
            var unpickler = new Unpickler();
            return (MultiArrayPickleWarpper)unpickler.load(bufferedStream);
        }

        public (NDArray, NDArray) meshgrid<T>(T[] array, bool copy = true, bool sparse = false)
        {
            var tensors = array_ops.meshgrid(array, copy: copy, sparse: sparse);
            return (new NDArray(tensors[0]), new NDArray(tensors[1]));
        }
    }
}
