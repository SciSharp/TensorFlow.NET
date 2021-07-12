using Tensorflow.NumPy;
using System;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public partial class Tensor
    {
        /// <summary>
        ///     
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public unsafe T[] ToArray<T>() where T : unmanaged
        {
            //Are the types matching?
            if (typeof(T).as_tf_dtype() != dtype)
                throw new ArrayTypeMismatchException($"dtype {dtype} mismatch.");

            if (NDims == 0 && size == 1)  //is it a scalar?
            {
                unsafe
                {
                    return new T[] { *(T*)buffer };
                }
            }

            //types match, no need to perform cast
            var ret = new T[size];
            var len = (long)(size * itemsize);
            var src = (T*)buffer;

            fixed (T* dst = ret)
                System.Buffer.MemoryCopy(src, dst, len, len);

            return ret;
        }

        /// <summary>
        /// Copy of the contents of this Tensor into a NumPy array or scalar.
        /// </summary>
        /// <returns>
        /// A NumPy array of the same shape and dtype or a NumPy scalar, if this
        /// Tensor has rank 0.
        /// </returns>
        public NDArray numpy()
            => GetNDArray(dtype);

        protected NDArray GetNDArray(TF_DataType dtype)
        {
            if (dtype == TF_DataType.TF_STRING)
            {
                var str= StringData();
                return new NDArray(str, new Shape(str.Length));
            }
                
            return new NDArray(this);
        }

        /// <summary>
        /// Copies the memory of current buffer onto newly allocated array.
        /// </summary>
        /// <returns></returns>
        public unsafe byte[] BufferToArray()
        {
            // ReSharper disable once LocalVariableHidesMember
            var data = new byte[bytesize];
            fixed (byte* dst = data)
                System.Buffer.MemoryCopy(buffer.ToPointer(), dst, bytesize, bytesize);

            return data;
        }
    }
}
