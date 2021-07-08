using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow.Numpy
{
    public partial class NDArray
    {
        Tensor _tensor;

        public NumpyDType dtype => _tensor.dtype.as_numpy_typecode();
        public ulong size => _tensor.size;
        public ulong dtypesize => _tensor.itemsize;
        public int ndim => _tensor.NDims;
        public long[] dims => _tensor.dims.Select(x => Convert.ToInt64(x)).ToArray();
        public Shape shape => _tensor.shape; 

        public NDArray(bool value)
        {

        }

        public NDArray(float value)
        {

        }

        public NDArray(double value)
        {

        }

        public NDArray(Array value, Shape shape = null)
        {

        }

        public NDArray(Type dtype, Shape shape)
        {

        }

        public NDArray(Shape shape, NumpyDType dtype = NumpyDType.Float)
        {
            Initialize(shape, dtype: dtype);
        }

        public NDArray(Tensor value)
        {
            _tensor = value;
        }

        public NDArray this[params int[] index]
        {
            get
            {
                throw new NotImplementedException("");
            }

            set
            {

            }
        }

        public NDArray this[params Slice[] slices]
        {
            get
            {
                throw new NotImplementedException("");
            }

            set
            {

            }
        }

        public NDArray this[NDArray mask]
        {
            get
            {
                throw new NotImplementedException("");
            }

            set
            {

            }
        }

        public static NDArray Scalar<T>(T value) where T : unmanaged
        {
            return value switch
            {
                bool b => new NDArray(b),
                _ => throw new NotImplementedException("")
            };
        }

        public T GetValue<T>(int index) where T : unmanaged
            => _tensor.ToArray<T>()[index];
        public T GetAtIndex<T>(int index) where T : unmanaged
            => _tensor.ToArray<T>()[index]; 
        public T[] GetData<T>() where T : unmanaged
            => _tensor.ToArray<T>();

        public NDArray[] GetNDArrays()
            => throw new NotImplementedException("");

        public ValueType GetValue(params int[] indices)
            => throw new NotImplementedException("");

        public void SetData(object value, params int[] indices)
            => throw new NotImplementedException("");

        public NDIterator<T> AsIterator<T>(bool autoreset = false) where T : unmanaged
            => throw new NotImplementedException("");

        public bool HasNext() => throw new NotImplementedException("");
        public T MoveNext<T>() => throw new NotImplementedException("");
        public NDArray reshape(params int[] shape) => throw new NotImplementedException("");
        public NDArray reshape(params long[] shape) => throw new NotImplementedException("");
        public NDArray astype(Type type) => throw new NotImplementedException("");
        public NDArray astype(NumpyDType type) => throw new NotImplementedException("");
        public bool array_equal(NDArray rhs) => throw new NotImplementedException("");
        public NDArray ravel() => throw new NotImplementedException("");
        public void shuffle(NDArray nd) => throw new NotImplementedException("");
        public Array ToMuliDimArray<T>() => throw new NotImplementedException("");
        public byte[] ToByteArray() => _tensor.ToArray<byte>();
        public static string[] AsStringArray(NDArray arr) => throw new NotImplementedException("");

        public T[] Data<T>() where T : unmanaged
            => _tensor.ToArray<T>();
        public T[] ToArray<T>() where T : unmanaged
            => _tensor.ToArray<T>();

        public static implicit operator NDArray(Array array)
            => new NDArray(array);

        public static implicit operator bool(NDArray nd)
            => nd._tensor.ToArray<bool>()[0];

        public static implicit operator int(NDArray nd)
            => nd._tensor.ToArray<int>()[0];

        public static implicit operator NDArray(bool value)
            => new NDArray(value);

        public static implicit operator NDArray(float value)
            => new NDArray(value);

        public static implicit operator NDArray(double value)
            => new NDArray(value);

        public static implicit operator NDArray(byte[] value)
            => new NDArray(value);

        public static implicit operator byte[](NDArray nd)
            => nd.ToByteArray();

        public static implicit operator NDArray(int[] value)
            => new NDArray(value, new Shape(value.Length));

        public static implicit operator NDArray(float[] value)
            => new NDArray(value);

        public static NDArray operator /(NDArray x, NDArray y) => throw new NotImplementedException("");

        public override string ToString()
        {
            return tensor_util.to_numpy_string(_tensor);
        }
    }
}
