using System.IO;
using static Tensorflow.Binding;

namespace Tensorflow.NumPy
{
    public partial class np
    {
        [AutoNumPy]
        public static NDArray array(Array data, TF_DataType? dtype = null) 
        {
            var nd = new NDArray(data);
            return dtype == null ? nd : nd.astype(dtype.Value);
        }

        [AutoNumPy]
        public static NDArray array<T>(params T[] data)
            where T : unmanaged => new NDArray(data);

        [AutoNumPy]
        public static NDArray arange<T>(T end)
            where T : unmanaged => new NDArray(tf.range(default(T), limit: end));

        [AutoNumPy]
        public static NDArray arange<T>(T start, T? end = null, T? step = null)
            where T : unmanaged => new NDArray(tf.range(start, limit: end, delta: step));

        [AutoNumPy]
        public static NDArray empty(Shape shape, TF_DataType dtype = TF_DataType.TF_DOUBLE)
            => new NDArray(tf.zeros(shape, dtype: dtype));

        [AutoNumPy]
        public static NDArray eye(int N, int? M = null, int k = 0, TF_DataType dtype = TF_DataType.TF_DOUBLE)
            => tf.numpy.eye(N, M: M, k: k, dtype: dtype);

        [AutoNumPy]
        public static NDArray full<T>(Shape shape, T fill_value)
            where T : unmanaged => new NDArray(tf.fill(tf.constant(shape), fill_value));

        [AutoNumPy]
        public static NDArray full_like<T>(NDArray x, T fill_value, TF_DataType? dtype = null, Shape shape = null)
            where T : unmanaged => new NDArray(array_ops.fill(x.shape, constant_op.constant(fill_value)));

        [AutoNumPy]
        public static NDArray frombuffer(byte[] bytes, Shape shape, TF_DataType dtype)
            => tf.numpy.frombuffer(bytes, shape, dtype);

        [AutoNumPy]
        public static NDArray frombuffer(byte[] bytes, string dtype)
            => tf.numpy.frombuffer(bytes, dtype);

        [AutoNumPy]
        public static NDArray linspace<T>(T start, T stop, int num = 50, bool endpoint = true, bool retstep = false,
            TF_DataType dtype = TF_DataType.TF_DOUBLE, int axis = 0) 
            where T : unmanaged => tf.numpy.linspace(start, stop, 
                num: num, 
                endpoint: endpoint, 
                retstep: retstep, 
                dtype: dtype, 
                axis: axis);

        [AutoNumPy]
        public static NDArray load(string file) => tf.numpy.load(file);

        [AutoNumPy]
        public static T Load<T>(string path)
            where T : class, ICloneable, IList, ICollection, IEnumerable, IStructuralComparable, IStructuralEquatable
        {
            using (var stream = new FileStream(path, FileMode.Open))
                return Load<T>(stream);
        }

        [AutoNumPy]
        public static T Load<T>(Stream stream)
            where T : class, ICloneable, IList, ICollection, IEnumerable, IStructuralComparable, IStructuralEquatable
            => tf.numpy.Load<T>(stream);

        [AutoNumPy]
        public static Array LoadMatrix(Stream stream) => tf.numpy.LoadMatrix(stream);

        [AutoNumPy]
        public static NpzDictionary<T> Load_Npz<T>(byte[] bytes)
            where T : class, IList, ICloneable, ICollection, IEnumerable, IStructuralComparable, IStructuralEquatable
            => Load_Npz<T>(new MemoryStream(bytes));

        [AutoNumPy]
        public static NpzDictionary<T> Load_Npz<T>(Stream stream)
            where T : class, ICloneable, IList, ICollection, IEnumerable, IStructuralComparable, IStructuralEquatable
            => new NpzDictionary<T>(stream);

        [AutoNumPy]
        public static (NDArray, NDArray) meshgrid<T>(T x, T y, bool copy = true, bool sparse = false)
            => tf.numpy.meshgrid(new[] { x, y }, copy: copy, sparse: sparse);

        [AutoNumPy]
        public static NDArray ndarray(Shape shape, TF_DataType dtype = TF_DataType.TF_DOUBLE)
            => new NDArray(tf.zeros(shape, dtype: dtype));

        [AutoNumPy]
        public static NDArray ones(Shape shape, TF_DataType dtype = TF_DataType.TF_DOUBLE)
            => new NDArray(tf.ones(shape, dtype: dtype));

        [AutoNumPy]
        public static NDArray ones_like(NDArray a, TF_DataType dtype = TF_DataType.DtInvalid)
            => new NDArray(tf.ones_like(a, dtype: dtype));

        [AutoNumPy]
        public static NDArray zeros(Shape shape, TF_DataType dtype = TF_DataType.TF_DOUBLE)
            => new NDArray(tf.zeros(shape, dtype: dtype));

        [AutoNumPy]
        public static NDArray zeros_like(NDArray a, TF_DataType dtype = TF_DataType.DtInvalid)
            => new NDArray(tf.zeros_like(a, dtype: dtype));
    }
}
