using static Tensorflow.Binding;

namespace Tensorflow.Gradients
{
    public class TapeTensor
    {
        long id;
        TF_DataType dtype;
        Shape shape;

        public TapeTensor(long id, TF_DataType dtype, Shape shape)
        {
            this.id = id;
            this.dtype = dtype;
            this.shape = shape;
        }

        public long GetID() => id;

        public Tensor ZerosLike()
            => tf.zeros(shape: shape, dtype: dtype);

        public Tensor OnesLike()
            => tf.ones(shape: shape, dtype: dtype);

        public override string ToString()
            => $"{id}, {shape}, {dtype.as_numpy_name()}";
    }
}
