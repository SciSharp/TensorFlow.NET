using static Tensorflow.Binding;

namespace Tensorflow.Gradients
{
    public class TapeTensor
    {
        Tensor tensor;
        long id => tensor.Id;
        TF_DataType dtype => tensor.dtype;
        Shape shape => tensor.shape;

        public TapeTensor(Tensor tensor)
        {
            this.tensor = tensor;
        }

        public long GetID() => tensor.Id;
        public Tensor GetTensor() => tensor;

        public Tensor ZerosLike()
            => tf.zeros(shape: shape, dtype: dtype);

        public Tensor OnesLike()
            => tf.ones(shape: shape, dtype: dtype);

        public override string ToString()
            => $"{id}, {shape}, {dtype.as_numpy_name()}";
    }
}
