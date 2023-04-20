using OneOf;
using static Tensorflow.Binding;

namespace Tensorflow.Gradients
{
    public class TapeTensor
    {
        internal Tensor tensor;
        internal long id;
        internal TF_DataType dtype;
        internal OneOf<Shape, Tensor> shape;

        public TapeTensor(long id, TF_DataType dtype, Shape shape)
        {
            this.id = id;
            this.dtype = dtype;
            this.shape = shape;
        }

        public TapeTensor(long id, TF_DataType dtype, Tensor shape)
        {
            this.id = id;
            this.dtype = dtype;
            this.shape = shape;
        }

        public TapeTensor(Tensor tensor)
        {
            this.id = tensor.Id;
            this.dtype = tensor.dtype;
            this.shape = tensor.shape;
            this.tensor = tensor;
        }

        public long GetID() => id;

        public Tensor ZerosLike()
        {
            if(dtype == dtypes.resource)
            {
                return null;
            }
            if(shape.Index == 1)
            {
                return tf.zeros_like(shape.AsT1);
            }
            return tf.zeros(shape.AsT0, dtype);
        }

        public Tensor OnesLike()
        {
            if (shape.Index == 1)
            {
                return tf.ones_like(shape.AsT1);
            }
            return tf.ones(shape.AsT0, dtype);
        }

        //public Tensor OnesLike()
        //    => tf.ones(shape: shape, dtype: dtype);

        public override string ToString()
            => $"{id}, {shape}, {dtype.as_numpy_name()}";
    }
}
