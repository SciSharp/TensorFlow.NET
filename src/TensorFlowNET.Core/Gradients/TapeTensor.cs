using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Eager;
using static Tensorflow.Binding;

namespace Tensorflow.Gradients
{
    public class TapeTensor
    {
        long id;
        TF_DataType dtype;
        TensorShape shape;

        public TapeTensor(long id, TF_DataType dtype, TensorShape shape)
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
    }
}
