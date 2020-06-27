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

        public Tensor ZerosLike(int[] shape = null, TF_DataType dtype = TF_DataType.TF_FLOAT)
            => tf.zeros(shape == null ? new int[0] : shape, dtype: dtype);

        public Tensor OnesLike(int[] shape = null, TF_DataType dtype = TF_DataType.TF_FLOAT)
            => tf.ones(shape == null ? new int[0] : shape, dtype: dtype);
    }
}
