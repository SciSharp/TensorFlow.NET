namespace Tensorflow.Eager
{
    public partial class EagerTensor
    {
        public override string ToString()
            => $"tf.Tensor: shape={shape}, dtype={dtype.as_numpy_name()}, numpy={tensor_util.to_numpy_string(this)}";
    }
}
