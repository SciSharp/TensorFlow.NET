using Tensorflow.NumPy;

namespace Tensorflow.Eager
{
    public partial class EagerTensor
    {
        public override string ToString()
        {
            var nd = new NDArray(this);
            var str = NDArrayRender.ToString(nd);
            return $"tf.Tensor: shape={shape}, dtype={dtype.as_numpy_name()}, numpy={str}";
        }
            
    }
}
