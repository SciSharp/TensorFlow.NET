using Tensorflow;

namespace Keras
{
    public static class Keras
    {
        public static Tensor create_tensor(int[] shape, float mean = 0, float stddev = 1, TF_DataType dtype = TF_DataType.TF_FLOAT, int? seed = null, string name = null)
        {
            return tf.truncated_normal(shape: shape, mean: mean, stddev: stddev, dtype: dtype, seed: seed, name: name);
        }
    }
}
