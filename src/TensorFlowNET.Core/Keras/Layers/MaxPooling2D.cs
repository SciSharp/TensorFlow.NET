using static Tensorflow.Binding;

namespace Tensorflow.Keras.Layers
{
    public class MaxPooling2D : Pooling2D
    {
        public MaxPooling2D(
            int[] pool_size,
            int[] strides,
            string padding = "valid",
            string data_format = null,
            string name = null) : base(tf.nn.max_pool_fn, pool_size,
                strides,
                padding: padding,
                data_format: data_format,
                name: name)
        {

        }
    }
}
