using Tensorflow.Keras;

namespace Tensorflow
{
    /// <summary>
    /// Deprecated, will use tf.keras
    /// </summary>
    public static class KerasApi
    {
        public static KerasInterface keras { get; } = KerasInterface.Instance;
    }
}
