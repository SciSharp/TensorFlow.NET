using Tensorflow.Keras;

namespace Tensorflow
{
    public static class KerasApi
    {
        public static KerasInterface Keras(this tensorflow tf)
            => new KerasInterface();

        public static KerasInterface keras { get; } = new KerasInterface();
    }
}
