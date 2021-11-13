using static Tensorflow.Binding;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class LayerNormalizationArgs : LayerArgs
    {
        public Axis Axis { get; set; } = -1;
        public float Epsilon { get; set; } = 1e-3f;
        public bool Center { get; set; } = true;
        public bool Scale { get; set; } = true;
        public IInitializer BetaInitializer { get; set; } = tf.zeros_initializer;
        public IInitializer GammaInitializer { get; set; } = tf.ones_initializer;
        public IRegularizer BetaRegularizer { get; set; }
        public IRegularizer GammaRegularizer { get; set; }
    }
}
