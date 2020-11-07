namespace Tensorflow.Keras.ArgsDefinition
{
    public class RMSpropArgs : OptimizerV2Args
    {
        public float RHO { get; set; } = 0.9f;
        public float Momentum { get; set; } = 0.0f;
        public float Epsilon { get; set; } = 1e-7f;
        public bool Centered { get; set; } = false;
    }
}
