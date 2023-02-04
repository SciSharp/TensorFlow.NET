using Tensorflow.Keras.Saving;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class OptimizerV2Args: IKerasConfig
    {
        public string Name { get; set; }
        public float LearningRate { get; set; } = 0.001f;
        public float InitialDecay { get; set; }
        public float ClipNorm { get; set; }
        public float ClipValue { get; set; }
    }
}
