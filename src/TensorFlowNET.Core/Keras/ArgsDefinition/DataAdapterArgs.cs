using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Saving;
using Tensorflow.NumPy;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class DataAdapterArgs: IKerasConfig
    {
        public Tensors X { get; set; }
        public Tensors Y { get; set; }
        public IDatasetV2 Dataset { get; set; }
        public int BatchSize { get; set; } = 32;
        public int Steps { get; set; }
        public int Epochs { get; set; }
        public bool Shuffle { get; set; }
        public int MaxQueueSize { get; set; }
        public int Worker { get; set; }
        public bool UseMultiprocessing { get; set; }
        public IModel Model { get; set; }
        public Dictionary<int, float> ClassWeight = null;
        public NDArray SampleWeight = null;
    }
}
