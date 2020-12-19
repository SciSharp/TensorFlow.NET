using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class DataHandlerArgs
    {
        public Tensor X { get; set; }
        public Tensor Y { get; set; }
        public IDatasetV2 Dataset { get; set; }
        public int BatchSize { get; set; } = 32;
        public int StepsPerEpoch { get; set; } = -1;
        public int InitialEpoch { get; set; } = 0;
        public int Epochs { get; set; } = 1;
        public bool Shuffle { get; set; } = false;
        public int MaxQueueSize { get; set; } = 10;
        public int Workers { get; set; } = 1;
        public bool UseMultiprocessing { get; set; } = false;
        public IModel Model { get; set; }
        public IVariableV1 StepsPerExecution { get; set; }
    }
}
