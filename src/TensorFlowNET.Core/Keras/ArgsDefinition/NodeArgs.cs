using Tensorflow.Keras.Saving;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class NodeArgs: IKerasConfig
    {
        public ILayer[] InboundLayers { get; set; }
        public int[] NodeIndices { get; set; }
        public int[] TensorIndices { get; set; }
        public Tensors InputTensors { get; set; }
        public Tensors Outputs { get; set; }
    }
}
