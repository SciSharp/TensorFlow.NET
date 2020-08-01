using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class NodeArgs
    {
        public Layer[] InboundLayers { get; set; }
        public int[] NodeIndices { get; set; }
        public int[] TensorIndices { get; set; }
        public Tensor[] InputTensors { get; set; }
        public Tensor[] Outputs { get; set; }
    }
}
