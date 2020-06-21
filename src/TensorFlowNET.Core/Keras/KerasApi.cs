using System.Data;
using Tensorflow.Keras;
using Tensorflow.Keras.Datasets;

namespace Tensorflow
{
    public class KerasApi
    {
        public KerasDataset datasets { get; } = new KerasDataset();
        public Initializers initializers { get; } = new Initializers();
    }
}
