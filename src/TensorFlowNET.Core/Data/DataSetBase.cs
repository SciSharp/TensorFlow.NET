using Tensorflow.Numpy;

namespace Tensorflow
{
    public abstract class DataSetBase : IDataSet
    {
        public NDArray Data { get; protected set; }
        public NDArray Labels { get; protected set; }
    }
}
