using Tensorflow.NumPy;

namespace Tensorflow
{
    public interface IDataSet
    {
        NDArray Data { get; }
        NDArray Labels { get; }
    }
}
