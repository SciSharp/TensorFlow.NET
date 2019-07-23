using NumSharp;

namespace TensorFlowNET.Examples.Utility
{
    public interface IDataSet
    {
        NDArray data { get; }
        NDArray labels { get; }
    }
}
