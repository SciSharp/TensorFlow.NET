using Tensorflow.Util;

namespace Tensorflow.Gradients
{
    /// <summary>
    /// Map from operation-id to tape entry.
    /// </summary>
    /// <typeparam name="BackwardFunction"></typeparam>
    /// <typeparam name="TapeTensor"></typeparam>
    public class OpTape<BackwardFunction, TapeTensor> :
        UnorderedMap<long, OpTapeEntry<BackwardFunction, TapeTensor>>
    {

    }
}
