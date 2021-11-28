using Tensorflow.Util;

namespace Tensorflow.Gradients
{
    /// <summary>
    /// Map from operation-id to tape entry.
    /// </summary>
    public class OpTape : UnorderedMap<long, OpTapeEntry>
    {

    }
}
