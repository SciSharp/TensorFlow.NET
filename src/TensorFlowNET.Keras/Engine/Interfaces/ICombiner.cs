using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Engine
{
    /// <summary>
    /// Functional object that defines a shardable computation.
    /// </summary>
    public interface ICombiner
    {
        void Compute(Tensor values, IAccumulator accumulator = null);
        void Merge();
        void Extract();
        IAccumulator Restore();
        void Serialize();
        void Deserialize();
    }
}
