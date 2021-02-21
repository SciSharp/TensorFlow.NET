using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras.Layers
{
    /// <summary>
    /// Combiner for the IndexLookup preprocessing layer.
    /// </summary>
    public class IndexLookupCombiner : ICombiner
    {
        int _vocab_size;
        string _mask_value;

        public IndexLookupCombiner(int vocab_size = -1, string mask_value = null)
        {
            _vocab_size = vocab_size;
            _mask_value = mask_value;
        }

        public void Compute(Tensor values, IAccumulator accumulator = null)
        {
            if(accumulator == null)
            {
                accumulator = new IndexLookupAccumulator();
            }
        }

        public void Deserialize()
        {
            throw new NotImplementedException();
        }

        public void Extract()
        {
            throw new NotImplementedException();
        }

        public void Merge()
        {
            throw new NotImplementedException();
        }

        public IAccumulator Restore()
        {
            throw new NotImplementedException();
        }

        public void Serialize()
        {
            throw new NotImplementedException();
        }
    }
}
