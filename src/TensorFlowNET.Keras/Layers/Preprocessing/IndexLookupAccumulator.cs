using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Engine;

namespace Tensorflow.Keras.Layers
{
    public class IndexLookupAccumulator : IAccumulator
    {
        public Dictionary<string, int> CountDict { get; set; }
        public IndexLookupAccumulator()
        {
            CountDict = new Dictionary<string, int>();
        }
    }
}
