using System;
using System.Collections.Generic;
using System.Linq;

namespace Tensorflow.Keras.Saving
{
    public class TensorShapeConfig
    {
        public string ClassName { get; set; }
        public int?[] Items { get; set; }

        public static implicit operator TensorShape(TensorShapeConfig shape)
            => new TensorShape(shape.Items.Select(x => x.HasValue ? x.Value : -1).ToArray());
    }
}
