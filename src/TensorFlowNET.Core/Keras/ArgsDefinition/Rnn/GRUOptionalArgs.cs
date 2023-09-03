using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class GRUOptionalArgs
    {
        public string Identifier => "GRU";

        public Tensor Mask { get; set; } = null;
    }
}
