using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Common.Types;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class RnnOptionalArgs: IOptionalArgs
    {
        public string Identifier => "Rnn";
        public Tensor Mask { get; set; } = null;
        public Tensors Constants { get; set; } = null;
    }
}
