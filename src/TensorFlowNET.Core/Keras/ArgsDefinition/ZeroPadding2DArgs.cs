using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class ZeroPadding2DArgs : LayerArgs
    {
        public NDArray Padding { get; set; }
    }
}
