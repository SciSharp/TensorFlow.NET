using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using Tensorflow.Layers;

namespace Keras.Initializer
{
    class BaseInitializer : IInitializer
    {
        public int seed;
    }
}
