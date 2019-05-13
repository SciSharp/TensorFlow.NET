using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using Tensorflow.Layers;

namespace Makina.Initializer
{
    class BaseInitializer : IInitializer
    {
        public int seed;
    }
}
