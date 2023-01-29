using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras
{
    public interface IInitializersApi
    {
        IInitializer Orthogonal(float gain = 1.0f, int? seed = null);

        IInitializer HeNormal(int? seed = null);
    }
}
