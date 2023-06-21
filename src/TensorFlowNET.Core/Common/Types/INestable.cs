using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Common.Types
{
    public interface INestable<T>
    {
        Nest<T> AsNest();
    }
}
