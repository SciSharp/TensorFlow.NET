using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Operations
{
    public interface ICanBeFlattened
    {
        object[] Flatten();
    }
}
