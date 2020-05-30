using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public enum GCItemType
    {
        TensorHandle = 0,
        LocalTensorHandle = 1,
        EagerTensorHandle = 2
    }
}
