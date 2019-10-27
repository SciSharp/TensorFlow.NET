using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public interface IPackable<T>
    {
        T Pack(object[] sequences);
    }
}
