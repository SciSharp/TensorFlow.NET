using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// Indicates when a distributed variable will be synced.
    /// </summary>
    public enum VariableSynchronization
    {
        AUTO = 0,
        NONE = 1,
        ON_WRITE = 2,
        ON_READ = 3
    }
}
