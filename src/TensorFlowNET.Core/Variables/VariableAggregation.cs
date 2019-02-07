using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public enum VariableAggregation
    {
        NONE = 0,
        SUM = 1,
        MEAN = 2,
        ONLY_FIRST_REPLICA = 3 // ONLY_FIRST_TOWER
    }
}
