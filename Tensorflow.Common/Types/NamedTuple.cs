using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;

namespace Tensorflow.Common.Types
{
    public class NamedTuple
    {
        public string Name { get; set; }
        public Dictionary<string, object> ValueDict { get; set; }
    }
}
