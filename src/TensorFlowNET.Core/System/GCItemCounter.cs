using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class GCItemCounter
    {
        public GCItemType ItemType { get; set; }
        public int RefCounter { get; set; }
        public DateTime LastUpdateTime { get; set; }
        public IntPtr Handle { get; set; }

        public override string ToString()
            => $"{ItemType} {RefCounter} {LastUpdateTime}";
    }
}
