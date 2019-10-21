using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Operations
{
    internal class LoopVar<TItem>
    {
        public Tensor Counter { get; }
        public TItem[] Items { get; }
        public TItem Item { get; }

        public LoopVar(Tensor counter, TItem[] items)
        {
            Counter = counter;
            Items = items;
        }

        public LoopVar(Tensor counter, TItem item)
        {
            Counter = counter;
            Item = item;
        }
    }
}
