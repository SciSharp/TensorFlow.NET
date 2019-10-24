using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Operations
{
    internal class LoopVar<TItem> : ICanBeFlattened
    {
        public Tensor Counter { get; }
        public TItem Item { get; }

        public LoopVar(Tensor counter, TItem item)
        {
            Counter = counter;
            Item = item;
        }

        public object[] Flatten()
        {
            var elements = new List<object> { Counter };
            if (typeof(TItem).GetInterface(typeof(ICanBeFlattened).Name) != null)
                elements.AddRange((Item as ICanBeFlattened).Flatten());
            else
                elements.Add(Item);
            return elements.ToArray();
        }

        public static implicit operator (Tensor, TItem)(LoopVar<TItem> loopVar)
        {
            return (loopVar.Counter, loopVar.Item);
        }
    }
}
