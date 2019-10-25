using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow.Operations
{
    internal class LoopVar<TItem> : ICanBeFlattened, IPackable
    {
        public Tensor Counter { get; set; }
        public TItem Item { get; set; }

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

        public void Pack(object[] sequences)
        {
            Counter = sequences[0] as Tensor;
            if (typeof(TItem).GetInterface(typeof(IPackable).Name) != null)
                (Item as IPackable).Pack(sequences.Skip(1).ToArray());
        }

        public static implicit operator (Tensor, TItem)(LoopVar<TItem> loopVar)
        {
            return (loopVar.Counter, loopVar.Item);
        }
    }
}
