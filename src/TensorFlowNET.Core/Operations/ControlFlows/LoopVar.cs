using System.Collections.Generic;
using System.Linq;

namespace Tensorflow.Operations
{
    internal class LoopVar<TItem> : ICanBeFlattened, IPackable<LoopVar<TItem>>
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

        public LoopVar<TItem> Pack(object[] sequences)
        {
            var counter = sequences[0] as Tensor;
            var item = default(TItem);
            if (typeof(TItem).GetInterface(typeof(IPackable<TItem>).Name) != null)
                item = (Item as IPackable<TItem>).Pack(sequences.Skip(1).ToArray());
            return new LoopVar<TItem>(counter, item);
        }

        public static implicit operator (Tensor, TItem)(LoopVar<TItem> loopVar)
        {
            return (loopVar.Counter, loopVar.Item);
        }
    }
}
