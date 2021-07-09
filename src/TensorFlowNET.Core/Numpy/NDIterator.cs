using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.NumPy
{
    public delegate ref T MoveNextReferencedDelegate<T>() where T : unmanaged;

    public interface NDIterator : IEnumerable
    {
        IMemoryBlock Block { get; }
        IteratorType Type { get; }
        Shape Shape { get; } //TODO! is there a performance difference if this shape is readonly or not?
        Shape? BroadcastedShape { get; }
        bool AutoReset { get; }

        Func<T> MoveNext<T>() where T : unmanaged;
        MoveNextReferencedDelegate<T> MoveNextReference<T>() where T : unmanaged;

        Func<bool> HasNext { get; }
        Action Reset { get; }
    }
}
