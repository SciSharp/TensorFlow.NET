using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.NumPy
{
    public partial class NDIterator<TOut> : NDIterator, IEnumerable<TOut>, IDisposable where TOut : unmanaged
    {
        public IMemoryBlock Block => throw new NotImplementedException();

        public IteratorType Type => throw new NotImplementedException();

        public Shape Shape => throw new NotImplementedException();

        public Shape BroadcastedShape => throw new NotImplementedException();

        public bool AutoReset => throw new NotImplementedException();

        public Func<bool> HasNext => throw new NotImplementedException();

        public Action Reset => throw new NotImplementedException();

        public void Dispose()
        {
            throw new NotImplementedException();
        }

        public IEnumerator GetEnumerator()
        {
            throw new NotImplementedException();
        }

        public Func<T> MoveNext<T>() where T : unmanaged
            => throw new NotImplementedException();

        public MoveNextReferencedDelegate<T> MoveNextReference<T>() where T : unmanaged
        {
            throw new NotImplementedException();
        }

        IEnumerator<TOut> IEnumerable<TOut>.GetEnumerator()
        {
            throw new NotImplementedException();
        }
    }
}
