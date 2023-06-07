using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Common.Types
{
    /// <summary>
    /// The implementation of a list that support nest structure, in which the depth is 1.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public sealed class NestList<T> : INestStructure<T>, IEnumerable<T>
    {
        public List<T> Value { get; set; }
        public NestList(IEnumerable<T> values)
        {
            Value = new List<T>(values);
        }
        public IEnumerable<T> Flatten()
        {
            return Value;
        }
        public INestStructure<TOut> MapStructure<TOut>(Func<T, TOut> func)
        {
            return new NestList<TOut>(Value.Select(x => func(x)));
        }

        public Nest<T> AsNest()
        {
            return new Nest<T>(Value.Select(x => new Nest<T>(x)));
        }

        // Enumerator implementation
        public IEnumerator<T> GetEnumerator()
        {
            return Value.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}
