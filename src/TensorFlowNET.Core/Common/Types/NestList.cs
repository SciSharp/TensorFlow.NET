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
        public NestType NestType => NestType.List;
        public List<T> Values { get; set; }
        public int ShallowNestedCount => Values.Count;

        public int TotalNestedCount => Values.Count;

        public NestList(params T[] values)
        {
            Values = new List<T>(values);
        }

        public NestList(IEnumerable<T> values)
        {
            Values = new List<T>(values);
        }
        public IEnumerable<T> Flatten()
        {
            return Values;
        }
        public INestStructure<TOut> MapStructure<TOut>(Func<T, TOut> func)
        {
            return new NestList<TOut>(Values.Select(x => func(x)));
        }

        public Nest<T> AsNest()
        {
            return new Nest<T>(Values.Select(x => new Nest<T>(x)));
        }

        // Enumerator implementation
        public IEnumerator<T> GetEnumerator()
        {
            return Values.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}
