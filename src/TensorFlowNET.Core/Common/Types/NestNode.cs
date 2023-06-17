using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Common.Types
{
    /// <summary>
    /// A nested structure with only one element.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class NestNode<T> : INestStructure<T>
    {
        public NestType NestType => NestType.Node;
        public T Value { get; set; }
        public int ShallowNestedCount => 1;

        public int TotalNestedCount => 1;
        public NestNode(T value)
        {
            Value = value;
        }
        public IEnumerable<T> Flatten()
        {
            yield return Value;
        }
        public INestStructure<TOut> MapStructure<TOut>(Func<T, TOut> func)
        {
            return new NestNode<TOut>(func(Value));
        }

        public Nest<T> AsNest()
        {
            return new Nest<T>(Value);
        }
    }
}
