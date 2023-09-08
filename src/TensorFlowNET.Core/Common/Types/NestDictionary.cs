using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Common.Types
{
    public class NestDictionary<TKey, TValue> : INestStructure<TValue>, IDictionary<TKey, TValue> where TKey : notnull
    {
        public NestType NestType => NestType.Dictionary;
        public IDictionary<TKey, TValue> Value { get; set; }
        public int ShallowNestedCount => Values.Count;

        public int TotalNestedCount => Values.Count;
        public NestDictionary(IDictionary<TKey, TValue> dict)
        {
            Value = dict;
        }
        public IEnumerable<TValue> Flatten()
        {
            return Value.Select(x => x.Value);
        }
        public INestStructure<TOut> MapStructure<TOut>(Func<TValue, TOut> func)
        {
            return new NestList<TOut>(Value.Select(x => func(x.Value)));
        }

        public Nest<TValue> AsNest()
        {
            return new Nest<TValue>(Value.Values.Select(x => new Nest<TValue>(x)));
        }

        // Required IDictionary<TKey, TValue> members
        public int Count => Value.Count;

        public bool IsReadOnly => Value.IsReadOnly;

        public ICollection<TKey> Keys => Value.Keys;

        public ICollection<TValue> Values => Value.Values;

        public void Add(TKey key, TValue value)
        {
            Value.Add(key, value);
        }

        public void Add(KeyValuePair<TKey, TValue> item)
        {
            Value.Add(item);
        }

        public void Clear()
        {
            Value.Clear();
        }

        public bool Contains(KeyValuePair<TKey, TValue> item)
        {
            return Value.Contains(item);
        }

        public bool ContainsKey(TKey key)
        {
            return Value.ContainsKey(key);
        }

        public void CopyTo(KeyValuePair<TKey, TValue>[] array, int arrayIndex)
        {
            Value.CopyTo(array, arrayIndex);
        }

        public IEnumerator<KeyValuePair<TKey, TValue>> GetEnumerator()
        {
            return Value.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public bool Remove(TKey key)
        {
            return Value.Remove(key);
        }

        public bool Remove(KeyValuePair<TKey, TValue> item)
        {
            return Value.Remove(item);
        }

        public bool TryGetValue(TKey key, out TValue value)
        {
            return Value.TryGetValue(key, out value);
        }

        // Optional IDictionary<TKey, TValue> members
        public TValue this[TKey key]
        {
            get => Value[key];
            set => Value[key] = value;
        }
    }
}
