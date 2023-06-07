using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Common.Extensions;

namespace Tensorflow.Common.Types
{
    public enum NestType
    {
        Empty,
        Node,
        List,
        Dictionary
    }

    /// <summary>
    /// A nested structure which may inclulde value, list and dictionary. 
    /// Note that dictionary does not ensure the data order. When using it as IEnumerable, 
    /// its order is depth-first.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class Nest<T> : INestStructure<T>, IEnumerable<T>
    {
        private static readonly Nest<T> _empty = new Nest<T>()
        {
            NestType = NestType.Empty,
        };
        public static Nest<T> Empty => _empty;
        public NestType NestType { get; protected set; }
        public string? Name { get; set; }
        public T? Value { get; protected set; }
        public List<Nest<T>>? ListValue { get; protected set; }
        public Dictionary<string, Nest<T>>? DictValue { get; protected set; }

        protected Nest() { }

        public Nest(T value, string? name = null)
        {
            Value = value;
            Name = name;
            NestType = NestType.Node;
        }

        public Nest(IEnumerable<Nest<T>> values, string? name = null)
        {
            ListValue = values.ToList();
            Name = name;
            NestType = NestType.List;
        }

        public Nest(Dictionary<string, Nest<T>> value, string? name = null)
        {
            DictValue = value;
            Name = name;
            NestType = NestType.Dictionary;
        }

        public Nest(Nest<T> other)
        {
            NestType = other.NestType;
            Value = other.Value;
            DictValue = other.DictValue;
            ListValue = other.ListValue;
            Name = other.Name;
        }

        public virtual IEnumerable<T> Flatten()
        {
            return FlattenInternal(this);
        }
        public virtual INestStructure<TOut> MapStructure<TOut>(Func<T, TOut> func)
        {
            return MapStructureInternal(func);
        }

        /// <summary>
        /// Pack the flat items to a nested sequence by the template.
        /// </summary>
        /// <param name="flatItems"></param>
        /// <returns></returns>
        public virtual Nest<T> PackSequence(T[] flatItems)
        {
            if(flatItems.Length == 0)
            {
                return Nest<T>.Empty;
            }
            int index = 0;
            return PackSequenceInternal(this, flatItems, ref index);
        }

        private static Nest<T> PackSequenceInternal(Nest<T> template, T[] flatItems, ref int index)
        {
            if(template.NestType == NestType.Node)
            {
                if(index >= flatItems.Length)
                {
                    throw new InvalidArgumentError("The template and flat items are not matched.");
                }
                return new Nest<T>(flatItems[index++]);
            }
            else if(template.NestType == NestType.List)
            {
                List<Nest<T>> nestedObjects = new List<Nest<T>>();
                for (int i = 0; i < template.ListValue!.Count; i++)
                {
                    nestedObjects.Add(PackSequenceInternal(template.ListValue![i], flatItems, ref index));
                }
                return new Nest<T>(nestedObjects);
            }
            else if(template.NestType == NestType.Node)
            {
                Dictionary<string, Nest<T>> dict = new Dictionary<string, Nest<T>>();
                foreach(var (key, value) in template.DictValue!)
                {
                    dict[key] = PackSequenceInternal(value, flatItems, ref index);
                }
                return new Nest<T>(dict);
            }
            // Consider Empty as invalid type.
            throw new InvalidArgumentError("When using `PackSequenceAs`, the template cannot contain empty node.");
        }

        public virtual Nest<T> AsNest()
        {
            return this;
        }

        public virtual Nest<T> MergeWith(Nest<T>? other)
        {
            if(other is null || other == Nest<T>.Empty)
            {
                return this;
            }
            if(this == Nest<T>.Empty)
            {
                return other;
            }
            if(NestType == NestType.Node && other.NestType == NestType.Node)
            {
                return new Nest<T>(new Nest<T>[] { this, other });
            }
            else if(NestType == NestType.List && other.NestType == NestType.List)
            {
                return new Nest<T>(this.ListValue!.Concat(other.ListValue!));
            }
            else if(NestType == NestType.Dictionary && other.NestType == NestType.Dictionary)
            {
                return new Nest<T>(this.DictValue!.Concat(other.DictValue!).ToDictionary(x => x.Key, x => x.Value));
            }
            else
            {
                return new Nest<T>(new Nest<T>[] { this, other });
            }
        }

        /// <summary>
        /// To see if the nested object is really nested. Despite being called `Nest`, sometimes it's actually not 
        /// nested. For example, [1, 2, 3] is not nested, while [1, [2, 3]] is nested.
        /// </summary>
        /// <returns></returns>
        public bool IsNested()
        {
            if(NestType is NestType.Empty or NestType.Node)
            {
                return false;
            }
            else if(NestType is NestType.List)
            {
                foreach(var item in ListValue!)
                {
                    if(item.NestType is NestType.List or NestType.Dictionary)
                    {
                        return true;
                    }
                }
                return false;
            }
            else
            {
                foreach (var item in DictValue!.Values)
                {
                    if (item.NestType is NestType.List or NestType.Dictionary)
                    {
                        return true;
                    }
                }
                return false;
            }
        }

        [Obsolete("The indexer of Tensors is not encouraged because it leads to unclear meanings.")]
        public T this[int index]
        {
            get
            {
                bool success = FindInternal(this, index, out var result);
                if (success)
                {
                    return result;
                }
                else
                {
                    throw new IndexOutOfRangeException();
                }
            }
            set
            {
                bool success = SetInternal(this, index, value);
                if (!success)
                {
                    throw new IndexOutOfRangeException();
                }
            }
        }

        /// <summary>
        /// If the existing nested structure if of type `Nest[INestStructure[T]]`, we can reduce it 
        /// to `Nest[T]`.
        /// </summary>
        /// <typeparam name="TOut"></typeparam>
        /// <param name="input"></param>
        /// <returns></returns>
        public static Nest<T> ReduceFrom<TOut>(INestStructure<TOut> input) where TOut: INestStructure<T>
        {
            var nested = input.AsNest();
            return ReduceInternal(nested);
        }

        private static Nest<T> ReduceInternal<TOut>(Nest<TOut> node) where TOut : INestStructure<T>
        {
            if(node.NestType == NestType.Empty)
            {
                return Nest<T>.Empty;
            }
            else if(node.NestType == NestType.Node)
            {
                return node.Value!.AsNest();
            }
            else if(node.NestType == NestType.List)
            {
                return new Nest<T>(node.ListValue!.Select(x => ReduceInternal(x)));
            }
            else // Dictionary type
            {
                return new Nest<T>(node.DictValue!.ToDictionary(x => x.Key, x => ReduceInternal(x.Value)));
            }
        }

        private static bool FindInternal(Nest<T> node, int index, out T? result)
        {
            if (node.NestType == NestType.Node)
            {
                if(index == 0)
                {
                    result = node.Value!;
                    return true;
                }
                result = default(T);
                return false;
            }
            else if (node.NestType == NestType.List)
            {
                foreach (var item in node.ListValue!)
                {
                    if(index == 0)
                    {
                        return FindInternal(item, index, out result);
                    }
                    index--;
                }
                result = default(T);
                return false;
            }
            else if(node.NestType == NestType.Dictionary)
            {
                foreach (var item in node.DictValue!.Values)
                {
                    if (index == 0)
                    {
                        return FindInternal(item, index, out result);
                    }
                    index--;
                }
                result = default(T);
                return false;
            }
            else
            {
                result = default(T);
                return false;
            }
        }

        private static bool SetInternal(Nest<T> node, int index, T newValue)
        {
            if (node.NestType == NestType.Node)
            {
                if (index == 0)
                {
                    node.Value = newValue;
                    return true;
                }
                return false;
            }
            else if (node.NestType == NestType.List)
            {
                foreach (var item in node.ListValue!)
                {
                    if (index == 0)
                    {
                        return SetInternal(item, index, newValue);
                    }
                    index--;
                }
                return false;
            }
            else if (node.NestType == NestType.Dictionary)
            {
                foreach (var item in node.DictValue!.Values)
                {
                    if (index == 0)
                    {
                        return SetInternal(item, index, newValue);
                    }
                    index--;
                }
                return false;
            }
            else
            {
                return false;
            }
        }

        private static IEnumerable<T> FlattenInternal(Nest<T> node)
        {
            if (node.NestType == NestType.Node)
            {
                yield return node.Value!;
            }
            else if (node.NestType == NestType.List)
            {
                foreach (var item in node.ListValue!)
                {
                    foreach(var val in FlattenInternal(item))
                    {
                        yield return val;
                    }
                }
            }
            else if (node.NestType == NestType.Dictionary)
            {
                foreach (var item in node.DictValue!.Values)
                {
                    foreach (var val in FlattenInternal(item))
                    {
                        yield return val;
                    }
                }
            }
        }

        private Nest<TOut> MapStructureInternal<TOut>(Func<T, TOut> func)
        {
            if (NestType == NestType.Node)
            {
                return new Nest<TOut>(func(Value!));
            }
            else if (NestType == NestType.List)
            {
                List<Nest<TOut>> outs = new List<Nest<TOut>>();
                foreach (var item in ListValue!)
                {
                    outs.Add(item.MapStructureInternal(func));
                }
                return new Nest<TOut>(outs);
            }
            else if (NestType == NestType.Dictionary)
            {
                Dictionary<string, Nest<TOut>> outs = new Dictionary<string, Nest<TOut>>();
                foreach (var (key, value) in DictValue!)
                {
                    outs.Add(key, value.MapStructureInternal(func));
                }
                return new Nest<TOut>(outs);
            }
            else
            {
                return Nest<TOut>.Empty;
            }
        }

        public IEnumerator<T> GetEnumerator()
        {
            return Flatten().GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append("(");
            WriteString(this, sb);
            sb.Append(")");
            return sb.ToString();
        }

        private static void WriteString(Nest<T> node,  StringBuilder sb)
        {
            if (!string.IsNullOrEmpty(node.Name))
            {
                sb.Append($"{node.Name}: ");
            }
            if (node.NestType == NestType.Node)
            {
                sb.Append(node.Value!.ToString());
            }
            else if (node.NestType == NestType.List)
            {
                sb.Append("[");
                for(int i = 0; i < node.ListValue!.Count; i++)
                {
                    WriteString(node.ListValue![i], sb);
                    if(i != node.ListValue!.Count - 1)
                    {
                        sb.Append(", ");
                    }
                }
                sb.Append("]");
            }
            else if (node.NestType == NestType.Dictionary)
            {
                sb.Append("{");
                int count = node.DictValue!.Count;
                int i = 0;
                foreach (var (key, value) in node.DictValue!)
                {
                    sb.Append($"{key}: ");
                    WriteString(value, sb);
                    if (i != count - 1)
                    {
                        sb.Append(", ");
                    }
                    i++;
                }
                sb.Append("}");
            }
            else
            {
                sb.Append("<empty>");
            }
        }
    }
}
