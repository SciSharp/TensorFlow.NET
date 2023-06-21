using Tensorflow.NumPy;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Common.Types;
using Tensorflow.Operations;
using Tensorflow.Common.Extensions;

namespace Tensorflow
{
    /// <summary>
    /// Tensors is used to represent a Tensor or a array of Tensor.
    /// It will simplify the API interface, it converts Tensor
    /// and Tensor[] to Tensors implicitily. And parse back to Tensor
    /// and Tensor[] from Tensors implicitily. 
    /// It works for tuple and scalar as well.
    /// </summary>
    public sealed class Tensors : Nest<Tensor>, IDisposable
    {
        public TF_DataType dtype => this.First().dtype; 
        public Shape shape => this.First().shape;
        public int rank => this.First().rank;
        public Graph graph => this.First().graph;
        public bool IsList { get; set; }
        public int Length => this.Count();
        /// <summary>
        /// Return a Tensor if `Tensors` has only one tensor, otherwise throw an exception.
        /// </summary>
        public Tensor Single
        {
            get
            {
                if (Length != 1)
                {
                    throw new ValueError("Tensors with more than one tensor cannot be " +
                        "implicitly converted to Tensor.");
                }
                return this.First();
            }
        }

        /// <summary>
        /// Return a Tensor if `Tensors` has only one tensor, and return null when `Tensors` is empty, 
        /// otherwise throw an exception.
        /// </summary>
        public Tensor? SingleOrNull
        {
            get
            {
                if (Length > 1)
                {
                    throw new ValueError($"Tensors with {Length} tensor cannot be " +
                        "implicitly converted to Tensor.");
                }
                return this.FirstOrDefault();
            }
        }

        public Tensor this[params string[] slices]
            => this.First()[slices];

        internal Tensors(Nest<Tensor> nested) : base(nested)
        {

        }

        public Tensors(params Tensor[] tensors): base(DealWithConstructorArrayInput(tensors))
        {
            
        }

        public Tensors(IList<Tensor> tensors) : base(tensors.Select(x => new Nest<Tensor>(x)))
        {

        }

        public Tensors(NDArray nd): base(ops.convert_to_tensor(nd))
        {
            
        }

        /// <summary>
        /// Get the element in shallow level. For example, for ts = [1, [2, 3], 4], 
        /// common indexer has ts[1] = 2. Shallow indexer has ts[1] = [2, 3]
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public Tensors GetShallow(int index)
        {
            if(NestType == NestType.Node)
            {
                if(index > 0)
                {
                    throw new IndexOutOfRangeException();
                }
                return this;
            }
            else if(NestType == NestType.List)
            {
                return ListValue![index].AsNest().ToTensors();
            }
            else
            {
                throw new NotImplementedException();
            }
        }

        private static Nest<Tensor> DealWithConstructorArrayInput(Tensor[] tensors)
        {
            if (tensors.Length == 0)
            {
                return Nest<Tensor>.Empty;
            }
            else if(tensors.Length == 1)
            {
                return new Nest<Tensor>(tensors[0]);
            }
            else
            {
                return new Nest<Tensor>(tensors.Select(x => new Nest<Tensor>(x)));
            }
        }

        public bool IsSingle()
        {
            return Length == 1;
        }

        public new Tensors MergeWith(Nest<Tensor>? other)
        {
            return FromNest(base.MergeWith(other));
        }

        [Obsolete("This method is not encouraged to be used. It may be removed in the future. If you do want to add " +
            "a tensor to `Tensors`, creating a new instance with your newly added tensor is a better choice.")]
        public void Add(Tensor tensor)
        {
            if(NestType == NestType.Dictionary)
            {
                throw new ValueError("Cannot add a tensor to dictionary type of nested tensors.");
            }
            else if(NestType == NestType.Node)
            {
                NestType = NestType.List;
                ListValue = new() { new Nest<Tensor>(NodeValue), new Nest<Tensor>(tensor) };
                NodeValue = null;
            }
            else if(NestType == NestType.List)
            {
                ListValue!.Add(new Nest<Tensor>(tensor));
            }
            else //Empty
            {
                NestType = NestType.Node;
                NodeValue = tensor;
            }
        }

        [Obsolete("This method is not encouraged to be used. It may be removed in the future. If you do want to add " +
            "some tensors to `Tensors`, creating a new instance with your newly added tensors is a better choice.")]
        public void AddRange(IEnumerable<Tensor> tensors)
        {
            if (NestType == NestType.Dictionary)
            {
                throw new ValueError("Cannot add a tensor to dictionary type of nested tensors.");
            }
            else if (NestType == NestType.Node)
            {
                NestType = NestType.List;
                ListValue = new() { new Nest<Tensor>(NodeValue) };
                ListValue.AddRange(tensors.Select(x => new Nest<Tensor>(x)));
                NodeValue = null;
            }
            else if(NestType == NestType.List)
            {
                ListValue!.AddRange(tensors.Select(x => new Nest<Tensor>(x)));
            }
            else // empty
            {
                NestType = NestType.List;
                ListValue = tensors.Select(x => new Nest<Tensor>(x) as INestStructure<Tensor>).ToList();
            }
        }

        [Obsolete("This method is not encouraged to be used. It may be removed in the future. If you do want to insert " +
            "a tensor to `Tensors`, creating a new instance with your newly added tensor is a better choice.")]
        public void Insert(int index, Tensor tensor)
        {
            if (NestType == NestType.List)
            {
                ListValue.Insert(index, new Nest<Tensor>(tensor));
            }
            else if(NestType == NestType.Node)
            {
                NestType = NestType.List;
                ListValue = new() { new Nest<Tensor>(NodeValue) };
                ListValue.Insert(index, new Nest<Tensor>(tensor));
                NodeValue = null;
            }
            else
            {
                throw new ValueError("Cannot add a tensor to dictionary type of nested tensors.");
            }
        }

        public string[] StringData()
        {
            return Single.StringData();
        }

        public string StringData(int index)
        {
            return Single.StringData(index);
        }

        public NDArray numpy()
        {
            return Single.numpy();
        }

        [Obsolete]
        public T[] ToArray<T>() where T: unmanaged
        {
            return Single.ToArray<T>();
        }

        #region Explicit Conversions
        public static explicit operator bool(Tensors tensor)
        {
            return (bool)tensor.Single;
        }

        public static explicit operator sbyte(Tensors tensor)
        {
            return (sbyte)tensor.Single;
        }

        public static explicit operator byte(Tensors tensor)
        {
            return (byte)tensor.Single;
        }

        public static explicit operator ushort(Tensors tensor)
        {
            return (ushort)tensor.Single;
        }

        public static explicit operator short(Tensors tensor)
        {
            return (short)tensor.Single;
        }

        public static explicit operator int(Tensors tensor)
        {
            return (int)tensor.Single;
        }

        public static explicit operator uint(Tensors tensor)
        {
            return (uint)tensor.Single;
        }

        public static explicit operator long(Tensors tensor)
        {
            return (long)tensor.Single;
        }

        public static explicit operator ulong(Tensors tensor)
        {
            return (ulong)tensor.Single;
        }

        public static explicit operator float(Tensors tensor)
        {
            return (byte)tensor.Single;
        }

        public static explicit operator double(Tensors tensor)
        {
            return (double)tensor.Single;
        }

        public static explicit operator string(Tensors tensor)
        {
            return (string)tensor.Single;
        }

        public static explicit operator object[](Tensors tensors)
            => tensors.Flatten().ToArray();
        #endregion

        #region Implicit Conversions
        public static implicit operator Tensors(Tensor tensor)
            => new Tensors(tensor);

        public static implicit operator Tensors((Tensor, Tensor) tuple)
            => new Tensors(tuple.Item1, tuple.Item2);

        [AutoNumPy]
        public static implicit operator Tensors(NDArray nd)
            => new Tensors(nd);

        public static implicit operator Tensors(Tensor[] tensors)
            => new Tensors(tensors);

        public static implicit operator Tensors(List<Tensor> tensors)
            => new Tensors(tensors.ToArray());

        public static implicit operator Tensor(Tensors? tensors)
            => tensors?.SingleOrNull;

        public static implicit operator Tensor[](Tensors tensors)
            => tensors.Flatten().ToArray();
        #endregion

        public static Tensors? FromNest(Nest<Tensor> nested)
        {
            if(nested == Nest<Tensor>.Empty)
            {
                return null;
            }
            return new Tensors(nested);
        }

        public void Deconstruct(out Tensor a, out Tensors? b)
        {
            a = this.First();
            b = Length == 1? null : new Tensors(this.Skip(1).ToArray());
        }

        public override string ToString()
        {
            if(Length == 1)
            {
                return this.First().ToString();
            }
            else
            {
                return $"Totally {Length} tensors: {base.ToString()}";
            }
        }

        public void Dispose()
        {
            foreach (var tensor in this)
                tensor.Dispose();
        }
    }
}
