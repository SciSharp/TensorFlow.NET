using System;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using FluentAssertions;
using FluentAssertions.Execution;
using FluentAssertions.Primitives;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Utilities;

namespace TensorFlowNET.UnitTest
{
    [DebuggerStepThrough]
    public static class FluentExtension
    {
        public static ShapeAssertions Should(this Shape shape)
        {
            return new ShapeAssertions(shape);
        }

        public static NDArrayAssertions Should(this NDArray arr)
        {
            return new NDArrayAssertions(arr);
        }

        public static NDArrayAssertions Should(this UnmanagedStorage arr)
        {
            return new NDArrayAssertions(arr);
        }

        public static string ToString(this Array arr, bool flat)
        {
            return new NDArray(arr).ToString(flat);
        }
    }

    [DebuggerStepThrough]
    public class ShapeAssertions : ReferenceTypeAssertions<Shape, ShapeAssertions>
    {
        public ShapeAssertions(Shape instance)
        {
            Subject = instance;
        }

        protected override string Identifier => "shape";

        public AndConstraint<ShapeAssertions> BeOfSize(int size, string because = null, params object[] becauseArgs)
        {
            Subject.Size.Should().Be(size, because, becauseArgs);
            return new AndConstraint<ShapeAssertions>(this);
        }

        public AndConstraint<ShapeAssertions> NotBeOfSize(int size, string because = null, params object[] becauseArgs)
        {
            Subject.Size.Should().NotBe(size, because, becauseArgs);
            return new AndConstraint<ShapeAssertions>(this);
        }

        public AndConstraint<ShapeAssertions> BeShaped(params int[] dimensions)
        {
            if (dimensions == null)
                throw new ArgumentNullException(nameof(dimensions));

            if (dimensions.Length == 0)
                throw new ArgumentException("Value cannot be an empty collection.", nameof(dimensions));

            Subject.Dimensions.Should().BeEquivalentTo(dimensions);
            return new AndConstraint<ShapeAssertions>(this);
        }

        public AndConstraint<ShapeAssertions> Be(Shape shape, string because = null, params object[] becauseArgs)
        {
            Execute.Assertion
                .BecauseOf(because, becauseArgs)
                .ForCondition(Subject.Equals(shape))
                .FailWith($"Expected shape to be {shape.ToString()} but got {Subject.ToString()}");

            return new AndConstraint<ShapeAssertions>(this);
        }

        public AndConstraint<ShapeAssertions> BeEquivalentTo(int? size = null, int? ndim = null, ITuple shape = null)
        {
            if (size.HasValue)
            {
                BeOfSize(size.Value, null);
            }

            if (ndim.HasValue)
                HaveNDim(ndim.Value);

            if (shape != null)
                for (int i = 0; i < shape.Length; i++)
                {
                    Subject.Dimensions[i].Should().Be((int) shape[i]);
                }

            return new AndConstraint<ShapeAssertions>(this);
        }

        public AndConstraint<ShapeAssertions> NotBe(Shape shape, string because = null, params object[] becauseArgs)
        {
            Execute.Assertion
                .BecauseOf(because, becauseArgs)
                .ForCondition(!Subject.Equals(shape))
                .FailWith($"Expected shape to be {shape.ToString()} but got {Subject.ToString()}");

            return new AndConstraint<ShapeAssertions>(this);
        }

        public AndConstraint<ShapeAssertions> HaveNDim(int ndim)
        {
            Subject.Dimensions.Length.Should().Be(ndim);
            return new AndConstraint<ShapeAssertions>(this);
        }

        public AndConstraint<ShapeAssertions> BeSliced()
        {
            Subject.IsSliced.Should().BeTrue();
            return new AndConstraint<ShapeAssertions>(this);
        }

        public AndConstraint<ShapeAssertions> BeScalar()
        {
            Subject.IsScalar.Should().BeTrue();
            return new AndConstraint<ShapeAssertions>(this);
        }

        public AndConstraint<ShapeAssertions> BeBroadcasted()
        {
            Subject.IsBroadcasted.Should().BeTrue();
            return new AndConstraint<ShapeAssertions>(this);
        }


        public AndConstraint<ShapeAssertions> NotBeSliced()
        {
            Subject.IsSliced.Should().BeFalse();
            return new AndConstraint<ShapeAssertions>(this);
        }

        public AndConstraint<ShapeAssertions> NotBeScalar()
        {
            Subject.IsScalar.Should().BeFalse();
            return new AndConstraint<ShapeAssertions>(this);
        }

        public AndConstraint<ShapeAssertions> NotBeBroadcasted()
        {
            Subject.IsBroadcasted.Should().BeFalse();
            return new AndConstraint<ShapeAssertions>(this);
        }

        public AndConstraint<ShapeAssertions> BeNDim(int ndim)
        {
            Subject.Dimensions.Length.Should().Be(ndim);
            return new AndConstraint<ShapeAssertions>(this);
        }
    }

    //[DebuggerStepThrough]
    public class NDArrayAssertions : ReferenceTypeAssertions<NDArray, NDArrayAssertions>
    {
        public NDArrayAssertions(NDArray instance)
        {
            Subject = instance;
        }

        public NDArrayAssertions(UnmanagedStorage instance)
        {
            Subject = new NDArray(instance);
        }

        protected override string Identifier => "shape";

        public AndConstraint<NDArrayAssertions> BeOfSize(int size, string because = null, params object[] becauseArgs)
        {
            Subject.size.Should().Be(size, because, becauseArgs);
            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> BeShaped(params int[] dimensions)
        {
            if (dimensions == null)
                throw new ArgumentNullException(nameof(dimensions));

            if (dimensions.Length == 0)
                throw new ArgumentException("Value cannot be an empty collection.", nameof(dimensions));

            Subject.Unsafe.Storage.Shape.Dimensions.Should().BeEquivalentTo(dimensions);
            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> BeShaped(int? size = null, int? ndim = null, ITuple shape = null)
        {
            if (size.HasValue)
            {
                BeOfSize(size.Value, null);
            }

            if (ndim.HasValue)
                HaveNDim(ndim.Value);

            if (shape != null)
                for (int i = 0; i < shape.Length; i++)
                {
                    Subject.Unsafe.Storage.Shape.Dimensions[i].Should().Be((int) shape[i]);
                }

            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> NotBeShaped(Shape shape, string because = null, params object[] becauseArgs)
        {
            Execute.Assertion
                .BecauseOf(because, becauseArgs)
                .ForCondition(!Subject.Unsafe.Storage.Shape.Equals(shape))
                .FailWith($"Expected shape to be {shape.ToString()} but got {Subject.ToString()}");

            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> HaveNDim(int ndim)
        {
            Subject.Unsafe.Storage.Shape.Dimensions.Length.Should().Be(ndim);
            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> BeBroadcasted()
        {
            Subject.Unsafe.Storage.Shape.IsBroadcasted.Should().BeTrue();
            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> NotBeBroadcasted()
        {
            Subject.Unsafe.Storage.Shape.IsBroadcasted.Should().BeFalse();
            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> BeSliced()
        {
            Subject.Unsafe.Storage.Shape.IsSliced.Should().BeTrue();
            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> BeScalar()
        {
            Subject.Unsafe.Storage.Shape.IsScalar.Should().BeTrue();
            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> BeScalar(object value)
        {
            Subject.Unsafe.Storage.Shape.IsScalar.Should().BeTrue();
            Subject.GetValue().Should().Be(value);
            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> BeOfType(NPTypeCode typeCode)
        {
            Subject.typecode.Should().Be(typeCode);
            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> BeOfType(Type typeCode)
        {
            Subject.dtype.Should().Be(typeCode);
            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> BeOfType<T>()
        {
            Subject.typecode.Should().Be(InfoOf<T>.NPTypeCode);
            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> NotBeSliced()
        {
            Subject.Unsafe.Storage.Shape.IsSliced.Should().BeFalse();
            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> NotBeScalar()
        {
            Subject.Unsafe.Storage.Shape.IsScalar.Should().BeFalse();
            return new AndConstraint<NDArrayAssertions>(this);
        }


        public AndConstraint<NDArrayAssertions> BeNDim(int ndim)
        {
            Subject.Unsafe.Storage.Shape.Dimensions.Length.Should().Be(ndim);
            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> Be(NDArray expected)
        {
            Execute.Assertion
                .ForCondition(np.array_equal(Subject, expected))
                .FailWith($"Expected the subject and other ndarray to be equals.\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n{expected.ToString(false)}");

            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> BeOfValues(params object[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            Subject.size.Should().Be(values.Length, "the method BeOfValues also confirms the sizes are matching with given values.");

#if _REGEN
            #region Compute
		    switch (Subject.typecode)
		    {
			    %foreach supported_dtypes,supported_dtypes_lowercase%
			    case NPTypeCode.#1: 
                {
                    var iter = Subject.AsIterator<#2>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");
                        
                        var expected = Convert.To#1(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith($"Expected NDArray's {{2}}th value to be {{0}}, but found {{1}} (dtype: #1).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n[{string.Join(", ", values.Select(v => v.ToString()))}]", expected, nextval, i);
                    }
                    break;
                }
			    %
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#else

            #region Compute

            switch (Subject.typecode)
            {
                case NPTypeCode.Boolean:
                {
                    var iter = Subject.AsIterator<bool>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToBoolean(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith($"Expected NDArray's {{2}}th value to be {{0}}, but found {{1}} (dtype: Boolean).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n[{string.Join(", ", values.Select(v => v.ToString()))}]", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Byte:
                {
                    var iter = Subject.AsIterator<byte>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToByte(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith($"Expected NDArray's {{2}}th value to be {{0}}, but found {{1}} (dtype: Byte).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n[{string.Join(", ", values.Select(v => v.ToString()))}]", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Int16:
                {
                    var iter = Subject.AsIterator<short>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToInt16(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith($"Expected NDArray's {{2}}th value to be {{0}}, but found {{1}} (dtype: Int16).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n[{string.Join(", ", values.Select(v => v.ToString()))}]", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.UInt16:
                {
                    var iter = Subject.AsIterator<ushort>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToUInt16(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith($"Expected NDArray's {{2}}th value to be {{0}}, but found {{1}} (dtype: UInt16).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n[{string.Join(", ", values.Select(v => v.ToString()))}]", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Int32:
                {
                    var iter = Subject.AsIterator<int>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToInt32(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith($"Expected NDArray's {{2}}th value to be {{0}}, but found {{1}} (dtype: Int32).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n[{string.Join(", ", values.Select(v => v.ToString()))}]", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.UInt32:
                {
                    var iter = Subject.AsIterator<uint>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToUInt32(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith($"Expected NDArray's {{2}}th value to be {{0}}, but found {{1}} (dtype: UInt32).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n[{string.Join(", ", values.Select(v => v.ToString()))}]", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Int64:
                {
                    var iter = Subject.AsIterator<long>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToInt64(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith($"Expected NDArray's {{2}}th value to be {{0}}, but found {{1}} (dtype: Int64).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n[{string.Join(", ", values.Select(v => v.ToString()))}]", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.UInt64:
                {
                    var iter = Subject.AsIterator<ulong>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToUInt64(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith($"Expected NDArray's {{2}}th value to be {{0}}, but found {{1}} (dtype: UInt64).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n[{string.Join(", ", values.Select(v => v.ToString()))}]", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Char:
                {
                    var iter = Subject.AsIterator<char>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToChar(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith($"Expected NDArray's {{2}}th value to be {{0}}, but found {{1}} (dtype: Char).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n[{string.Join(", ", values.Select(v => v.ToString()))}]", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Double:
                {
                    var iter = Subject.AsIterator<double>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToDouble(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith($"Expected NDArray's {{2}}th value to be {{0}}, but found {{1}} (dtype: Double).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n[{string.Join(", ", values.Select(v => v.ToString()))}]", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Single:
                {
                    var iter = Subject.AsIterator<float>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToSingle(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith($"Expected NDArray's {{2}}th value to be {{0}}, but found {{1}} (dtype: Single).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n[{string.Join(", ", values.Select(v => v.ToString()))}]", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Decimal:
                {
                    var iter = Subject.AsIterator<decimal>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToDecimal(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith($"Expected NDArray's {{2}}th value to be {{0}}, but found {{1}} (dtype: Decimal).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n[{string.Join(", ", values.Select(v => v.ToString()))}]", expected, nextval, i);
                    }

                    break;
                }

                default:
                    throw new NotSupportedException();
            }

            #endregion

#endif


            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> AllValuesBe(object val)
        {
#if _REGEN
            #region Compute
		    switch (Subject.typecode)
		    {
			    %foreach supported_dtypes,supported_dtypes_lowercase%
			    case NPTypeCode.#1: 
                {
                    var iter = Subject.AsIterator<#2>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    var expected = Convert.To#1(val);
                    for (int i = 0; hasnext(); i++)
                    {
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith($"Expected NDArray's {2}th value to be {0}, but found {1} (dtype: #1).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n{val}", expected, nextval, i);
                    }
                    break;
                }
			    %
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#else

            #region Compute

            switch (Subject.typecode)
            {
                case NPTypeCode.Boolean:
                {
                    var iter = Subject.AsIterator<bool>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    var expected = Convert.ToBoolean(val);
                    for (int i = 0; hasnext(); i++)
                    {
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith($"Expected NDArray's {2}th value to be {0}, but found {1} (dtype: Boolean).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n{val}", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Byte:
                {
                    var iter = Subject.AsIterator<byte>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    var expected = Convert.ToByte(val);
                    for (int i = 0; hasnext(); i++)
                    {
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith($"Expected NDArray's {2}th value to be {0}, but found {1} (dtype: Byte).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n{val}", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Int16:
                {
                    var iter = Subject.AsIterator<short>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    var expected = Convert.ToInt16(val);
                    for (int i = 0; hasnext(); i++)
                    {
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith($"Expected NDArray's {2}th value to be {0}, but found {1} (dtype: Int16).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n{val}", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.UInt16:
                {
                    var iter = Subject.AsIterator<ushort>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    var expected = Convert.ToUInt16(val);
                    for (int i = 0; hasnext(); i++)
                    {
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith($"Expected NDArray's {2}th value to be {0}, but found {1} (dtype: UInt16).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n{val}", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Int32:
                {
                    var iter = Subject.AsIterator<int>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    var expected = Convert.ToInt32(val);
                    for (int i = 0; hasnext(); i++)
                    {
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith($"Expected NDArray's {2}th value to be {0}, but found {1} (dtype: Int32).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n{val}", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.UInt32:
                {
                    var iter = Subject.AsIterator<uint>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    var expected = Convert.ToUInt32(val);
                    for (int i = 0; hasnext(); i++)
                    {
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith($"Expected NDArray's {2}th value to be {0}, but found {1} (dtype: UInt32).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n{val}", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Int64:
                {
                    var iter = Subject.AsIterator<long>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    var expected = Convert.ToInt64(val);
                    for (int i = 0; hasnext(); i++)
                    {
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith($"Expected NDArray's {2}th value to be {0}, but found {1} (dtype: Int64).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n{val}", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.UInt64:
                {
                    var iter = Subject.AsIterator<ulong>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    var expected = Convert.ToUInt64(val);
                    for (int i = 0; hasnext(); i++)
                    {
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith($"Expected NDArray's {2}th value to be {0}, but found {1} (dtype: UInt64).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n{val}", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Char:
                {
                    var iter = Subject.AsIterator<char>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    var expected = Convert.ToChar(val);
                    for (int i = 0; hasnext(); i++)
                    {
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith($"Expected NDArray's {2}th value to be {0}, but found {1} (dtype: Char).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n{val}", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Double:
                {
                    var iter = Subject.AsIterator<double>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    var expected = Convert.ToDouble(val);
                    for (int i = 0; hasnext(); i++)
                    {
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith($"Expected NDArray's {2}th value to be {0}, but found {1} (dtype: Double).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n{val}", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Single:
                {
                    var iter = Subject.AsIterator<float>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    var expected = Convert.ToSingle(val);
                    for (int i = 0; hasnext(); i++)
                    {
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith($"Expected NDArray's {2}th value to be {0}, but found {1} (dtype: Single).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n{val}", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Decimal:
                {
                    var iter = Subject.AsIterator<decimal>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    var expected = Convert.ToDecimal(val);
                    for (int i = 0; hasnext(); i++)
                    {
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith($"Expected NDArray's {2}th value to be {0}, but found {1} (dtype: Decimal).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n{val}", expected, nextval, i);
                    }

                    break;
                }

                default:
                    throw new NotSupportedException();
            }

            #endregion

#endif


            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> BeOfValuesApproximately(double sensitivity, params object[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            Subject.size.Should().Be(values.Length, "the method BeOfValuesApproximately also confirms the sizes are matching with given values.");

#if _REGEN
            #region Compute
		    switch (Subject.typecode)
		    {
			    %foreach supported_dtypes,supported_dtypes_lowercase%
			    case NPTypeCode.#1: 
                {
                    var iter = Subject.AsIterator<#2>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");
                        
                        var expected = Convert.To#1(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(Math.Abs(expected - nextval) <= sensitivity)
                            .FailWith($"Expected NDArray's {{2}}th value to be {{0}}, but found {{1}} (dtype: Boolean).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n[{string.Join(", ", values.Select(v => v.ToString()))}]", expected, nextval, i);
                    }
                    break;
                }
			    %
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#else

            #region Compute

            switch (Subject.typecode)
            {
                case NPTypeCode.Boolean:
                {
                    var iter = Subject.AsIterator<bool>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToBoolean(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith($"Expected NDArray's {{2}}th value to be {{0}}, but found {{1}} (dtype: Boolean).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n[{string.Join(", ", values.Select(v => v.ToString()))}]", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Byte:
                {
                    var iter = Subject.AsIterator<byte>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToByte(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(Math.Abs(expected - nextval) <= sensitivity)
                            .FailWith($"Expected NDArray's {{2}}th value to be {{0}}, but found {{1}} (dtype: Boolean).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n[{string.Join(", ", values.Select(v => v.ToString()))}]", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Int16:
                {
                    var iter = Subject.AsIterator<short>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToInt16(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(Math.Abs(expected - nextval) <= sensitivity)
                            .FailWith($"Expected NDArray's {{2}}th value to be {{0}}, but found {{1}} (dtype: Boolean).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n[{string.Join(", ", values.Select(v => v.ToString()))}]", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.UInt16:
                {
                    var iter = Subject.AsIterator<ushort>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToUInt16(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(Math.Abs(expected - nextval) <= sensitivity)
                            .FailWith($"Expected NDArray's {{2}}th value to be {{0}}, but found {{1}} (dtype: Boolean).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n[{string.Join(", ", values.Select(v => v.ToString()))}]", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Int32:
                {
                    var iter = Subject.AsIterator<int>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToInt32(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(Math.Abs(expected - nextval) <= sensitivity)
                            .FailWith($"Expected NDArray's {{2}}th value to be {{0}}, but found {{1}} (dtype: Boolean).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n[{string.Join(", ", values.Select(v => v.ToString()))}]", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.UInt32:
                {
                    var iter = Subject.AsIterator<uint>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToUInt32(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(Math.Abs(expected - nextval) <= sensitivity)
                            .FailWith($"Expected NDArray's {{2}}th value to be {{0}}, but found {{1}} (dtype: Boolean).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n[{string.Join(", ", values.Select(v => v.ToString()))}]", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Int64:
                {
                    var iter = Subject.AsIterator<long>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToInt64(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(Math.Abs(expected - nextval) <= sensitivity)
                            .FailWith($"Expected NDArray's {{2}}th value to be {{0}}, but found {{1}} (dtype: Boolean).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n[{string.Join(", ", values.Select(v => v.ToString()))}]", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.UInt64:
                {
                    var iter = Subject.AsIterator<ulong>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToUInt64(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(Math.Abs((double) (expected - nextval)) <= sensitivity)
                            .FailWith($"Expected NDArray's {{2}}th value to be {{0}}, but found {{1}} (dtype: Boolean).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n[{string.Join(", ", values.Select(v => v.ToString()))}]", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Char:
                {
                    var iter = Subject.AsIterator<char>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToChar(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(Math.Abs(expected - nextval) <= sensitivity)
                            .FailWith($"Expected NDArray's {{2}}th value to be {{0}}, but found {{1}} (dtype: Boolean).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n[{string.Join(", ", values.Select(v => v.ToString()))}]", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Double:
                {
                    var iter = Subject.AsIterator<double>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToDouble(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(Math.Abs(expected - nextval) <= sensitivity)
                            .FailWith($"Expected NDArray's {{2}}th value to be {{0}}, but found {{1}} (dtype: Boolean).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n[{string.Join(", ", values.Select(v => v.ToString()))}]", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Single:
                {
                    var iter = Subject.AsIterator<float>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToSingle(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(Math.Abs(expected - nextval) <= sensitivity)
                            .FailWith($"Expected NDArray's {{2}}th value to be {{0}}, but found {{1}} (dtype: Boolean).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n[{string.Join(", ", values.Select(v => v.ToString()))}]", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Decimal:
                {
                    var iter = Subject.AsIterator<decimal>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToDecimal(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(Math.Abs(expected - nextval) <= (decimal) sensitivity)
                            .FailWith($"Expected NDArray's {{2}}th value to be {{0}}, but found {{1}} (dtype: Boolean).\n------- Subject -------\n{Subject.ToString(false)}\n------- Expected -------\n[{string.Join(", ", values.Select(v => v.ToString()))}]", expected, nextval, i);
                    }

                    break;
                }

                default:
                    throw new NotSupportedException();
            }

            #endregion

#endif


            return new AndConstraint<NDArrayAssertions>(this);
        }
    }
}