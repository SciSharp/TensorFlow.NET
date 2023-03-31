using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using Tensorflow.Util;
using static Tensorflow.Binding;

namespace Tensorflow.Training.Saving.SavedModel
{
    internal interface ICodec
    {
        //bool CanEncode(StructuredValue value);
        bool CanDecode(StructuredValue value);
        //StructuredValue DoEecode(object value, Func<object, StructuredValue> encode_fn);
        object DoDecode(StructuredValue value, Func<StructuredValue, object> decode_fn);
    }
    public class nested_structure_coder
    {
        private static Dictionary<StructuredValue.KindOneofCase, ICodec> _codecs = null;
        public static object decode_proto(StructuredValue proto)
        {
            if(_codecs is null)
            {
                _codecs = new Dictionary<StructuredValue.KindOneofCase, ICodec>();
                _codecs[StructuredValue.KindOneofCase.ListValue] = new ListCodec();
                _codecs[StructuredValue.KindOneofCase.TupleValue] = new TupleCodec();
                _codecs[StructuredValue.KindOneofCase.DictValue] = new DictCodec();
                _codecs[StructuredValue.KindOneofCase.NamedTupleValue] = new NamedTupleCodec();
                _codecs[StructuredValue.KindOneofCase.Float64Value] = new Float64Codec();
                _codecs[StructuredValue.KindOneofCase.Int64Value] = new Int64Codec();
                _codecs[StructuredValue.KindOneofCase.StringValue] = new StringCodec();
                _codecs[StructuredValue.KindOneofCase.NoneValue] = new NoneCodec();
                _codecs[StructuredValue.KindOneofCase.BoolValue] = new BoolCodec();
                _codecs[StructuredValue.KindOneofCase.TensorShapeValue] = new TensorShapeCodec();
                _codecs[StructuredValue.KindOneofCase.TensorDtypeValue] = new TensorTypeCodec();
                _codecs[StructuredValue.KindOneofCase.TensorSpecValue] = new TensorSpecCodec();
                _codecs[StructuredValue.KindOneofCase.BoundedTensorSpecValue] = new BoundedTensorSpecCodec();
                _codecs[StructuredValue.KindOneofCase.TypeSpecValue] = new TypeSpecCodec();
            }

            return decode_proto_internal(proto, x => decode_proto(x));
        }

        public static object decode_proto_internal(StructuredValue proto, Func<StructuredValue, object> encode_fn)
        {
            Debug.Assert(_codecs[proto.KindCase].CanDecode(proto));
            return _codecs[proto.KindCase].DoDecode(proto, encode_fn);
        }
    }

    internal class ListCodec : ICodec
    {
        public bool CanDecode(StructuredValue value)
        {
            return value.ListValue is not null;
        }

        public object DoDecode(StructuredValue value, Func<StructuredValue, object> decode_fn)
        {
            return value.ListValue.Values.Select(x => decode_fn(x)).ToList();
        }
    }

    internal class TupleCodec: ICodec
    {
        public bool CanDecode(StructuredValue value)
        {
            return value.TupleValue is not null;
        }

        public object DoDecode(StructuredValue value, Func<StructuredValue, object> decode_fn)
        {
            return value.TupleValue.Values.Select(x => decode_fn(x)).ToArray();
        }
    }

    internal class DictCodec : ICodec
    {
        public bool CanDecode(StructuredValue value)
        {
            return value.DictValue is not null;
        }

        public object DoDecode(StructuredValue value, Func<StructuredValue, object> decode_fn)
        {
            return value.DictValue.Fields.ToDictionary(x => x.Key, x => decode_fn(x.Value));
        }
    }

    internal class NamedTupleCodec : ICodec
    {
        public bool CanDecode(StructuredValue value)
        {
            return value.NamedTupleValue is not null;
        }

        public object DoDecode(StructuredValue value, Func<StructuredValue, object> decode_fn)
        {
            var key_value_pairs = value.NamedTupleValue.Values;
            var items = key_value_pairs.ToDictionary(x => x.Key, x => decode_fn(x.Value));
            return new Common.Types.NamedTuple()
            {
                Name = value.NamedTupleValue.Name,
                ValueDict = items
            };
        }
    }

    internal class Float64Codec : ICodec
    {
        public bool CanDecode(StructuredValue value)
        {
            return value.KindCase == StructuredValue.KindOneofCase.Float64Value;
        }

        public object DoDecode(StructuredValue value, Func<StructuredValue, object> decode_fn)
        {
            return value.Float64Value;
        }
    }

    internal class Int64Codec : ICodec
    {
        public bool CanDecode(StructuredValue value)
        {
            return value.KindCase == StructuredValue.KindOneofCase.Int64Value;
        }

        public object DoDecode(StructuredValue value, Func<StructuredValue, object> decode_fn)
        {
            return (int)value.Int64Value;
        }
    }

    internal class StringCodec : ICodec
    {
        public bool CanDecode(StructuredValue value)
        {
            return value.StringValue is not null;
        }

        public object DoDecode(StructuredValue value, Func<StructuredValue, object> decode_fn)
        {
            return tf.compat.as_str(value.StringValue);
        }
    }

    internal class NoneCodec : ICodec
    {
        public bool CanDecode(StructuredValue value)
        {
            return value.NoneValue is not null;
        }

        public object DoDecode(StructuredValue value, Func<StructuredValue, object> decode_fn)
        {
            return null;
        }
    }

    internal class BoolCodec : ICodec
    {
        public bool CanDecode(StructuredValue value)
        {
            return value.KindCase == StructuredValue.KindOneofCase.BoolValue;
        }

        public object DoDecode(StructuredValue value, Func<StructuredValue, object> decode_fn)
        {
            return value.BoolValue;
        }
    }

    internal class TensorShapeCodec : ICodec
    {
        public bool CanDecode(StructuredValue value)
        {
            return value.TensorShapeValue is not null;
        }

        public object DoDecode(StructuredValue value, Func<StructuredValue, object> decode_fn)
        {
            return new Shape(value.TensorShapeValue);
        }
    }

    internal class TensorTypeCodec : ICodec
    {
        public bool CanDecode(StructuredValue value)
        {
            return value.KindCase == StructuredValue.KindOneofCase.TensorDtypeValue;
        }

        public object DoDecode(StructuredValue value, Func<StructuredValue, object> decode_fn)
        {
            return value.TensorDtypeValue.as_tf_dtype();
        }
    }

    internal class TensorSpecCodec : ICodec
    {
        public bool CanDecode(StructuredValue value)
        {
            return value.TensorSpecValue is not null;
        }

        public object DoDecode(StructuredValue value, Func<StructuredValue, object> decode_fn)
        {
            var name = value.TensorSpecValue.Name;
            var shape = decode_fn(new StructuredValue()
            {
                TensorShapeValue = value.TensorSpecValue.Shape
            });
            Debug.Assert(shape is Shape);
            var dtype = decode_fn(new StructuredValue()
            {
                TensorDtypeValue = value.TensorSpecValue.Dtype
            });
            Debug.Assert(dtype is TF_DataType);
            return new Framework.Models.TensorSpec(shape as Shape, (TF_DataType)dtype, 
                string.IsNullOrEmpty(name) ? null : name);
        }
    }

    internal class BoundedTensorSpecCodec : ICodec
    {
        public bool CanDecode(StructuredValue value)
        {
            return value.BoundedTensorSpecValue is not null;
        }

        public object DoDecode(StructuredValue value, Func<StructuredValue, object> decode_fn)
        {
            var btsv = value.BoundedTensorSpecValue;
            var name = btsv.Name;
            var shape = decode_fn(new StructuredValue()
            {
                TensorShapeValue = btsv.Shape
            });
            Debug.Assert(shape is Shape);
            var dtype = decode_fn(new StructuredValue()
            {
                TensorDtypeValue = btsv.Dtype
            });
            Debug.Assert(dtype is TF_DataType);
            throw new NotImplementedException("The `BoundedTensorSpec` has not been supported, " +
                "please submit an issue to https://github.com/SciSharp/TensorFlow.NET/issues");
        }
    }

    internal class TypeSpecCodec : ICodec
    {
        public bool CanDecode(StructuredValue value)
        {
            return value.TypeSpecValue is not null;
        }

        public object DoDecode(StructuredValue value, Func<StructuredValue, object> decode_fn)
        {
            var type_spec_proto = value.TypeSpecValue;
            var type_spec_class_enum = type_spec_proto.TypeSpecClass;
            var class_name = type_spec_proto.TypeSpecClassName;

            throw new NotImplementedException("The `TypeSpec` analysis has not been supported, " +
                "please submit an issue to https://github.com/SciSharp/TensorFlow.NET/issues");
        }
    }
}
