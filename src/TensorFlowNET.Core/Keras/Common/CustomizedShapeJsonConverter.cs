using Newtonsoft.Json;
using Newtonsoft.Json.Converters;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Common
{
    public class CustomizedShapeJsonConverter: JsonConverter
    {
        public override bool CanConvert(Type objectType)
        {
            return objectType == typeof(Shape);
        }

        public override bool CanRead => true;

        public override bool CanWrite => true;

        public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer)
        {
            if(value is null)
            {
                var token = JToken.FromObject(null);
                token.WriteTo(writer);
            }
            else if(value is not Shape)
            {
                throw new TypeError($"Unable to use `CustomizedShapeJsonConverter` to serialize the type {value.GetType()}.");
            }
            else
            {
                var shape = (value as Shape)!;
                long?[] dims = new long?[shape.ndim];
                for(int i = 0; i < dims.Length; i++)
                {
                    if (shape.dims[i] == -1)
                    {
                        dims[i] = null;
                    }
                    else
                    {
                        dims[i] = shape.dims[i];
                    }
                }
                var token = JToken.FromObject(dims);
                token.WriteTo(writer);
            }
        }

        public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
        {
            var dims = serializer.Deserialize(reader, typeof(long?[])) as long?[];
            if(dims is null)
            {
                throw new ValueError("Cannot deserialize 'null' to `Shape`.");
            }
            long[] convertedDims = new long[dims.Length];
            for(int i = 0; i < dims.Length; i++)
            {
                convertedDims[i] = dims[i] ?? (-1);
            }
            return new Shape(convertedDims);
        }
    }
}
