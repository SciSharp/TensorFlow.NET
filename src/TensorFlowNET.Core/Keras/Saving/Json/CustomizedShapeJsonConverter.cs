using Newtonsoft.Json;
using Newtonsoft.Json.Converters;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Saving.Common
{
    class ShapeInfoFromPython
    {
        public string class_name { get; set; }
        public long?[] items { get; set; }
    }
    public class CustomizedShapeJsonConverter : JsonConverter
    {
        public override bool CanConvert(Type objectType)
        {
            return objectType == typeof(Shape);
        }

        public override bool CanRead => true;

        public override bool CanWrite => true;

        public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer)
        {
            if (value is null)
            {
                var token = JToken.FromObject(null);
                token.WriteTo(writer);
            }
            else if (value is not Shape)
            {
                throw new TypeError($"Unable to use `CustomizedShapeJsonConverter` to serialize the type {value.GetType()}.");
            }
            else
            {
                var shape = (value as Shape)!;
                long?[] dims = new long?[shape.ndim];
                for (int i = 0; i < dims.Length; i++)
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
                var token = JToken.FromObject(new ShapeInfoFromPython()
                {
                    class_name = "__tuple__",
                    items = dims
                });
                token.WriteTo(writer);
            }
        }

        public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
        {
            long?[] dims;
            if (reader.TokenType == JsonToken.StartObject)
            {
                var shape_info_from_python = serializer.Deserialize<ShapeInfoFromPython>(reader);
                if (shape_info_from_python is null)
                {
                    return null;
                }
                dims = shape_info_from_python.items;
            }
            else if (reader.TokenType == JsonToken.StartArray)
            {
                dims = serializer.Deserialize<long?[]>(reader);
            }
            else if (reader.TokenType == JsonToken.Null)
            {
                return null;
            }
            else
            {
                throw new ValueError($"Cannot deserialize the token {reader} as Shape.");
            }
            long[] convertedDims = new long[dims.Length];
            for (int i = 0; i < dims.Length; i++)
            {
                convertedDims[i] = dims[i] ?? -1;
            }
            return new Shape(convertedDims);
        }
    }
}
