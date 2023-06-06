using Newtonsoft.Json.Linq;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Common.Types;

namespace Tensorflow.Keras.Saving.Json
{
    public class CustomizedKerasShapesWrapperJsonConverter : JsonConverter
    {
        public override bool CanConvert(Type objectType)
        {
            return objectType == typeof(KerasShapesWrapper);
        }

        public override bool CanRead => true;

        public override bool CanWrite => true;

        public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer)
        {
            if (value is null)
            {
                JToken.FromObject(null).WriteTo(writer);
                return;
            }
            if (value is not KerasShapesWrapper wrapper)
            {
                throw new TypeError($"Expected `KerasShapesWrapper` to be serialized, bug got {value.GetType()}");
            }
            if (wrapper.Shapes.Length == 0)
            {
                JToken.FromObject(null).WriteTo(writer);
            }
            else if (wrapper.Shapes.Length == 1)
            {
                JToken.FromObject(wrapper.Shapes[0]).WriteTo(writer);
            }
            else
            {
                JToken.FromObject(wrapper.Shapes).WriteTo(writer);
            }
        }

        public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
        {
            if (reader.TokenType == JsonToken.StartArray)
            {
                TensorShapeConfig[] shapes = serializer.Deserialize<TensorShapeConfig[]>(reader);
                if (shapes is null)
                {
                    return null;
                }
                return new KerasShapesWrapper(shapes);
            }
            else if (reader.TokenType == JsonToken.StartObject)
            {
                var shape = serializer.Deserialize<TensorShapeConfig>(reader);
                if (shape is null)
                {
                    return null;
                }
                return new KerasShapesWrapper(shape);
            }
            else if (reader.TokenType == JsonToken.Null)
            {
                return null;
            }
            else
            {
                throw new ValueError($"Cannot deserialize the token type {reader.TokenType}");
            }
        }
    }
}
