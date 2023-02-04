using Newtonsoft.Json.Linq;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Common
{
    public class CustomizedAxisJsonConverter : JsonConverter
    {
        public override bool CanConvert(Type objectType)
        {
            return objectType == typeof(Axis);
        }

        public override bool CanRead => true;

        public override bool CanWrite => true;

        public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer)
        {
            if (value is null)
            {
                var token = JToken.FromObject(new int[] { });
                token.WriteTo(writer);
            }
            else if (value is not Axis)
            {
                throw new TypeError($"Unable to use `CustomizedAxisJsonConverter` to serialize the type {value.GetType()}.");
            }
            else
            {
                var token = JToken.FromObject((value as Axis)!.axis);
                token.WriteTo(writer);
            }
        }

        public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
        {
            var axis = serializer.Deserialize(reader, typeof(long[]));
            if (axis is null)
            {
                throw new ValueError("Cannot deserialize 'null' to `Axis`.");
            }
            return new Axis((int[])(axis!));
        }
    }
}
