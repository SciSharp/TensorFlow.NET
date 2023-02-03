using Newtonsoft.Json;
using Newtonsoft.Json.Converters;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Common
{
    public class CustomizedActivationJsonConverter : JsonConverter
    {
        public override bool CanConvert(Type objectType)
        {
            return objectType == typeof(Activation);
        }

        public override bool CanRead => true;

        public override bool CanWrite => true;

        public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer)
        {
            if (value is null)
            {
                var token = JToken.FromObject("");
                token.WriteTo(writer);
            }
            else if (value is not Activation)
            {
                throw new TypeError($"Unable to use `CustomizedActivationJsonConverter` to serialize the type {value.GetType()}.");
            }
            else
            {
                var token = JToken.FromObject((value as Activation)!.GetType().Name);
                token.WriteTo(writer);
            }
        }

        public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
        {
            throw new NotImplementedException();
            //var dims = serializer.Deserialize(reader, typeof(string));
            //if (dims is null)
            //{
            //    throw new ValueError("Cannot deserialize 'null' to `Activation`.");
            //}
            //return new Shape((long[])(dims!));
        }
    }
}
