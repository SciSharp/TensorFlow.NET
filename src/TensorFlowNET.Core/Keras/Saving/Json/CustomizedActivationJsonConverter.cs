using Newtonsoft.Json;
using Newtonsoft.Json.Converters;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Saving.Common
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
                var token = JToken.FromObject(((Activation)value).Name);
                token.WriteTo(writer);
            }
        }

        public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
        {
            var activationName = serializer.Deserialize<string>(reader);
            if (tf.keras is null)
            {
                throw new RuntimeError("Tensorflow.Keras is not loaded, please install it first.");
            }
            return tf.keras.activations.GetActivationFromName(string.IsNullOrEmpty(activationName) ? "linear" : activationName);
        }
    }
}
