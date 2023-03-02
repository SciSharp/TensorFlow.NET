using Newtonsoft.Json.Linq;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Common
{
    public class CustomizedDTypeJsonConverter : JsonConverter
    {
        public override bool CanConvert(Type objectType)
        {
            return objectType == typeof(TF_DataType);
        }

        public override bool CanRead => true;

        public override bool CanWrite => true;

        public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer)
        {
            var token = JToken.FromObject(value);
            token.WriteTo(writer);
        }

        public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
        {
            if (reader.ValueType == typeof(string))
            {
                var str = (string)serializer.Deserialize(reader, typeof(string));
                return dtypes.tf_dtype_from_name(str);
            }
            else
            {
                return (TF_DataType)serializer.Deserialize(reader, typeof(int));
            }
        }
    }
}
