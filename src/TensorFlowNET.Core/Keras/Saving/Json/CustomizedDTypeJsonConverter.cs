using Newtonsoft.Json.Linq;
using Newtonsoft.Json;

namespace Tensorflow.Keras.Saving.Common
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
            var token = JToken.FromObject(((TF_DataType)value).as_numpy_name());
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
