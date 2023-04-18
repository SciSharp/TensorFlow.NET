using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Saving.SavedModel
{
    [JsonConverter(typeof(CustomizedRevivedConfigJsonConverter))]
    public class RevivedConfig: IKerasConfig
    {
        public JObject Config { get; set; }
    }

    public class CustomizedRevivedConfigJsonConverter : JsonConverter
    {
        public override bool CanConvert(Type objectType)
        {
            return objectType == typeof(RevivedConfig);
        }

        public override bool CanRead => true;

        public override bool CanWrite => true;

        public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer)
        {
            ((RevivedConfig)value).Config.WriteTo(writer);
        }

        public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
        {
            var config = (JObject)serializer.Deserialize(reader, typeof(JObject));
            return new RevivedConfig() { Config = config };
        }
    }
}
