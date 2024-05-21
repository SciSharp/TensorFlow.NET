using Newtonsoft.Json.Linq;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Operations.Regularizers;

namespace Tensorflow.Keras.Saving.Common
{
  class RegularizerInfo
  {
    public string class_name { get; set; }
    public JObject config { get; set; }
  }

  public class CustomizedRegularizerJsonConverter : JsonConverter
    {
        public override bool CanConvert(Type objectType)
        {
            return objectType == typeof(IRegularizer);
        }

        public override bool CanRead => true;

        public override bool CanWrite => true;

        public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer)
        {
            var regularizer = value as IRegularizer;
            if (regularizer is null)
            {
                JToken.FromObject(null).WriteTo(writer);
                return;
            }
            JToken.FromObject(new RegularizerInfo()
            {
              class_name = regularizer.ClassName,
              config = JObject.FromObject(regularizer.Config)
            }, serializer).WriteTo(writer);
        }

        public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
        {
            var info = serializer.Deserialize<RegularizerInfo>(reader);
            if (info is null)
            {
                return null;
            }
            return info.class_name switch
            {
                "L1L2" => new L1L2 (info.config["l1"].ToObject<float>(), info.config["l2"].ToObject<float>()),
                "L1" => new L1(info.config["l1"].ToObject<float>()),
                "L2" => new L2(info.config["l2"].ToObject<float>()),
            };
        }
    }
}
