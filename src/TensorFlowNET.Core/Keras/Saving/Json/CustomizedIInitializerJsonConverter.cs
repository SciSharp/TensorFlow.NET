using Newtonsoft.Json.Linq;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Operations;

using Tensorflow.Operations.Initializers;

namespace Tensorflow.Keras.Saving.Common
{
    class InitializerInfo
    {
        public string class_name { get; set; }
        public JObject config { get; set; }
    }
    public class CustomizedIinitializerJsonConverter : JsonConverter
    {
        public override bool CanConvert(Type objectType)
        {
            return objectType == typeof(IInitializer);
        }

        public override bool CanRead => true;

        public override bool CanWrite => true;

        public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer)
        {
            var initializer = value as IInitializer;
            if (initializer is null)
            {
                JToken.FromObject(null).WriteTo(writer);
                return;
            }
            JToken.FromObject(new InitializerInfo()
            {
                class_name = initializer.ClassName,
                config = JObject.FromObject(initializer.Config)
            }, serializer).WriteTo(writer);
        }

        public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
        {
            var info = serializer.Deserialize<InitializerInfo>(reader);
            if (info is null)
            {
                return null;
            }
            return info.class_name switch
            {
                "Constant" => new Constant<float>(info.config["value"].ToObject<float>()),
                "GlorotUniform" => new GlorotUniform(seed: info.config["seed"].ToObject<int?>()),
                "Ones" => new Ones(),
                "Orthogonal" => new Orthogonal(info.config["gain"].ToObject<float>(), info.config["seed"].ToObject<int?>()),
                "RandomNormal" => new RandomNormal(info.config["mean"].ToObject<float>(), info.config["stddev"].ToObject<float>(),
                    info.config["seed"].ToObject<int?>()),
                "RandomUniform" => new RandomUniform(minval: info.config["minval"].ToObject<float>(),
                    maxval: info.config["maxval"].ToObject<float>(), seed: info.config["seed"].ToObject<int?>()),
                "TruncatedNormal" => new TruncatedNormal(info.config["mean"].ToObject<float>(), info.config["stddev"].ToObject<float>(),
                    info.config["seed"].ToObject<int?>()),
                "VarianceScaling" => new VarianceScaling(info.config["scale"].ToObject<float>(), info.config["mode"].ToObject<string>(),
                    info.config["distribution"].ToObject<string>(), info.config["seed"].ToObject<int?>()),
                "Zeros" => new Zeros(),
                _ => throw new ValueError($"The specified initializer {info.class_name} cannot be recognized.")
            };
        }
    }
}
