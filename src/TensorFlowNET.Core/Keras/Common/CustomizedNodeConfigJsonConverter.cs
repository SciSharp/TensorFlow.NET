using Newtonsoft.Json;
using Newtonsoft.Json.Converters;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Keras.Saving;

namespace Tensorflow.Keras.Common
{
    public class CustomizedNodeConfigJsonConverter : JsonConverter
    {
        public override bool CanConvert(Type objectType)
        {
            return objectType == typeof(NodeConfig);
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
            else if (value is not NodeConfig)
            {
                throw new TypeError($"Unable to use `CustomizedNodeConfigJsonConverter` to serialize the type {value.GetType()}.");
            }
            else
            {
                var config = value as NodeConfig;
                var token = JToken.FromObject(new object[] { config!.Name, config.NodeIndex, config.TensorIndex });
                token.WriteTo(writer);
            }
        }

        public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
        {
            var values = serializer.Deserialize(reader, typeof(object[])) as object[];
            if (values is null)
            {
                throw new ValueError("Cannot deserialize 'null' to `Shape`.");
            }
            if(values.Length != 3)
            {
                throw new ValueError($"The value ({string.Join(", ", values)}) cannot be deserialized to type `NodeConfig`.");
            }
            if (values[0] is not string)
            {
                throw new TypeError($"The first value of `NodeConfig` is expected to be `string`, but got `{values[0].GetType().Name}`");
            }
            if (values[1] is not int)
            {
                throw new TypeError($"The first value of `NodeConfig` is expected to be `int`, but got `{values[1].GetType().Name}`");
            }
            if (values[2] is not int)
            {
                throw new TypeError($"The first value of `NodeConfig` is expected to be `int`, but got `{values[2].GetType().Name}`");
            }
            return new NodeConfig()
            {
                Name = values[0] as string,
                NodeIndex = (int)values[1],
                TensorIndex = (int)values[2]
            };
        }
    }
}
