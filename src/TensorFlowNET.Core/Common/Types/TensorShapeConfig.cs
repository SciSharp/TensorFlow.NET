using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Tensorflow.Common.Types
{
    public class TensorShapeConfig
    {
        [JsonProperty("class_name")]
        public string ClassName { get; set; } = "TensorShape";
        [JsonProperty("items")]
        public long?[] Items { get; set; }

        public static implicit operator Shape(TensorShapeConfig shape)
            => shape == null ? null : new Shape(shape.Items.Select(x => x.HasValue ? x.Value : -1).ToArray());

        public static implicit operator TensorShapeConfig(Shape shape)
            => new TensorShapeConfig() { Items = shape.dims.Select<long, long?>(x => x == -1 ? null : x).ToArray() };
    }
}
