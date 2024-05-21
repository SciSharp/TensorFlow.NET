using Newtonsoft.Json;
using System.Collections.Generic;
using Tensorflow.Keras.Saving.Common;

namespace Tensorflow.Keras
{
  [JsonConverter(typeof(CustomizedRegularizerJsonConverter))]
  public interface IRegularizer
  {
    [JsonProperty("class_name")]
    string ClassName { get; }
    [JsonProperty("config")]
    IDictionary<string, object> Config { get; }
    Tensor Apply(RegularizerArgs args);
  }

  public interface IRegularizerApi
  {
    IRegularizer GetRegularizerFromName(string name);
    IRegularizer L1 { get; }
    IRegularizer L2 { get; }
    IRegularizer L1L2 { get; }
  }

}
