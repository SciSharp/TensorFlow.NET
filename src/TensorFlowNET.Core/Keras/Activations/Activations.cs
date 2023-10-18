using Newtonsoft.Json;
using System.Reflection;
using System.Runtime.Versioning;
using Tensorflow.Keras.Saving.Common;

namespace Tensorflow.Keras
{
    [JsonConverter(typeof(CustomizedActivationJsonConverter))]
    public class Activation
    {
        public string Name { get; set; }
        /// <summary>
        /// The parameters are `features` and `name`.
        /// </summary>
        public Func<Tensor, string, Tensor> ActivationFunction { get; set; }

        public Tensor Apply(Tensor input, string name = null) => ActivationFunction(input, name);

        public static implicit operator Activation(Func<Tensor, string, Tensor> func)
        {
            return new Activation()
            {
                Name = func.GetMethodInfo().Name,
                ActivationFunction = func
            };
        }
    }

    public interface IActivationsApi
    {
        Activation GetActivationFromName(string name);
        Activation Linear { get; }

        Activation Relu { get; }
        Activation Relu6 { get; }

        Activation Sigmoid { get; }

        Activation Softmax { get; }

        Activation Tanh { get; }

        Activation Mish { get; }
    }
}
