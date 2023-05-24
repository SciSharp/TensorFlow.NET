using Newtonsoft.Json;
using System;
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
    public class ActivationAdaptor
    {
        public string? Name { get; set; }

        public Activation? Activation { get; set; }

        public Func<Tensor, string, Tensor>? Func { get; set; }

        public static implicit operator ActivationAdaptor(string name)
        {
            return new ActivationAdaptor()
            {
                Name = name,
                Activation = null,
                Func = null
            };
        }

        public static implicit operator ActivationAdaptor(Activation activation)
        {
            return new ActivationAdaptor()
            {
                Name = null,
                Activation = activation,
                Func = null
            };
        }

        public static implicit operator ActivationAdaptor(Func<Tensor, string, Tensor> func)
        {
            return new ActivationAdaptor()
            {
                Name = null,
                Activation = null,
                Func = func
            };
        }
    }


    public interface IActivationsApi
    {
        Activation GetActivationFromName(string name);
        
        Activation GetActivationFromAdaptor(ActivationAdaptor adaptor);

        Activation Linear { get; }

        Activation Relu { get; }

        Activation Sigmoid { get; }

        Activation Softmax { get; }

        Activation Tanh { get; }

        Activation Mish { get; }
    }
}
