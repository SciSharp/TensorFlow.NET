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

    /// <summary>
    /// The ActivationAdaptor is used to store string, Activation, and Func for Laysers Api to accept different types of activation parameters.
    /// One of the properties must be specified while initializing.
    /// </summary>
    public class ActivationAdaptor
    {
        /// <summary>
        /// The name of activaiton function, such as `tanh`, `sigmoid`.
        /// </summary>
        public string? Name { get; set; } = null;

        /// <summary>
        /// The available Activation instance of activaiton function, such as keras.activations.Tanh, keras.activations.Sigmoid.
        /// </summary>
        public Activation? Activation { get; set; } = null;

        /// <summary>
        /// The Func definition of activation function, which can be customized.
        /// </summary>
        public Func<Tensor, string, Tensor>? Func { get; set; } = null;

        public ActivationAdaptor(string name)
        {
            Name = name;
        }

        public ActivationAdaptor(Activation activation)
        {
            Activation = activation;
        }

        public ActivationAdaptor(Func<Tensor, string, Tensor> func)
        {
            Func = func;
        }

        public static implicit operator ActivationAdaptor(string name)
        {
            return new ActivationAdaptor(name);
        }

        public static implicit operator ActivationAdaptor(Activation activation)
        {
            return new ActivationAdaptor(activation);
        }

        public static implicit operator ActivationAdaptor(Func<Tensor, string, Tensor> func)
        {
            return new ActivationAdaptor(func);
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
