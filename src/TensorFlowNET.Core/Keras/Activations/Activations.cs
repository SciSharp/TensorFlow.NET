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
    /// The ActivationAdapter is used to store string, Activation, and Func for Laysers Api to accept different types of activation parameters.
    /// One of the properties must be specified while initializing.
    /// </summary>
    public class ActivationAdapter
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

        public ActivationAdapter(string name)
        {
            Name = name;
        }

        public ActivationAdapter(Activation activation)
        {
            Activation = activation;
        }

        public ActivationAdapter(Func<Tensor, string, Tensor> func)
        {
            Func = func;
        }

        public static implicit operator ActivationAdapter(string name)
        {
            return new ActivationAdapter(name);
        }

        public static implicit operator ActivationAdapter(Activation activation)
        {
            return new ActivationAdapter(activation);
        }

        public static implicit operator ActivationAdapter(Func<Tensor, string, Tensor> func)
        {
            return new ActivationAdapter(func);
        }
    }


    public interface IActivationsApi
    {
        Activation GetActivationFromName(string name);
        
        Activation GetActivationFromAdapter(ActivationAdapter adapter);

        Activation Linear { get; }

        Activation Relu { get; }

        Activation Sigmoid { get; }

        Activation Softmax { get; }

        Activation Tanh { get; }

        Activation Mish { get; }
    }
}
