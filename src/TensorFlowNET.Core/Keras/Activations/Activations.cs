using Newtonsoft.Json;
using System;
using System.Reflection;
using System.Runtime.Versioning;
using Tensorflow.Keras.Saving.Common;
using Tensorflow.Operations.Activation;

namespace Tensorflow.Keras
{
    [JsonConverter(typeof(CustomizedActivationJsonConverter))]
    public class Activation : IActivation
    {
        public string Name { get; set; }
        /// <summary>
        /// The parameters are `features` and `name`.
        /// </summary>
        public Func<Tensor, string, Tensor> ActivationFunction { get; set; }

        /// <summary>
        /// The implementation function of `IActivation`
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor Activate(Tensor x, string name = null) => ActivationFunction(x, name);

        /// <summary>
        /// The function for calling in LayersApi, an alias for `Activate`.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor Apply(Tensor input, string name = null) => Activate(input, name);

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
    /// The `ActivationAdapter` is used to store the string, the `IActivation` implementation class, and the `Func` for LayersApi to accept different types of activation parameters.
    /// One of the properties must be specified while initializing.
    /// </summary>
    public class ActivationAdapter
    {
        /// <summary>
        /// The name of the activaiton function, such as "tanh", "sigmoid".
        /// </summary>
        public string? Name { get; set; } = null;

        /// <summary>
        /// The available `IActivation` implementation class of the activaiton function, such as the `Activation` instances (keras.activations.Tanh, keras.activations.Sigmoid) and other `IActivation` implementation class.
        /// </summary>
        public IActivation? Activation { get; set; } = null;

        /// <summary>
        /// The `Func` definition of the activation function, which can be customized.
        /// </summary>
        public Func<Tensor, string, Tensor>? Func { get; set; } = null;

        public ActivationAdapter(string name)
        {
            Name = name;
        }

        public ActivationAdapter(IActivation activation)
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
