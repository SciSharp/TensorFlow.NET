using System;
using System.Reflection;
using System.Collections.Generic;
using System.Text;
using System.Xml.Linq;
using Tensorflow.Operations.Activation;
using static Tensorflow.Binding;

namespace Tensorflow.Keras
{
    public class Activations: IActivationsApi
    {
        private static Dictionary<string, Activation> _nameActivationMap;

        private static Activation _linear = new Activation()
        {
            Name = "linear",
            ActivationFunction = (features, name) => features
        };
        private static Activation _relu = new Activation()
        {
            Name = "relu",
            ActivationFunction = (features, name) => tf.Context.ExecuteOp("Relu", name, new ExecuteOpArgs(features))
        };
        private static Activation _sigmoid = new Activation()
        {
            Name = "sigmoid",
            ActivationFunction = (features, name) => tf.Context.ExecuteOp("Sigmoid", name, new ExecuteOpArgs(features))
        };
        private static Activation _softmax = new Activation()
        {
            Name = "softmax",
            ActivationFunction = (features, name) => tf.Context.ExecuteOp("Softmax", name, new ExecuteOpArgs(features))
        };
        private static Activation _tanh = new Activation()
        {
            Name = "tanh",
            ActivationFunction = (features, name) => tf.Context.ExecuteOp("Tanh", name, new ExecuteOpArgs(features))
        };
        private static Activation _mish = new Activation()
        {
            Name = "mish",
            ActivationFunction = (features, name) => features * tf.math.tanh(tf.math.softplus(features))
        };

        /// <summary>
        /// Register the name-activation mapping in this static class.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="activation"></param>
        private static void RegisterActivation(Activation activation)
        {
            _nameActivationMap[activation.Name] = activation;
        }

        static Activations()
        {
            _nameActivationMap = new Dictionary<string, Activation>();

            RegisterActivation(_relu);
            RegisterActivation(_linear);
            RegisterActivation(_sigmoid);
            RegisterActivation(_softmax);
            RegisterActivation(_tanh);
            RegisterActivation(_mish);
        }

        public Activation Linear => _linear;

        public Activation Relu => _relu;

        public Activation Sigmoid => _sigmoid;

        public Activation Softmax => _softmax;

        public Activation Tanh => _tanh;

        public Activation Mish => _mish;

        public Activation GetActivationFromName(string name)
        {
            if (name == null)
            {
                return _linear;
            }
            if (!_nameActivationMap.TryGetValue(name, out var res))
            {
                throw new Exception($"Activation {name} not found");
            }
            else
            {
                return res;
            }
        }

        /// <summary>
        /// Convert `ActivationAdapter` to `Activation`.
        /// If more than one properties of `ActivationAdapter` are specified, the order of priority is `Name`, `Activation`, `Func`
        /// </summary>
        /// <param name="adapter"></param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        public Activation GetActivationFromAdapter(ActivationAdapter adapter)
        {
            if(adapter == null)
            {
                return _linear;
            }
            if(adapter.Name != null)
            {
                return GetActivationFromName(adapter.Name);
            }
            else if(adapter.Activation != null)
            {
                return (Activation) adapter.Activation;
            }
            else if(adapter.Func != null)
            {
                return new Activation()
                {
                    Name = adapter.Func.GetMethodInfo().Name,
                    ActivationFunction = adapter.Func
                };
            }
            else
            {
                throw new Exception("Could not interpret activation adapter");
            }
        }
    }
}
