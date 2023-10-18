using System;
using System.Collections.Generic;
using System.Text;
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
        private static Activation _relu6 = new Activation()
        {
            Name = "relu6",
            ActivationFunction = (features, name) => tf.Context.ExecuteOp("Relu6", name, new ExecuteOpArgs(features))
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
        /// <param name="activation"></param>
        private static void RegisterActivation(Activation activation)
        {
            _nameActivationMap[activation.Name] = activation;
        }

        static Activations()
        {
            _nameActivationMap = new Dictionary<string, Activation>();

            RegisterActivation(_relu);
            RegisterActivation(_relu6);
            RegisterActivation(_linear);
            RegisterActivation(_sigmoid);
            RegisterActivation(_softmax);
            RegisterActivation(_tanh);
            RegisterActivation(_mish);
        }

        public Activation Linear => _linear;

        public Activation Relu => _relu;
        public Activation Relu6 => _relu6;

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
    }
}
