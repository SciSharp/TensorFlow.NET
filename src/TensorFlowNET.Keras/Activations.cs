using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Operations.Activation;
using static Tensorflow.Binding;

namespace Tensorflow.Keras
{
    public class Activations
    {
        private static Dictionary<string, Activation> _nameActivationMap;
        private static Dictionary<Activation, string> _activationNameMap;

        private static Activation _linear = (features, name) => features;
        private static Activation _relu = (features, name)
            => tf.Context.ExecuteOp("Relu", name, new ExecuteOpArgs(features));
        private static Activation _sigmoid = (features, name)
            => tf.Context.ExecuteOp("Sigmoid", name, new ExecuteOpArgs(features));
        private static Activation _softmax = (features, name)
                => tf.Context.ExecuteOp("Softmax", name, new ExecuteOpArgs(features));
        private static Activation _tanh = (features, name)
            => tf.Context.ExecuteOp("Tanh", name, new ExecuteOpArgs(features));
        private static Activation _mish = (features, name)
            => features * tf.math.tanh(tf.math.softplus(features));

        /// <summary>
        /// Register the name-activation mapping in this static class.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="activation"></param>
        private static void RegisterActivation(string name, Activation activation)
        {
            _nameActivationMap[name] = activation;
            _activationNameMap[activation] = name;
        }

        static Activations()
        {
            _nameActivationMap = new Dictionary<string, Activation>();
            _activationNameMap= new Dictionary<Activation, string>();

            RegisterActivation("relu", _relu);
            RegisterActivation("linear", _linear);
            RegisterActivation("sigmoid", _sigmoid);
            RegisterActivation("softmax", _softmax);
            RegisterActivation("tanh", _tanh);
            RegisterActivation("mish", _mish);
        }

        public Activation Linear => _linear;

        public Activation Relu => _relu;

        public Activation Sigmoid => _sigmoid;

        public Activation Softmax => _softmax;

        public Activation Tanh => _tanh;

        public Activation Mish => _mish;

        public static Activation GetActivationByName(string name)
        {
            if (!_nameActivationMap.TryGetValue(name, out var res))
            {
                throw new Exception($"Activation {name} not found");
            }
            else
            {
                return res;
            }
        }

        public static string GetNameByActivation(Activation activation)
        {
            if(!_activationNameMap.TryGetValue(activation, out var name))
            {
                throw new Exception($"Activation {activation} not found");
            }
            else
            {
                return name;
            }
        }
    }
}
