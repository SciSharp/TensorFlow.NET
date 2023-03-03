using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow.NumPy;

namespace TensorFlowNET.Keras.UnitTest;

[TestClass]
public class GradientTest
{
    public Model get_actor(int num_states)
    {
        var inputs = keras.layers.Input(shape: num_states);
        var outputs = keras.layers.Dense(1, activation: keras.activations.Tanh).Apply(inputs);

        Model model = keras.Model(inputs, outputs);

        return model;
    }

    public Model get_critic(int num_states, int num_actions)
    {
        // State as input
        var state_input = keras.layers.Input(shape: num_states);

        // Action as input
        var action_input = keras.layers.Input(shape: num_actions);

        var concat = keras.layers.Concatenate(axis: 1).Apply(new Tensors(state_input, action_input));

        var outputs = keras.layers.Dense(1).Apply(concat);

        Model model = keras.Model(new Tensors(state_input, action_input), outputs);
        model.summary();

        return model;
    }

    [TestMethod]
    public void GetGradient_Test()
    {
        var numStates = 3;
        var numActions = 1;
        var batchSize = 64;
        var gamma = 0.99f;

        var target_actor_model = get_actor(numStates);
        var target_critic_model = get_critic(numStates, numActions);
        var critic_model = get_critic(numStates, numActions);

        Tensor state_batch = tf.convert_to_tensor(np.zeros((batchSize, numStates)), TF_DataType.TF_FLOAT);
        Tensor action_batch = tf.convert_to_tensor(np.zeros((batchSize, numActions)), TF_DataType.TF_FLOAT);
        Tensor reward_batch = tf.convert_to_tensor(np.zeros((batchSize, 1)), TF_DataType.TF_FLOAT);
        Tensor next_state_batch = tf.convert_to_tensor(np.zeros((batchSize, numStates)), TF_DataType.TF_FLOAT);

        using (var tape = tf.GradientTape())
        {
            var target_actions = target_actor_model.Apply(next_state_batch, training: true);
            var target_critic_value = target_critic_model.Apply(new Tensors(new Tensor[] { next_state_batch, target_actions }), training: true);

            var y = reward_batch + tf.multiply(gamma, target_critic_value);

            var critic_value = critic_model.Apply(new Tensors(new Tensor[] { state_batch, action_batch }), training: true);

            var critic_loss = math_ops.reduce_mean(math_ops.square(y - critic_value));

            var critic_grad = tape.gradient(critic_loss, critic_model.TrainableVariables);

            Assert.IsNotNull(critic_grad);
            Assert.IsNotNull(critic_grad.First());
        }
    }
}