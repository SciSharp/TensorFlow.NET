using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Tensorflow;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.Keras.UnitTest
{
    /// <summary>
    /// https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/layers
    /// </summary>
    [TestClass]
    public class LayersTest : EagerModeTestBase
    {
        [TestMethod]
        public void Sequential()
        {
            var model = keras.Sequential();
            model.add(keras.Input(shape: 16));
        }

        [TestMethod]
        public void Functional()
        {
            var layers = keras.layers;

            var inputs = keras.Input(shape: 784);
            Assert.AreEqual((-1, 784), inputs.TensorShape);

            var dense = layers.Dense(64, activation: keras.activations.Relu);
            var x = dense.Apply(inputs);

            x = layers.Dense(64, activation: keras.activations.Relu).Apply(x);
            var outputs = layers.Dense(10).Apply(x);

            var model = keras.Model(inputs, outputs, name: "mnist_model");
            model.summary();
        }
        
        /// <summary>
        /// Custom layer test, used in Dueling DQN
        /// </summary>
        [TestMethod, Ignore]
        public void FunctionalTest()
        {
            var layers = keras.layers;
            var inputs = layers.Input(shape: 24);
            var x = layers.Dense(128, activation:"relu").Apply(inputs);
            var value = layers.Dense(24).Apply(x);
            var adv = layers.Dense(1).Apply(x);
            
            var adv_out = adv - Binding.tf.reduce_mean(adv, axis: 1, keepdims: true); // Here's problem.
            var outputs = layers.Add().Apply(new Tensors(adv_out, value));
            var model = keras.Model(inputs, outputs);
            model.summary();
            model.compile(optimizer: keras.optimizers.RMSprop(0.001f),
                          loss: keras.losses.MeanSquaredError(),
                          metrics: new[] { "acc" });
            // Here we consider the adv_out is one layer, which is a little different from py's version
            Assert.AreEqual(model.Layers.Count, 6);

            // py code:
            //from tensorflow.keras.layers import Input, Dense, Add, Subtract, Lambda
            //from tensorflow.keras.models import Model
            //from tensorflow.keras.optimizers import RMSprop
            //import tensorflow.keras.backend as K

            //inputs = Input(24)
            //x = Dense(128, activation = "relu")(inputs)
            //value = Dense(24)(x)
            //adv = Dense(1)(x)
            //meam = Lambda(lambda x: K.mean(x, axis = 1, keepdims = True))(adv)
            //adv = Subtract()([adv, meam])
            //outputs = Add()([value, adv])
            //model = Model(inputs, outputs)
            //model.compile(loss = "mse", optimizer = RMSprop(1e-3))
            //model.summary()

            //py output:
            //Model: "functional_3"
            //__________________________________________________________________________________________________
            //Layer(type)                    Output Shape         Param #     Connected to
            //==================================================================================================
            //input_2 (InputLayer)            [(None, 24)]         0
            //__________________________________________________________________________________________________
            //dense_3 (Dense)                 (None, 128)          3200        input_2[0][0]
            //__________________________________________________________________________________________________
            //dense_5 (Dense)                 (None, 1)            129         dense_3[0][0]
            //__________________________________________________________________________________________________
            //lambda_1 (Lambda)               (None, 1)            0           dense_5[0][0]
            //__________________________________________________________________________________________________
            //dense_4 (Dense)                 (None, 24)           3096        dense_3[0][0]
            //__________________________________________________________________________________________________
            //subtract_1 (Subtract)           (None, 1)            0           dense_5[0][0]
            //                                                                 lambda_1[0][0]
            //__________________________________________________________________________________________________
            //add_1 (Add)                     (None, 24)           0           dense_4[0][0]
            //                                                                 subtract_1[0][0]
            //==================================================================================================
            //Total params: 6,425
            //Trainable params: 6,425
            //Non-trainable params: 0
            //__________________________________________________________________________________________________
        }

        /// <summary>
        /// https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
        /// </summary>
        [TestMethod, Ignore]
        public void Embedding()
        {
            var model = keras.Sequential();
            var layer = keras.layers.Embedding(7, 2, input_length: 4);
            model.add(layer);
            // the model will take as input an integer matrix of size (batch,
            // input_length).
            // the largest integer (i.e. word index) in the input should be no larger
            // than 999 (vocabulary size).
            // now model.output_shape == (None, 10, 64), where None is the batch
            // dimension.
            var input_array = np.array(new int[,]
            {
                { 1, 2, 3, 4 },
                { 2, 3, 4, 5 },
                { 3, 4, 5, 6 }
            });
            // model.compile("rmsprop", "mse");
            var output_array = model.predict(input_array);
            Assert.AreEqual((32, 10, 64), output_array.shape);
        }

        /// <summary>
        /// https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
        /// </summary>
        [TestMethod]
        public void Dense()
        {
            // Create a `Sequential` model and add a Dense layer as the first layer.
            var model = keras.Sequential();
            model.add(keras.Input(shape: 16));
            model.add(keras.layers.Dense(32, activation: keras.activations.Relu));
            // Now the model will take as input arrays of shape (None, 16)
            // and output arrays of shape (None, 32).
            // Note that after the first layer, you don't need to specify
            // the size of the input anymore:
            model.add(keras.layers.Dense(32));
            Assert.AreEqual((-1, 32), model.output_shape);
        }

        [TestMethod]
        public void SimpleRNN()
        {

        }
    }
}
