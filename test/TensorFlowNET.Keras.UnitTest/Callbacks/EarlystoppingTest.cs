using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;
using Tensorflow.Keras.Callbacks;
using Tensorflow.Keras.Engine;
using static Tensorflow.KerasApi;


namespace Tensorflow.Keras.UnitTest.Callbacks
{
    [TestClass]
    public class EarlystoppingTest
    {
        [TestMethod]
        // Because loading the weight variable into the model has not yet been implemented,
        // so you'd better not set patience too large, because the weights will equal to the last epoch's weights.
        public void Earlystopping()
        {
            var layers = keras.layers;
            var model = keras.Sequential(new List<ILayer>
            {
                layers.Rescaling(1.0f / 255, input_shape: (32, 32, 3)),
                layers.Conv2D(32, 3, padding: "same", activation: keras.activations.Relu),
                layers.MaxPooling2D(),
                layers.Flatten(),
                layers.Dense(128, activation: keras.activations.Relu),
                layers.Dense(10)
            });


            model.summary();

            model.compile(optimizer: keras.optimizers.RMSprop(1e-3f),
            loss: keras.losses.SparseCategoricalCrossentropy(from_logits: true),
            metrics: new[] { "acc" });

            var num_epochs = 3;
            var batch_size = 8;

            var ((x_train, y_train), (x_test, y_test)) = keras.datasets.cifar10.load_data();
            x_train = x_train / 255.0f;
            // define a CallbackParams first, the parameters you pass al least contain Model and Epochs.
            CallbackParams callback_parameters = new CallbackParams
            {
                Model = model,
                Epochs = num_epochs,
            };
            // define your earlystop
            ICallback earlystop = new EarlyStopping(callback_parameters, "accuracy");
            // define a callbcaklist, then add the earlystopping to it.
            var callbacks = new List<ICallback>();
            callbacks.add(earlystop);

            model.fit(x_train[new Slice(0, 2000)], y_train[new Slice(0, 2000)], batch_size, num_epochs, callbacks: callbacks);
        }

    }


}

