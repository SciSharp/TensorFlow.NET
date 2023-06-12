using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;
using Tensorflow.Keras.Callbacks;
using Tensorflow.Keras.Engine;
using Tensorflow.NumPy;
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
                layers.Rescaling(1.0f / 255, input_shape: (28, 28, 1)),
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

            var data_loader = new MnistModelLoader();

            var dataset = data_loader.LoadAsync(new ModelLoadSetting
            {
                TrainDir = "mnist",
                OneHot = false,
                ValidationSize = 59900,
            }).Result;

            NDArray x1 = np.reshape(dataset.Train.Data, (dataset.Train.Data.shape[0], 28, 28, 1));
            NDArray x2 = x1;

            var x = new NDArray[] { x1, x2 };

            // define a CallbackParams first, the parameters you pass al least contain Model and Epochs.
            CallbackParams callback_parameters = new CallbackParams
            {
                Model = model,
                Epochs = num_epochs,
            };
            // define your earlystop
            ICallback earlystop = new EarlyStopping(callback_parameters, "accuracy");
            // define a callbcaklist, then add the earlystopping to it.
            var callbacks = new List<ICallback>{ earlystop};
            model.fit(x, dataset.Train.Labels, batch_size, num_epochs, callbacks: callbacks);
        }

    }


}

