using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow.Keras.Layers;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.UnitTest
{
    [TestClass]
    public class InitLayerNameTest
    {
        [TestMethod]
        public void RNNLayerNameTest() 
        {
            var simpleRnnCell = keras.layers.SimpleRNNCell(1);
            Assert.AreEqual("simple_rnn_cell", simpleRnnCell.Name);
            var simpleRnn = keras.layers.SimpleRNN(2);
            Assert.AreEqual("simple_rnn", simpleRnn.Name);
            var lstmCell = keras.layers.LSTMCell(2);
            Assert.AreEqual("lstm_cell", lstmCell.Name);
            var lstm = keras.layers.LSTM(3);
            Assert.AreEqual("lstm", lstm.Name);
        }

        [TestMethod]
        public void ConvLayerNameTest()
        {
            var conv2d = keras.layers.Conv2D(8, activation: "linear");
            Assert.AreEqual("conv2d", conv2d.Name);
            var conv2dTranspose = keras.layers.Conv2DTranspose(8);
            Assert.AreEqual("conv2d_transpose", conv2dTranspose.Name);
        }
    }
}
