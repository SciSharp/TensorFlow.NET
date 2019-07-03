using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using TensorFlowNET.Examples.Utility;

namespace TensorFlowNET.Examples.ImageProcess
{
    /// <summary>
    /// Neural Network classifier for Hand Written Digits
    /// Sample Neural Network architecture with two layers implemented for classifying MNIST digits
    /// http://www.easy-tensorflow.com/tf-tutorials/neural-networks
    /// </summary>
    public class DigitRecognitionNN : IExample
    {
        public bool Enabled { get; set; } = true;
        public bool IsImportingGraph { get; set; } = false;

        public string Name => "Digits Recognition Neural Network";

        const int img_h = 28;
        const int img_w = 28;
        int img_size_flat = img_h * img_w; // 784, the total number of pixels
        int n_classes = 10; // Number of classes, one class per digit
        int training_epochs = 10;
        int? train_size = null;
        int validation_size = 5000;
        int? test_size = null;
        int batch_size = 100;
        Datasets mnist;

        public bool Run()
        {
            PrepareData();
            return true;
        }

        public Graph BuildGraph()
        {
            throw new NotImplementedException();
        }

        public Graph ImportGraph()
        {
            throw new NotImplementedException();
        }

        public bool Predict()
        {
            throw new NotImplementedException();
        }

        public void PrepareData()
        {
            mnist = MnistDataSet.read_data_sets("mnist", one_hot: true, train_size: train_size, validation_size: validation_size, test_size: test_size);
        }

        public bool Train()
        {
            throw new NotImplementedException();
        }
    }
}
