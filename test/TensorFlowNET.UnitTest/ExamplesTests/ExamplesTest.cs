using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow;
using TensorFlowNET.Examples;

namespace TensorFlowNET.ExamplesTests
{
    [TestClass]
    public class ExamplesTest
    {
        [TestMethod]
        public void BasicOperations()
        {
            tf.Graph().as_default();
            new BasicOperations() { Enabled = true }.Run();
        }

        [TestMethod]
        public void HelloWorld()
        {
            tf.Graph().as_default();
            new HelloWorld() { Enabled = true }.Run();
        }

        [TestMethod]
        public void ImageRecognition()
        {
            tf.Graph().as_default();
            new HelloWorld() { Enabled = true }.Run();
        }

        [Ignore]
        [TestMethod]
        public void InceptionArchGoogLeNet()
        {
            tf.Graph().as_default();
            new InceptionArchGoogLeNet() { Enabled = true }.Run();
        }

        [TestMethod]
        public void KMeansClustering()
        {
            tf.Graph().as_default();
            new KMeansClustering() { Enabled = true, IsImportingGraph = true, train_size = 500, validation_size = 100, test_size = 100, batch_size =100 }.Run();
        }

        [TestMethod]
        public void LinearRegression()
        {
            tf.Graph().as_default();
            new LinearRegression() { Enabled = true }.Run();
        }

        [TestMethod]
        public void LogisticRegression()
        {
            tf.Graph().as_default();
            new LogisticRegression() { Enabled = true, training_epochs=10, train_size = 500, validation_size = 100, test_size = 100 }.Run();
        }

        [Ignore]
        [TestMethod]
        public void NaiveBayesClassifier()
        {
            tf.Graph().as_default();
            new NaiveBayesClassifier() { Enabled = false }.Run();
        }

        [Ignore]
        [TestMethod]
        public void NamedEntityRecognition()
        {
            tf.Graph().as_default();
            new NamedEntityRecognition() { Enabled = true }.Run();
        }

        [TestMethod]
        public void NearestNeighbor()
        {
            tf.Graph().as_default();
            new NearestNeighbor() { Enabled = true, TrainSize = 500, ValidationSize = 100, TestSize = 100 }.Run();
        }

        [Ignore]
        [TestMethod]
        public void TextClassificationTrain()
        {
            tf.Graph().as_default();
            new TextClassificationTrain() { Enabled = true, DataLimit=100 }.Run();
        }

        [Ignore]
        [TestMethod]
        public void TextClassificationWithMovieReviews()
        {
            tf.Graph().as_default();
            new BinaryTextClassification() { Enabled = true }.Run();
        }

        [TestMethod]
        public void NeuralNetXor()
        {
            tf.Graph().as_default();
            Assert.IsTrue(new NeuralNetXor() { Enabled = true, IsImportingGraph = false }.Run());
        }

        [TestMethod]
        public void NeuralNetXor_ImportedGraph()
        {
            tf.Graph().as_default();
            Assert.IsTrue(new NeuralNetXor() { Enabled = true, IsImportingGraph = true }.Run());
        }


        [TestMethod]
        public void ObjectDetection()
        {
            tf.Graph().as_default();
            Assert.IsTrue(new ObjectDetection() { Enabled = true, IsImportingGraph = true }.Run());
        }
    }
}
