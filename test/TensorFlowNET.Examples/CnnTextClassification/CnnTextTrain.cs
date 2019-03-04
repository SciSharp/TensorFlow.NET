using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.Examples.CnnTextClassification
{
    public class CnnTextTrain : Python, IExample
    {
        // Percentage of the training data to use for validation
        private float dev_sample_percentage = 0.1f; 
        // Data source for the positive data.
        private string positive_data_file = "https://raw.githubusercontent.com/dennybritz/cnn-text-classification-tf/master/data/rt-polaritydata/rt-polarity.pos";
        // Data source for the negative data.
        private string negative_data_file = "https://raw.githubusercontent.com/dennybritz/cnn-text-classification-tf/master/data/rt-polaritydata/rt-polarity.neg";
        // Dimensionality of character embedding (default: 128)
        private int embedding_dim = 128;
        // Comma-separated filter sizes (default: '3,4,5')
        private string filter_sizes = "3,4,5";
        // Number of filters per filter size (default: 128)
        private int num_filters = 128;
        // Dropout keep probability (default: 0.5)
        private float dropout_keep_prob = 0.5f;
        // L2 regularization lambda (default: 0.0)
        private float l2_reg_lambda = 0.0f;
        // Batch Size (default: 64)
        private int batch_size = 64;
        // Number of training epochs (default: 200)
        private int num_epochs = 200;
        // Evaluate model on dev set after this many steps (default: 100)
        private int evaluate_every = 100;
        // Save model after this many steps (default: 100)
        private int checkpoint_every = 100;
        // Number of checkpoints to store (default: 5)
        private int num_checkpoints = 5;
        // Allow device soft device placement
        private bool allow_soft_placement = true;
        // Log placement of ops on devices
        private bool log_device_placement = false;

        public void Run()
        {
            var (x_train, y_train, vocab_processor, x_dev, y_dev) = preprocess();
        }

        public (NDArray, NDArray, NDArray, NDArray, NDArray) preprocess()
        {
            DataHelpers.load_data_and_labels(positive_data_file, negative_data_file);
            throw new NotImplementedException("");
        }
    }
}
