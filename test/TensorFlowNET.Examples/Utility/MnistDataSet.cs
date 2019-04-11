using NumSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.Examples.Utility
{
    public class MnistDataSet
    {
        private const string DEFAULT_SOURCE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/";
        private const string TRAIN_IMAGES = "train-images-idx3-ubyte.gz";
        private const string TRAIN_LABELS = "train-labels-idx1-ubyte.gz";
        private const string TEST_IMAGES = "t10k-images-idx3-ubyte.gz";
        private const string TEST_LABELS = "t10k-labels-idx1-ubyte.gz";

        public static Datasets read_data_sets(string train_dir, 
            bool one_hot = false,
            TF_DataType dtype = TF_DataType.TF_FLOAT,
            bool reshape = true,
            int validation_size = 5000,
            int test_size = 5000,
            string source_url = DEFAULT_SOURCE_URL)
        {
            var train_size = validation_size * 2;

            Web.Download(source_url + TRAIN_IMAGES, train_dir, TRAIN_IMAGES);
            Compress.ExtractGZip(Path.Join(train_dir, TRAIN_IMAGES), train_dir);
            var train_images = extract_images(Path.Join(train_dir, TRAIN_IMAGES.Split('.')[0]), limit: train_size);

            Web.Download(source_url + TRAIN_LABELS, train_dir, TRAIN_LABELS);
            Compress.ExtractGZip(Path.Join(train_dir, TRAIN_LABELS), train_dir);
            var train_labels = extract_labels(Path.Join(train_dir, TRAIN_LABELS.Split('.')[0]), one_hot: one_hot, limit: train_size);

            Web.Download(source_url + TEST_IMAGES, train_dir, TEST_IMAGES);
            Compress.ExtractGZip(Path.Join(train_dir, TEST_IMAGES), train_dir);
            var test_images = extract_images(Path.Join(train_dir, TEST_IMAGES.Split('.')[0]), limit: test_size);

            Web.Download(source_url + TEST_LABELS, train_dir, TEST_LABELS);
            Compress.ExtractGZip(Path.Join(train_dir, TEST_LABELS), train_dir);
            var test_labels = extract_labels(Path.Join(train_dir, TEST_LABELS.Split('.')[0]), one_hot: one_hot, limit:test_size);

            int end = train_images.shape[0];
            var validation_images = train_images[np.arange(validation_size)];
            var validation_labels = train_labels[np.arange(validation_size)];
            train_images = train_images[np.arange(validation_size, end)];
            train_labels = train_labels[np.arange(validation_size, end)];

            var train = new DataSet(train_images, train_labels, dtype, reshape);
            var validation = new DataSet(validation_images, validation_labels, dtype, reshape);
            var test = new DataSet(test_images, test_labels, dtype, reshape);

            return new Datasets(train, validation, test);
        }

        public static NDArray extract_images(string file, int? limit=null)
        {
            using (var bytestream = new FileStream(file, FileMode.Open))
            {
                var magic = _read32(bytestream);
                if (magic != 2051)
                    throw new ValueError($"Invalid magic number {magic} in MNIST image file: {file}");
                var num_images =  _read32(bytestream);
                num_images = limit == null ? num_images : Math.Min(num_images, (uint)limit);
                var rows = _read32(bytestream);
                var cols = _read32(bytestream);
                var buf = new byte[rows * cols * num_images];
                bytestream.Read(buf, 0, buf.Length);
                var data = np.frombuffer(buf, np.uint8);
                data = data.reshape((int)num_images, (int)rows, (int)cols, 1);
                return data;
            }
        }

        public static NDArray extract_labels(string file, bool one_hot = false, int num_classes = 10, int? limit = null)
        {
            using (var bytestream = new FileStream(file, FileMode.Open))
            {
                var magic = _read32(bytestream);
                if (magic != 2049)
                    throw new ValueError($"Invalid magic number {magic} in MNIST label file: {file}");
                var num_items = _read32(bytestream);
                num_items = limit == null ? num_items : Math.Min(num_items,(uint) limit);
                var buf = new byte[num_items];
                bytestream.Read(buf, 0, buf.Length);
                var labels = np.frombuffer(buf, np.uint8);
                if (one_hot)
                    return dense_to_one_hot(labels, num_classes);
                return labels;
            }
        }

        private static NDArray dense_to_one_hot(NDArray labels_dense, int num_classes)
        {
            var num_labels = labels_dense.shape[0];
            var index_offset = np.arange(num_labels) * num_classes;
            var labels_one_hot = np.zeros(num_labels, num_classes);

            for(int row = 0; row < num_labels; row++)
            {
                var col = labels_dense.Data<byte>(row);
                labels_one_hot.SetData(1, row, col);
            }

            return labels_one_hot;
        }

        private static uint _read32(FileStream bytestream)
        {
            var buffer = new byte[sizeof(uint)];
            var count = bytestream.Read(buffer, 0, 4);
            return np.frombuffer(buffer, ">u4").Data<uint>(0);
        }
    }
}
