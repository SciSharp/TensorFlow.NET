using NumSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using static Tensorflow.Binding;
using Tensorflow.Keras.Utils;

namespace Tensorflow.Keras.Datasets
{
    public class Cifar10
    {
        string origin_folder = "https://www.cs.toronto.edu/~kriz/";
        string file_name = "cifar-10-python.tar.gz";
        string dest_folder = "cifar-10-batches";

        /// <summary>
        /// Loads [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
        /// </summary>
        /// <returns></returns>
        public DatasetPass load_data()
        {
            var dst = Download();

            var data_list = new List<Tensor>();
            var label_list = new List<Tensor>();

            foreach (var i in range(1, 6))
            {
                var fpath = Path.Combine(dst, $"data_batch_{i}");
                var (data, labels) = load_batch(fpath);
                data_list.Add(data);
                label_list.Add(labels);
            }

            var x_train_tensor = tf.concat(data_list, 0);
            var y_train_tensor = tf.concat(label_list, 0);
            var y_train = np.array(y_train_tensor.BufferToArray()).reshape(y_train_tensor.shape);

            // test data
            var fpath_test = Path.Combine(dst, "test_batch");
            var (x_test, y_test) = load_batch(fpath_test);

            // channels_last
            x_train_tensor = tf.transpose(x_train_tensor, new[] { 0, 2, 3, 1 });
            var x_train = np.array(x_train_tensor.BufferToArray()).reshape(x_train_tensor.shape);

            var x_test_tensor = tf.transpose(x_test, new[] { 0, 2, 3, 1 });
            x_test = np.array(x_test_tensor.BufferToArray()).reshape(x_test_tensor.shape);

            return new DatasetPass
            {
                Train = (x_train, y_train),
                Test = (x_test, y_test)
            };
        }

        (NDArray, NDArray) load_batch(string fpath, string label_key = "labels")
        {
            var pickle = File.ReadAllBytes(fpath);
            // read description
            var start_pos = 7;
            var desc = read_description(ref start_pos, pickle);
            var labels = read_labels(ref start_pos, pickle);
            var data = read_data(ref start_pos, pickle);

            return (data.Item2, labels.Item2);
        }

        (string, string) read_description(ref int start_pos, byte[] pickle)
        {
            var key_length = pickle[start_pos];
            start_pos++;
            var span = new Span<byte>(pickle, start_pos, key_length);
            var key = Encoding.ASCII.GetString(span.ToArray());
            start_pos += key_length + 3;

            var value_length = pickle[start_pos];
            start_pos++;
            var value = Encoding.ASCII.GetString(new Span<byte>(pickle, start_pos, value_length).ToArray());
            start_pos += value_length;
            start_pos += 3;

            return (key, value);
        }

        (string, NDArray) read_labels(ref int start_pos, byte[] pickle)
        {
            byte[] value = new byte[10000];

            var key_length = pickle[start_pos];
            start_pos++;
            var span = new Span<byte>(pickle, start_pos, key_length);
            var key = Encoding.ASCII.GetString(span.ToArray());
            start_pos += key_length + 6;

            var value_length = 10000;
            for (int i = 0; i < value_length; i++)
            {
                if (i > 0 && i % 1000 == 0)
                    start_pos += 2;
                value[i] = pickle[start_pos + 1];
                start_pos += 2;
            }
            start_pos += 2;

            return (key, np.array(value));
        }

        (string, NDArray) read_data(ref int start_pos, byte[] pickle)
        {
            var key_length = pickle[start_pos];
            start_pos++;
            var span = new Span<byte>(pickle, start_pos, key_length);
            var key = Encoding.ASCII.GetString(span.ToArray());
            start_pos += key_length + 133;
            var value_length = 3072 * 10000;
            var value = new Span<byte>(pickle, start_pos, value_length).ToArray();
            start_pos += value_length;

            return (key, np.array(value).reshape(10000, 3, 32, 32));
        }

        string Download()
        {
            var dst = Path.Combine(Path.GetTempPath(), dest_folder);
            Web.Download(origin_folder + file_name, dst, file_name);
            Compress.ExtractTGZ(Path.Combine(dst, file_name), dst);

            return Path.Combine(dst, "cifar-10-batches-py");
        }
    }
}
