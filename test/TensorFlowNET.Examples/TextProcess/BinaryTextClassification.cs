using System;
using System.Collections.Generic;
using System.IO;
using Tensorflow;
using Newtonsoft.Json;
using System.Linq;
using System.Text.RegularExpressions;
using NumSharp;
using static Tensorflow.Python;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// This example classifies movie reviews as positive or negative using the text of the review. 
    /// This is a binary—or two-class—classification, an important and widely applicable kind of machine learning problem.
    /// https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_text_classification.ipynb
    /// </summary>
    public class BinaryTextClassification : IExample
    {
        public bool Enabled { get; set; } = true;
        public string Name => "Binary Text Classification";
        public bool IsImportingGraph { get; set; } = true;

        string dir = "binary_text_classification";
        string dataFile = "imdb.zip";
        NDArray train_data, train_labels, test_data, test_labels;

        public bool Run()
        {
            PrepareData();

            Console.WriteLine($"Training entries: {train_data.len}, labels: {train_labels.len}");

            // A dictionary mapping words to an integer index
            var word_index = GetWordIndex();
            
            /*train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                value: word_index["<PAD>"],
                padding: "post",
                maxlen: 256);

            test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                value: word_index["<PAD>"],
                padding: "post",
                maxlen: 256);*/

            // input shape is the vocabulary count used for the movie reviews (10,000 words)
            int vocab_size = 10000;

            var model = keras.Sequential();
            var layer = keras.layers.Embedding(vocab_size, 16);
            model.add(layer);

            return false;
        }

        public void PrepareData()
        {
            Directory.CreateDirectory(dir);

            // get model file
            string url = $"https://github.com/SciSharp/TensorFlow.NET/raw/master/data/{dataFile}";

            Utility.Web.Download(url, dir, "imdb.zip");
            Utility.Compress.UnZip(Path.Join(dir, $"imdb.zip"), dir);

            // prepare training dataset
            var x_train = ReadData(Path.Join(dir, "x_train.txt"));
            var labels_train = ReadData(Path.Join(dir, "y_train.txt"));
            var indices_train = ReadData(Path.Join(dir, "indices_train.txt"));
            x_train = x_train[indices_train];
            labels_train = labels_train[indices_train];

            var x_test = ReadData(Path.Join(dir, "x_test.txt"));
            var labels_test = ReadData(Path.Join(dir, "y_test.txt"));
            var indices_test = ReadData(Path.Join(dir, "indices_test.txt"));
            x_test = x_test[indices_test];
            labels_test = labels_test[indices_test];

            // not completed
            var xs = x_train.hstack<string>(x_test);
            var labels = labels_train.hstack<int>(labels_test);

            var idx = x_train.size;
            var y_train = labels_train;
            var y_test = labels_test;

            // convert x_train
            train_data = new NDArray(np.int32, (x_train.size, 256));
            for (int i = 0; i < x_train.size; i++)
                train_data[i] = x_train[i].Data<string>(0).Split(',').Select(x => int.Parse(x)).ToArray();

            test_data = new NDArray(np.int32, (x_test.size, 256));
            for (int i = 0; i < x_test.size; i++)
                test_data[i] = x_test[i].Data<string>(0).Split(',').Select(x => int.Parse(x)).ToArray();

            train_labels = y_train;
            test_labels = y_test;
        }

        private NDArray ReadData(string file)
        {
            var lines = File.ReadAllLines(file);
            var nd = new NDArray(lines[0].StartsWith("[") ? typeof(string) : np.int32, new Shape(lines.Length));

            if (lines[0].StartsWith("["))
            {
                for (int i = 0; i < lines.Length; i++)
                {
                    /*var matches = Regex.Matches(lines[i], @"\d+\s*");
                    var data = new int[matches.Count];
                    for (int j = 0; j < data.Length; j++)
                        data[j] = Convert.ToInt32(matches[j].Value);
                    nd[i] = data.ToArray();*/
                    nd[i] = lines[i].Substring(1, lines[i].Length - 2).Replace(" ", string.Empty);
                }
            }
            else
            {
                for (int i = 0; i < lines.Length; i++)
                    nd[i] = Convert.ToInt32(lines[i]);
            }
            return nd;
        }

        private Dictionary<string, int> GetWordIndex()
        {
            var result = new Dictionary<string, int>();
            var json = File.ReadAllText(Path.Join(dir, "imdb_word_index.json"));
            var dict = JsonConvert.DeserializeObject<Dictionary<string, int>>(json);

            dict.Keys.Select(k => result[k] = dict[k] + 3).ToList();
            result["<PAD>"] = 0;
            result["<START>"] = 1;
            result["<UNK>"] = 2; // unknown
            result["<UNUSED>"] = 3;

            return result;
        }

        public Graph ImportGraph()
        {
            throw new NotImplementedException();
        }

        public Graph BuildGraph()
        {
            throw new NotImplementedException();
        }

        public bool Train()
        {
            throw new NotImplementedException();
        }

        public bool Predict()
        {
            throw new NotImplementedException();
        }
    }
}
