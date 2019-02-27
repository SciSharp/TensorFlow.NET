using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Tensorflow;
using NumSharp.Core;
using Newtonsoft.Json;
using System.Linq;
using Keras;
using System.Text.RegularExpressions;

namespace TensorFlowNET.Examples
{
    public class TextClassificationWithMovieReviews : Python, IExample
    {
        string dir = "text_classification_with_movie_reviews";
        string dataFile = "imdb.zip";

        public void Run()
        {
            var((train_data, train_labels), (test_data, test_labels)) = PrepareData();

            Console.WriteLine($"Training entries: {train_data.size}, labels: {train_labels.size}");

            // A dictionary mapping words to an integer index
            var word_index = GetWordIndex();

            train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                value: word_index["<PAD>"],
                padding: "post",
                maxlen: 256);
        }

        private ((NDArray, NDArray), (NDArray, NDArray)) PrepareData()
        {
            Directory.CreateDirectory(dir);

            // get model file
            string url = $"https://github.com/SciSharp/TensorFlow.NET/raw/master/data/{dataFile}";

            string zipFile = Path.Join(dir, $"imdb.zip");
            Utility.Web.Download(url, zipFile);
            Utility.Compress.UnZip(zipFile, dir);

            // prepare training dataset
            var x_train = ReadData(Path.Join(dir, "x_train.txt"));
            var labels_train = ReadData(Path.Join(dir, "y_train.txt"));
            var indices_train = ReadData(Path.Join(dir, "indices_train.txt"));
            // x_train = x_train[indices_train];
            // labels_train = labels_train[indices_train];

            var x_test = ReadData(Path.Join(dir, "x_test.txt"));
            var labels_test = ReadData(Path.Join(dir, "y_test.txt"));
            var indices_test = ReadData(Path.Join(dir, "indices_test.txt"));
            // x_test = x_test[indices_test];
            // labels_test = labels_test[indices_test];

            // not completed
            /*var xs = x_train.hstack(x_test);
            var labels = labels_train.hstack(labels_test);

            var idx = x_train.size;
            var y_train = labels_train;
            var y_test = labels_test;

            return ((x_train, y_train), (x_test, y_test));*/

            throw new NotImplementedException();
        }

        private int[][] ReadData(string file)
        {
            var lines = new List<int[]>();

            foreach(var line in File.ReadAllLines(file))
            {
                var matches = Regex.Matches(line, @"\d+,*");
                var data = new int[matches.Count];
                for (int i = 0; i < data.Length; i++)
                    data[i] = Convert.ToInt32(matches[i].Value.Trim(','));
                lines.Add(data.ToArray());
            }

            return lines.ToArray();
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
    }
}
