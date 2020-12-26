using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Tensorflow.Keras.Utils;
using NumSharp;
using System.Linq;

namespace Tensorflow.Keras.Datasets
{
    /// <summary>
    /// This is a dataset of 25,000 movies reviews from IMDB, labeled by sentiment
    /// (positive/negative). Reviews have been preprocessed, and each review is
    /// encoded as a list of word indexes(integers).
    /// </summary>
    public class Imdb
    {
        string origin_folder = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/";
        string file_name = "imdb.npz";
        string dest_folder = "imdb";

        /// <summary>
        /// Loads the [IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/).
        /// </summary>
        /// <param name="path"></param>
        /// <param name="num_words"></param>
        /// <param name="skip_top"></param>
        /// <param name="maxlen"></param>
        /// <param name="seed"></param>
        /// <param name="start_char"></param>
        /// <param name="oov_char"></param>
        /// <param name="index_from"></param>
        /// <returns></returns>
        public DatasetPass load_data(string path = "imdb.npz",
            int num_words = -1,
            int skip_top = 0,
            int maxlen = -1,
            int seed = 113,
            int start_char = 1,
            int oov_char= 2,
            int index_from = 3)
        {
            var dst = Download();

            var lines = File.ReadAllLines(Path.Combine(dst, "imdb_train.txt"));
            var x_train_string = new string[lines.Length];
            var y_train = np.zeros(new int[] { lines.Length }, NPTypeCode.Int64);
            for (int i = 0; i < lines.Length; i++)
            {
                y_train[i] = long.Parse(lines[i].Substring(0, 1));
                x_train_string[i] = lines[i].Substring(2);
            }

            var x_train = np.array(x_train_string);

            File.ReadAllLines(Path.Combine(dst, "imdb_test.txt"));
            var x_test_string = new string[lines.Length];
            var y_test = np.zeros(new int[] { lines.Length }, NPTypeCode.Int64);
            for (int i = 0; i < lines.Length; i++)
            {
                y_test[i] = long.Parse(lines[i].Substring(0, 1));
                x_test_string[i] = lines[i].Substring(2);
            }

            var x_test = np.array(x_test_string);

            return new DatasetPass
            {
                Train = (x_train, y_train),
                Test = (x_test, y_test)
            };
        }

        (NDArray, NDArray) LoadX(byte[] bytes)
        {
            var y = np.Load_Npz<byte[]>(bytes);
            return (y["x_train.npy"], y["x_test.npy"]);
        }

        (NDArray, NDArray) LoadY(byte[] bytes)
        {
            var y = np.Load_Npz<long[]>(bytes);
            return (y["y_train.npy"], y["y_test.npy"]);
        }

        string Download()
        {
            var dst = Path.Combine(Path.GetTempPath(), dest_folder);
            Directory.CreateDirectory(dst);

            Web.Download(origin_folder + file_name, dst, file_name);

            return dst;
            // return Path.Combine(dst, file_name);
        }
    }
}
