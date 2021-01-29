using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Tensorflow.Keras.Utils;
using NumSharp;
using System.Linq;
using NumSharp.Utilities;
using Tensorflow.Queues;

namespace Tensorflow.Keras.Datasets
{
    /// <summary>
    /// This is a dataset of 25,000 movies reviews from IMDB, labeled by sentiment
    /// (positive/negative). Reviews have been preprocessed, and each review is
    /// encoded as a list of word indexes(integers).
    /// </summary>
    public class Imdb
    {
        //string origin_folder = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/";
        string origin_folder = "http://ai.stanford.edu/~amaas/data/sentiment/";
        //string file_name = "imdb.npz";
        string file_name = "aclImdb_v1.tar.gz";
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
            int oov_char = 2,
            int index_from = 3)
        {
            var dst = Download();

            var vocab = BuildVocabulary(Path.Combine(dst, "imdb.vocab"), start_char, oov_char, index_from);

            var (x_train,y_train) = GetDataSet(Path.Combine(dst, "train"));
            var (x_test, y_test) = GetDataSet(Path.Combine(dst, "test"));

            return new DatasetPass
            {
                Train = (x_train, y_train),
                Test = (x_test, y_test)
            };
        }

        private static Dictionary<string, int> BuildVocabulary(string path, 
            int start_char,
            int oov_char,
            int index_from)
        {
            var words = File.ReadAllLines(path);
            var result = new Dictionary<string, int>();
            var idx = index_from;

            foreach (var word in words)
            {
                result[word] = idx;
                idx += 1;
            }

            return result;
        }

        private static (NDArray, NDArray) GetDataSet(string path)
        {
            var posFiles = Directory.GetFiles(Path.Combine(path, "pos")).Slice(0,10);
            var negFiles = Directory.GetFiles(Path.Combine(path, "neg")).Slice(0,10);

            var x_string = new string[posFiles.Length + negFiles.Length];
            var y = new int[posFiles.Length + negFiles.Length];
            var trg = 0;
            var longest = 0;

            for (int i = 0; i < posFiles.Length; i++, trg++)
            {
                y[trg] = 1;
                x_string[trg] = File.ReadAllText(posFiles[i]);
                longest = Math.Max(longest, x_string[trg].Length);
            }
            for (int i = 0; i < posFiles.Length; i++, trg++)
            {
                y[trg] = 0;
                x_string[trg] = File.ReadAllText(negFiles[i]);
                longest = Math.Max(longest, x_string[trg].Length);
            }
            var x = np.array(x_string);

            return (x, y);
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

            Tensorflow.Keras.Utils.Compress.ExtractTGZ(Path.Combine(dst, file_name), dst);

            return Path.Combine(dst, "aclImdb");
        }
    }
}
