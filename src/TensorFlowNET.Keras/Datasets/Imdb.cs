using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Tensorflow.Keras.Utils;
using Tensorflow.NumPy;
using System.Linq;

namespace Tensorflow.Keras.Datasets
{
    /// <summary>
    /// This is a dataset of 25,000 movies reviews from IMDB, labeled by sentiment
    /// (positive/negative). Reviews have been preprocessed, and each review is
    /// encoded as a list of word indexes(integers).
    /// For convenience, words are indexed by overall frequency in the dataset,
    /// so that for instance the integer "3" encodes the 3rd most frequent word in
    /// the data.This allows for quick filtering operations such as:
    /// "only consider the top 10,000 most
    /// common words, but eliminate the top 20 most common words".
    /// As a convention, "0" does not stand for a specific word, but instead is used
    /// to encode the pad token.
    /// Args:
    /// path: where to cache the data (relative to %TEMP%/imdb/imdb.npz).
    /// num_words: integer or None.Words are
    ///     ranked by how often they occur(in the training set) and only
    ///     the `num_words` most frequent words are kept.Any less frequent word
    ///     will appear as `oov_char` value in the sequence data.If None,
    ///     all words are kept.Defaults to `None`.
    /// skip_top: skip the top N most frequently occurring words
    ///     (which may not be informative). These words will appear as
    ///     `oov_char` value in the dataset.When 0, no words are
    ///     skipped. Defaults to `0`.
    /// maxlen: int or None.Maximum sequence length.
    ///     Any longer sequence will be truncated. None, means no truncation.
    ///     Defaults to `None`.
    /// seed: int. Seed for reproducible data shuffling.
    /// start_char: int. The start of a sequence will be marked with this
    ///     character. 0 is usually the padding character. Defaults to `1`.
    /// oov_char: int. The out-of-vocabulary character.
    ///     Words that were cut out because of the `num_words` or
    ///     `skip_top` limits will be replaced with this character.
    /// index_from: int. Index actual words with this index and higher.
    ///     Returns:
    /// Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    /// 
    /// ** x_train, x_test**: lists of sequences, which are lists of indexes
    ///     (integers). If the num_words argument was specific, the maximum
    ///     possible index value is `num_words - 1`. If the `maxlen` argument was
    ///     specified, the largest possible sequence length is `maxlen`.
    /// 
    /// ** y_train, y_test**: lists of integer labels(1 or 0).
    /// 
    /// Raises:
    /// ValueError: in case `maxlen` is so low
    ///     that no input sequence could be kept.
    /// Note that the 'out of vocabulary' character is only used for
    /// words that were present in the training set but are not included
    /// because they're not making the `num_words` cut here.
    /// Words that were not seen in the training set but are in the test set
    /// have simply been skipped.
    /// </summary>
    /// """Loads the [IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/).
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
            var fileBytes = File.ReadAllBytes(Path.Combine(dst, file_name));
            var (y_train, y_test) = LoadY(fileBytes);
            var (x_train, x_test) = LoadX(fileBytes);
            
            /*var lines = File.ReadAllLines(Path.Combine(dst, "imdb_train.txt"));
            var x_train_string = new string[lines.Length];
            var y_train = np.zeros(new int[] { lines.Length }, np.int64);
            for (int i = 0; i < lines.Length; i++)
            {
                y_train[i] = long.Parse(lines[i].Substring(0, 1));
                x_train_string[i] = lines[i].Substring(2);
            }

            var x_train = np.array(x_train_string);

            File.ReadAllLines(Path.Combine(dst, "imdb_test.txt"));
            var x_test_string = new string[lines.Length];
            var y_test = np.zeros(new int[] { lines.Length }, np.int64);
            for (int i = 0; i < lines.Length; i++)
            {
                y_test[i] = long.Parse(lines[i].Substring(0, 1));
                x_test_string[i] = lines[i].Substring(2);
            }

            var x_test = np.array(x_test_string);*/

            return new DatasetPass
            {
                Train = (x_train, y_train),
                Test = (x_test, y_test)
            };
        }

        (NDArray, NDArray) LoadX(byte[] bytes)
        {
            var y = np.Load_Npz<int[,]>(bytes);
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
