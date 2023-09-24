using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Tensorflow.Keras.Utils;

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
    /// Tuple of Numpy arrays: `(x_train, labels_train), (x_test, labels_test)`.
    /// 
    /// ** x_train, x_test**: lists of sequences, which are lists of indexes
    ///     (integers). If the num_words argument was specific, the maximum
    ///     possible index value is `num_words - 1`. If the `maxlen` argument was
    ///     specified, the largest possible sequence length is `maxlen`.
    /// 
    /// ** labels_train, labels_test**: lists of integer labels(1 or 0).
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
        public DatasetPass load_data(
            string path = "imdb.npz",
            int? num_words = null,
            int skip_top = 0,
            int? maxlen = null,
            int seed = 113,
            int? start_char = 1,
            int? oov_char = 2,
            int index_from = 3)
        {
            path = data_utils.get_file(
                path,
                origin: Path.Combine(origin_folder, "imdb.npz"),
                file_hash: "69664113be75683a8fe16e3ed0ab59fda8886cb3cd7ada244f7d9544e4676b9f"
            );
            path = Path.Combine(path, "imdb.npz");
            var fileBytes = File.ReadAllBytes(path);
            var (x_train, x_test) = LoadX(fileBytes);
            var (labels_train, labels_test) = LoadY(fileBytes);

            var indices = np.arange<int>(len(x_train));
            np.random.shuffle(indices, seed);
            x_train = x_train[indices];
            labels_train = labels_train[indices];

            indices = np.arange<int>(len(x_test));
            np.random.shuffle(indices, seed);
            x_test = x_test[indices];
            labels_test = labels_test[indices];

            var x_train_array = (int[,])x_train.ToMultiDimArray<int>();
            var x_test_array = (int[,])x_test.ToMultiDimArray<int>();
            var labels_train_array = (long[])labels_train.ToArray<long>();
            var labels_test_array = (long[])labels_test.ToArray<long>();

            if (start_char != null)
            {
                var (d1, d2) = (x_train_array.GetLength(0), x_train_array.GetLength(1));
                int[,] new_x_train_array = new int[d1, d2 + 1];
                for (var i = 0; i < d1; i++)
                {
                    new_x_train_array[i, 0] = (int)start_char;
                    Array.Copy(x_train_array, i * d2, new_x_train_array, i * (d2 + 1) + 1, d2);
                }
                (d1, d2) = (x_test_array.GetLength(0), x_test_array.GetLength(1));
                int[,] new_x_test_array = new int[d1, d2 + 1];
                for (var i = 0; i < d1; i++)
                {
                    new_x_test_array[i, 0] = (int)start_char;
                    Array.Copy(x_test_array, i * d2, new_x_test_array, i * (d2 + 1) + 1, d2);
                }
                x_train_array = new_x_train_array;
                x_test_array = new_x_test_array;
            }
            else if (index_from != 0)
            {
                var (d1, d2) = (x_train_array.GetLength(0), x_train_array.GetLength(1));
                for (var i = 0; i < d1; i++)
                {
                    for (var j = 0; j < d2; j++)
                    {
                        if (x_train_array[i, j] == 0)
                            break;
                        x_train_array[i, j] += index_from;
                    }
                }
                (d1, d2) = (x_test_array.GetLength(0), x_test_array.GetLength(1));
                for (var i = 0; i < d1; i++)
                {
                    for (var j = 0; j < d2; j++)
                    {
                        if (x_test_array[i, j] == 0)
                            break;
                        x_test[i, j] += index_from;
                    }
                }
            }

            if (maxlen == null)
            {
                maxlen = max(x_train_array.GetLength(1), x_test_array.GetLength(1));
            }
            (x_train_array, labels_train_array) = data_utils._remove_long_seq((int)maxlen, x_train_array, labels_train_array);
            (x_test_array, labels_test_array) = data_utils._remove_long_seq((int)maxlen, x_test_array, labels_test_array);
            if (x_train_array.Length == 0 || x_test_array.Length == 0)
                throw new ValueError("After filtering for sequences shorter than maxlen=" +
                    $"{maxlen}, no sequence was kept. Increase maxlen.");

            int[,] xs_array = new int[x_train_array.GetLength(0) + x_test_array.GetLength(0), (int)maxlen];
            Array.Copy(x_train_array, xs_array, x_train_array.Length);
            Array.Copy(x_test_array, 0, xs_array, x_train_array.Length, x_train_array.Length);

            long[] labels_array = new long[labels_train_array.Length + labels_test_array.Length];
            Array.Copy(labels_train_array, labels_array, labels_train_array.Length);
            Array.Copy(labels_test_array, 0, labels_array, labels_train_array.Length, labels_test_array.Length);

            if (num_words == null)
            {
                var (d1, d2) = (xs_array.GetLength(0), xs_array.GetLength(1));
                num_words = 0;
                for (var i = 0; i < d1; i++)
                    for (var j = 0; j < d2; j++)
                        num_words = max((int)num_words, (int)xs_array[i, j]);
            }

            // by convention, use 2 as OOV word
            // reserve 'index_from' (=3 by default) characters:
            // 0 (padding), 1 (start), 2 (OOV)
            if (oov_char != null)
            {
                var (d1, d2) = (xs_array.GetLength(0), xs_array.GetLength(1));
                int[,] new_xs_array = new int[d1, d2];
                for (var i = 0; i < d1; i++)
                {
                    for (var j = 0; j < d2; j++)
                    {
                        if (xs_array[i, j] == 0 || skip_top <= xs_array[i, j] && xs_array[i, j] < num_words)
                            new_xs_array[i, j] = xs_array[i, j];
                        else
                            new_xs_array[i, j] = (int)oov_char;
                    }
                }
                xs_array = new_xs_array;
            }
            else
            {
                var (d1, d2) = (xs_array.GetLength(0), xs_array.GetLength(1));
                int[,] new_xs_array = new int[d1, d2];
                for (var i = 0; i < d1; i++)
                {
                    int k = 0;
                    for (var j = 0; j < d2; j++)
                    {
                        if (xs_array[i, j] == 0 || skip_top <= xs_array[i, j] && xs_array[i, j] < num_words)
                            new_xs_array[i, k++] = xs_array[i, j];
                    }
                }
                xs_array = new_xs_array;
            }

            Array.Copy(xs_array, x_train_array, x_train_array.Length);
            Array.Copy(xs_array, x_train_array.Length, x_test_array, 0, x_train_array.Length);

            Array.Copy(labels_array, labels_train_array, labels_train_array.Length);
            Array.Copy(labels_array, labels_train_array.Length, labels_test_array, 0, labels_test_array.Length);

            return new DatasetPass
            {
                Train = (x_train_array, labels_train_array),
                Test = (x_test_array, labels_test_array)
            };
        }

        (NDArray, NDArray) LoadX(byte[] bytes)
        {
            var x = np.Load_Npz<int[,]>(bytes);
            return (x["x_train.npy"], x["x_test.npy"]);
        }

        (NDArray, NDArray) LoadY(byte[] bytes)
        {
            var y = np.Load_Npz<long[]>(bytes);
            return (y["y_train.npy"], y["y_test.npy"]);
        }
    }
}
