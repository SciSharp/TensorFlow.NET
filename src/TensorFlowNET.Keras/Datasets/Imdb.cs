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
            x_test.astype(np.int32);
            labels_test.astype(np.int32);

            var indices = np.arange<int>(len(x_train));
            np.random.shuffle(indices, seed);
            x_train = x_train[indices];
            labels_train = labels_train[indices];

            indices = np.arange<int>(len(x_test));
            np.random.shuffle(indices, seed);
            x_test = x_test[indices];
            labels_test = labels_test[indices];

            if (start_char != null)
            {
                int[,] new_x_train = new int[x_train.shape[0], x_train.shape[1] + 1];
                for (var i = 0; i < x_train.shape[0]; i++)
                {
                    new_x_train[i, 0] = (int)start_char;
                    for (var j = 0; j < x_train.shape[1]; j++)
                    {
                        new_x_train[i, j + 1] = x_train[i][j];
                    }
                }
                int[,] new_x_test = new int[x_test.shape[0], x_test.shape[1] + 1];
                for (var i = 0; i < x_test.shape[0]; i++)
                {
                    new_x_test[i, 0] = (int)start_char;
                    for (var j = 0; j < x_test.shape[1]; j++)
                    {
                        new_x_test[i, j + 1] = x_test[i][j];
                    }
                }
                x_train = new NDArray(new_x_train);
                x_test = new NDArray(new_x_test);
            }
            else if (index_from != 0)
            {
                for (var i = 0; i < x_train.shape[0]; i++)
                {
                    for (var j = 0; j < x_train.shape[1]; j++)
                    {
                        if (x_train[i, j] != 0)
                            x_train[i, j] += index_from;
                    }
                }
                for (var i = 0; i < x_test.shape[0]; i++)
                {
                    for (var j = 0; j < x_test.shape[1]; j++)
                    {
                        if (x_test[i, j] != 0)
                            x_test[i, j] += index_from;
                    }
                }
            }

            if (maxlen != null)
            {
                (x_train, labels_train) = data_utils._remove_long_seq((int)maxlen, x_train, labels_train);
                (x_test, labels_test) = data_utils._remove_long_seq((int)maxlen, x_test, labels_test);
                if (x_train.size == 0 || x_test.size == 0)
                    throw new ValueError("After filtering for sequences shorter than maxlen=" +
                        $"{maxlen}, no sequence was kept. Increase maxlen.");
            }

            var xs = np.concatenate(new[] { x_train, x_test });
            var labels = np.concatenate(new[] { labels_train, labels_test });

            if(num_words == null)
            {
                num_words = 0;
                for (var i = 0; i < xs.shape[0]; i++)
                    for (var j = 0; j < xs.shape[1]; j++)
                        num_words = max((int)num_words, (int)xs[i][j]);
            }

            // by convention, use 2 as OOV word
            // reserve 'index_from' (=3 by default) characters:
            // 0 (padding), 1 (start), 2 (OOV)
            if (oov_char != null)
            {
                int[,] new_xs = new int[xs.shape[0], xs.shape[1]];
                for(var i = 0; i < xs.shape[0]; i++)
                {
                    for(var j = 0; j < xs.shape[1]; j++)
                    {
                        if ((int)xs[i][j] == 0 || skip_top <= (int)xs[i][j] && (int)xs[i][j] < num_words)
                            new_xs[i, j] = (int)xs[i][j];
                        else
                            new_xs[i, j] = (int)oov_char;
                    }
                }
                xs = new NDArray(new_xs);
            }
            else
            {
                int[,] new_xs = new int[xs.shape[0], xs.shape[1]];
                for (var i = 0; i < xs.shape[0]; i++)
                {
                    int k = 0;
                    for (var j = 0; j < xs.shape[1]; j++)
                    {
                        if ((int)xs[i][j] == 0 || skip_top <= (int)xs[i][j] && (int)xs[i][j] < num_words)
                            new_xs[i, k++] = (int)xs[i][j];
                    }
                }
                xs = new NDArray(new_xs);
            }

            var idx = len(x_train);
            x_train = xs[$"0:{idx}"];
            x_test = xs[$"{idx}:"];
            var y_train = labels[$"0:{idx}"];
            var y_test = labels[$"{idx}:"];

            return new DatasetPass
            {
                Train = (x_train, y_train),
                Test = (x_test, y_test)
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
