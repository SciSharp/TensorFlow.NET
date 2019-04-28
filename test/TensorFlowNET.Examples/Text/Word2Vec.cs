using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Tensorflow;
using TensorFlowNET.Examples.Utility;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// Implement Word2Vec algorithm to compute vector representations of words.
    /// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/word2vec.py
    /// </summary>
    public class Word2Vec : Python, IExample
    {
        public int Priority => 12;
        public bool Enabled { get; set; } = true;
        public string Name => "Word2Vec";

        // Training Parameters
        float learning_rate = 0.1f;
        int batch_size = 128;
        int num_steps = 3000000;
        int display_step = 10000;
        int eval_step = 200000;

        // Evaluation Parameters
        string[] eval_words = new string[] { "five", "of", "going", "hardware", "american", "britain" };
        string[] text_words;

        // Word2Vec Parameters
        int embedding_size = 200; // Dimension of the embedding vector
        int max_vocabulary_size = 50000; // Total number of different words in the vocabulary
        int min_occurrence = 10; // Remove all words that does not appears at least n times
        int skip_window = 3; // How many words to consider left and right
        int num_skips = 2; // How many times to reuse an input to generate a label
        int num_sampled = 64; // Number of negative examples to sample

        int data_index;

        public bool Run()
        {
            PrepareData();

            var graph = tf.Graph().as_default();

            tf.train.import_meta_graph("graph/word2vec.meta");

            // Initialize the variables (i.e. assign their default value)
            var init = tf.global_variables_initializer();

            with(tf.Session(graph), sess =>
            {
                sess.run(init);
            });

            return false;
        }

        // Generate training batch for the skip-gram model
        private void next_batch()
        {

        }

        public void PrepareData()
        {
            // Download graph meta
            var url = "https://github.com/SciSharp/TensorFlow.NET/raw/master/graph/word2vec.meta";
            Web.Download(url, "graph", "word2vec.meta");

            // Download a small chunk of Wikipedia articles collection
            url = "https://raw.githubusercontent.com/SciSharp/TensorFlow.NET/master/data/text8.zip";
            Web.Download(url, "word2vec", "text8.zip");
            // Unzip the dataset file. Text has already been processed
            Compress.UnZip(@"word2vec\text8.zip", "word2vec");

            int wordId = 0;
            text_words = File.ReadAllText(@"word2vec\text8").Trim().ToLower().Split();
            // Build the dictionary and replace rare words with UNK token
            var word2id = text_words.GroupBy(x => x)
                .Select(x => new WordId
                {
                    Word = x.Key,
                    Occurrence = x.Count()
                })
                .Where(x => x.Occurrence >= min_occurrence) // Remove samples with less than 'min_occurrence' occurrences
                .OrderByDescending(x => x.Occurrence) // Retrieve the most common words
                .Select(x => new WordId
                {
                    Word = x.Word,
                    Id = ++wordId, // Assign an id to each word
                    Occurrence = x.Occurrence
                })
                .ToList();

            // Retrieve a word id, or assign it index 0 ('UNK') if not in dictionary
            var data = (from word in text_words
                        join id in word2id on word equals id.Word into wi
                        from wi2 in wi.DefaultIfEmpty()
                        select wi2 == null ? 0 : wi2.Id).ToList();

            word2id.Insert(0, new WordId { Word = "UNK", Id = 0, Occurrence = data.Count(x => x == 0) });

            print($"Words count: {text_words.Length}");
            print($"Unique words: {text_words.Distinct().Count()}");
            print($"Vocabulary size: {word2id.Count}");
            print($"Most common words: {string.Join(", ", word2id.Take(10))}");
        }

        private class WordId
        {
            public string Word { get; set; }
            public int Id { get; set; }
            public int Occurrence { get; set; }

            public override string ToString()
            {
                return Word + " " + Id + " " + Occurrence;
            }
        }
    }
}
