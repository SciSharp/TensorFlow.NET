using NumSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Tensorflow;
using TensorFlowNET.Examples.Utility;
using static Tensorflow.Python;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// Implement Word2Vec algorithm to compute vector representations of words.
    /// https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/word2vec.py
    /// </summary>
    public class Word2Vec : IExample
    {
        public bool Enabled { get; set; } = true;
        public string Name => "Word2Vec";
        public bool IsImportingGraph { get; set; } = true;

        // Training Parameters
        float learning_rate = 0.1f;
        int batch_size = 128;
        int num_steps = 30000; //3000000;
        int display_step = 1000; //10000;
        int eval_step = 5000;//200000;

        // Evaluation Parameters
        string[] eval_words = new string[] { "five", "of", "going", "hardware", "american", "britain" };
        string[] text_words;
        List<WordId> word2id;
        int[] data;

        // Word2Vec Parameters
        int embedding_size = 200; // Dimension of the embedding vector
        int max_vocabulary_size = 50000; // Total number of different words in the vocabulary
        int min_occurrence = 10; // Remove all words that does not appears at least n times
        int skip_window = 3; // How many words to consider left and right
        int num_skips = 2; // How many times to reuse an input to generate a label
        int num_sampled = 64; // Number of negative examples to sample

        int data_index = 0;
        int top_k = 8; // number of nearest neighbors
        float average_loss = 0;

        public bool Run()
        {
            PrepareData();

            var graph = tf.Graph().as_default();

            tf.train.import_meta_graph("graph/word2vec.meta");

            // Input data
            Tensor X = graph.OperationByName("Placeholder");
            // Input label
            Tensor Y = graph.OperationByName("Placeholder_1");

            // Compute the average NCE loss for the batch
            Tensor loss_op = graph.OperationByName("Mean");
            // Define the optimizer
            var train_op = graph.OperationByName("GradientDescent");
            Tensor cosine_sim_op = graph.OperationByName("MatMul_1");

            // Initialize the variables (i.e. assign their default value)
            var init = tf.global_variables_initializer();

            with(tf.Session(graph), sess =>
            {
                // Run the initializer
                sess.run(init);

                var x_test = (from word in eval_words
                              join id in word2id on word equals id.Word into wi
                              from wi2 in wi.DefaultIfEmpty()
                              select wi2 == null ? 0 : wi2.Id).ToArray();

                foreach (var step in range(1, num_steps + 1))
                {
                    // Get a new batch of data
                    var (batch_x, batch_y) = next_batch(batch_size, num_skips, skip_window);

                    var result = sess.run(new ITensorOrOperation[] { train_op, loss_op }, new FeedItem(X, batch_x), new FeedItem(Y, batch_y));
                    average_loss += result[1];

                    if (step % display_step == 0 || step == 1)
                    {
                        if (step > 1)
                            average_loss /= display_step;

                        print($"Step {step}, Average Loss= {average_loss.ToString("F4")}");
                        average_loss = 0;
                    }

                    // Evaluation
                    if (step % eval_step == 0 || step == 1)
                    {
                        print("Evaluation...");
                        var sim = sess.run(cosine_sim_op, new FeedItem(X, x_test));
                        foreach(var i in range(len(eval_words)))
                        {
                            var nearest = (0f - sim[i]).argsort<float>()
                                .Data<int>()
                                .Skip(1)
                                .Take(top_k)
                                .ToArray();
                            string log_str = $"\"{eval_words[i]}\" nearest neighbors:";
                            foreach (var k in range(top_k))
                                log_str = $"{log_str} {word2id.First(x => x.Id == nearest[k]).Word},";
                            print(log_str);
                        }
                    }
                }
            });

            return average_loss < 100;
        }

        // Generate training batch for the skip-gram model
        private (NDArray, NDArray) next_batch(int batch_size, int num_skips, int skip_window)
        {
            var batch = np.ndarray((batch_size), dtype: np.int32);
            var labels = np.ndarray((batch_size, 1), dtype: np.int32);
            // get window size (words left and right + current one)
            int span = 2 * skip_window + 1;
            var buffer = new Queue<int>(span);
            if (data_index + span > data.Length)
                data_index = 0;
            data.Skip(data_index).Take(span).ToList().ForEach(x => buffer.Enqueue(x));
            data_index += span;

            foreach (var i in range(batch_size / num_skips))
            {
                var context_words = range(span).Where(x => x != skip_window).ToArray();
                var words_to_use = new int[] { 1, 6 };
                foreach(var (j, context_word) in enumerate(words_to_use))
                {
                    batch[i * num_skips + j] = buffer.ElementAt(skip_window);
                    labels[i * num_skips + j, 0] = buffer.ElementAt(context_word);
                }

                if (data_index == len(data))
                {
                    //buffer.extend(data[0:span]);
                    data_index = span;
                }
                else
                {
                    buffer.Enqueue(data[data_index]);
                    data_index += 1;
                }
            }

            // Backtrack a little bit to avoid skipping words in the end of a batch
            data_index = (data_index + len(data) - span) % len(data);

            return (batch, labels);
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
            word2id = text_words.GroupBy(x => x)
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
            data = (from word in text_words
                        join id in word2id on word equals id.Word into wi
                        from wi2 in wi.DefaultIfEmpty()
                        select wi2 == null ? 0 : wi2.Id).ToArray();

            word2id.Insert(0, new WordId { Word = "UNK", Id = 0, Occurrence = data.Count(x => x == 0) });

            print($"Words count: {text_words.Length}");
            print($"Unique words: {text_words.Distinct().Count()}");
            print($"Vocabulary size: {word2id.Count}");
            print($"Most common words: {string.Join(", ", word2id.Take(10))}");
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
