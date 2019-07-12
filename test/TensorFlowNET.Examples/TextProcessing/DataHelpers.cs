using NumSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using TensorFlowNET.Examples.Utility;

namespace TensorFlowNET.Examples
{
    public class DataHelpers
    {
        public static Dictionary<string, int> build_word_dict(string path)
        {
            var contents = File.ReadAllLines(path);

            var words = new List<string>();
            foreach (var content in contents)
                words.AddRange(clean_str(content).Split(' ').Where(x => x.Length > 1));
            var word_counter = words.GroupBy(x => x)
                .Select(x => new { Word = x.Key, Count = x.Count() })
                .OrderByDescending(x => x.Count)
                .ToArray();

            var word_dict = new Dictionary<string, int>();
            word_dict["<pad>"] = 0;
            word_dict["<unk>"] = 1;
            word_dict["<eos>"] = 2;
            foreach (var word in word_counter)
                word_dict[word.Word] = word_dict.Count;

            return word_dict;
        }

        public static (int[][], int[]) build_word_dataset(string path, Dictionary<string, int> word_dict, int document_max_len)
        {
            var contents = File.ReadAllLines(path);
            var x = contents.Select(c => (clean_str(c) + " <eos>")
                .Split(' ').Take(document_max_len)
                .Select(w => word_dict.ContainsKey(w) ? word_dict[w] : word_dict["<unk>"]).ToArray())
                .ToArray();

            for (int i = 0; i < x.Length; i++)
                if (x[i].Length == document_max_len)
                    x[i][document_max_len - 1] = word_dict["<eos>"];
                else
                    Array.Resize(ref x[i], document_max_len);

            var y = contents.Select(c => int.Parse(c.Substring(0, c.IndexOf(','))) - 1).ToArray();

            return (x, y);
        }

        public static (int[][], int[], int) build_char_dataset(string path, string model, int document_max_len, int? limit = null, bool shuffle=true)
        {
            string alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’'\"/|_#$%ˆ&*˜‘+=<>()[]{} ";
            /*if (step == "train")
                df = pd.read_csv(TRAIN_PATH, names =["class", "title", "content"]);*/
            var char_dict = new Dictionary<string, int>();
            char_dict["<pad>"] = 0;
            char_dict["<unk>"] = 1;
            foreach (char c in alphabet)
                char_dict[c.ToString()] = char_dict.Count;
            var contents = File.ReadAllLines(path);
            if (shuffle)
                new Random(17).Shuffle(contents);
            //File.WriteAllLines("text_classification/dbpedia_csv/train_6400.csv", contents.Take(6400));
            var size = limit == null ? contents.Length : limit.Value;

            var x = new int[size][];
            var y = new int[size];
            var tenth = size / 10;
            var percent = 0;
            for (int i = 0; i < size; i++)
            {
                if ((i + 1) % tenth == 0)
                {
                    percent += 10;
                    Console.WriteLine($"\t{percent}%");
                }

                string[] parts = contents[i].ToLower().Split(",\"").ToArray();
                string content = parts[2];
                content = content.Substring(0, content.Length - 1);
                var a = new int[document_max_len];
                for (int j = 0; j < document_max_len; j++)
                {
                    if (j >= content.Length)
                        a[j] = char_dict["<pad>"];
                    else
                        a[j] = char_dict.ContainsKey(content[j].ToString()) ? char_dict[content[j].ToString()] : char_dict["<unk>"];
                }
                x[i] = a;
                y[i] = int.Parse(parts[0]);
            }

            return (x, y, alphabet.Length + 2);
        }

        /// <summary>
        /// Loads MR polarity data from files, splits the data into words and generates labels.
        /// Returns split sentences and labels.
        /// </summary>
        /// <param name="positive_data_file"></param>
        /// <param name="negative_data_file"></param>
        /// <returns></returns>
        public static (string[], NDArray) load_data_and_labels(string positive_data_file, string negative_data_file)
        {
            Directory.CreateDirectory("CnnTextClassification");
            Utility.Web.Download(positive_data_file, "CnnTextClassification", "rt -polarity.pos");
            Utility.Web.Download(negative_data_file, "CnnTextClassification", "rt-polarity.neg");

            // Load data from files
            var positive_examples = File.ReadAllLines("CnnTextClassification/rt-polarity.pos")
                .Select(x => x.Trim())
                .ToArray();

            var negative_examples = File.ReadAllLines("CnnTextClassification/rt-polarity.neg")
                .Select(x => x.Trim())
                .ToArray();

            var x_text = new List<string>();
            x_text.AddRange(positive_examples);
            x_text.AddRange(negative_examples);
            x_text = x_text.Select(x => clean_str(x)).ToList();

            var positive_labels = positive_examples.Select(x => new int[2] { 0, 1 }).ToArray();
            var negative_labels = negative_examples.Select(x => new int[2] { 1, 0 }).ToArray();
            var y = np.concatenate(new int[][][] { positive_labels, negative_labels });
            return (x_text.ToArray(), y);
        }

        private static string clean_str(string str)
        {
            str = Regex.Replace(str, "[^A-Za-z0-9(),!?]", " ");
            str = Regex.Replace(str, ",", " ");
            return str;
        }

        /// <summary>
        /// Padding
        /// </summary>
        /// <param name="sequences"></param>
        /// <param name="pad_tok">the char to pad with</param>
        /// <returns>a list of list where each sublist has same length</returns>
        public static (int[][], int[]) pad_sequences(int[][] sequences, int pad_tok = 0)
        {
            int max_length = sequences.Select(x => x.Length).Max();
            return _pad_sequences(sequences, pad_tok, max_length);
        }

        public static (int[][][], int[][]) pad_sequences(int[][][] sequences, int pad_tok = 0)
        {
            int max_length_word = sequences.Select(x => x.Select(w => w.Length).Max()).Max();
            int[][][] sequence_padded;
            var sequence_length = new int[sequences.Length][];
            for (int i = 0; i < sequences.Length; i++)
            {
                // all words are same length now
                var (sp, sl) = _pad_sequences(sequences[i], pad_tok, max_length_word);
                sequence_length[i] = sl;
            }

            int max_length_sentence = sequences.Select(x => x.Length).Max();
            (sequence_padded, _) = _pad_sequences(sequences, np.repeat(pad_tok, max_length_word).Data<int>(), max_length_sentence);
            (sequence_length, _) = _pad_sequences(sequence_length, 0, max_length_sentence);

            return (sequence_padded, sequence_length);
        }

        private static (int[][], int[]) _pad_sequences(int[][] sequences, int pad_tok, int max_length)
        {
            var sequence_length = new int[sequences.Length];
            for (int i = 0; i < sequences.Length; i++)
            {
                sequence_length[i] = sequences[i].Length;
                Array.Resize(ref sequences[i], max_length);
            }

            return (sequences, sequence_length);
        }

        private static (int[][][], int[]) _pad_sequences(int[][][] sequences, int[] pad_tok, int max_length)
        {
            var sequence_length = new int[sequences.Length];
            for (int i = 0; i < sequences.Length; i++)
            {
                sequence_length[i] = sequences[i].Length;
                Array.Resize(ref sequences[i], max_length);
                for (int j = 0; j < max_length - sequence_length[i]; j++)
                {
                    sequences[i][max_length - j - 1] = new int[pad_tok.Length];
                    Array.Copy(pad_tok, sequences[i][max_length - j - 1], pad_tok.Length);
                }
            }

            return (sequences, sequence_length);
        }

        public static string CalculateMD5Hash(string input)
        {
            // step 1, calculate MD5 hash from input
            MD5 md5 = System.Security.Cryptography.MD5.Create();
            byte[] inputBytes = System.Text.Encoding.ASCII.GetBytes(input);
            byte[] hash = md5.ComputeHash(inputBytes);

            // step 2, convert byte array to hex string
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < hash.Length; i++)
            {
                sb.Append(hash[i].ToString("X2"));
            }
            return sb.ToString();
        }
    }
}
