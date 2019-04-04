using NumSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace TensorFlowNET.Examples.CnnTextClassification
{
    public class DataHelpers
    {
        private const string TRAIN_PATH = "text_classification/dbpedia_csv/train.csv";
        private const string TEST_PATH = "text_classification/dbpedia_csv/test.csv";

        public static (int[][], int[], int) build_char_dataset(string step, string model, int document_max_len)
        {
            string alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’'\"/|_#$%ˆ&*˜‘+=<>()[]{} ";
            /*if (step == "train")
                df = pd.read_csv(TRAIN_PATH, names =["class", "title", "content"]);*/
            var char_dict = new Dictionary<string, int>();
            char_dict["<pad>"] = 0;
            char_dict["<unk>"] = 1;
            foreach (char c in alphabet)
                char_dict[c.ToString()] = char_dict.Count;

            var contents = File.ReadAllLines(TRAIN_PATH);
            
            var x = new int[contents.Length][];
            var y = new int[contents.Length];
            for (int i = 0; i < contents.Length; i++)
            {
                string[] parts = contents[i].ToLower().Split(",\"").ToArray();
                string content = parts[2];
                content = content.Substring(0, content.Length - 1);
                x[i] = new int[document_max_len];
                for (int j = 0; j < document_max_len; j++)
                {
                    if (j >= content.Length)
                        x[i][j] = char_dict["<pad>"];
                    else
                        x[i][j] = char_dict.ContainsKey(content[j].ToString()) ? char_dict[content[j].ToString()] : char_dict["<unk>"];
                }
                    
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
            str = Regex.Replace(str, @"[^A-Za-z0-9(),!?\'\`]", " ");
            str = Regex.Replace(str, @"\'s", " \'s");
            return str;
        }
    }
}
