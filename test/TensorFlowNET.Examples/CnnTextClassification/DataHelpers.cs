using NumSharp.Core;
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
        /// <summary>
        /// Loads MR polarity data from files, splits the data into words and generates labels.
        /// Returns split sentences and labels.
        /// </summary>
        /// <param name="positive_data_file"></param>
        /// <param name="negative_data_file"></param>
        /// <returns></returns>
        public static (NDArray, NDArray) load_data_and_labels(string positive_data_file, string negative_data_file)
        {
            Directory.CreateDirectory("CnnTextClassification");
            Utility.Web.Download(positive_data_file, "CnnTextClassification/rt-polarity.pos");
            Utility.Web.Download(negative_data_file, "CnnTextClassification/rt-polarity.neg");

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
            // var y = np.
            // return (x_text, y);
            throw new NotImplementedException("load_data_and_labels");
        }

        private static string clean_str(string str)
        {
            str = Regex.Replace(str, @"[^A-Za-z0-9(),!?\'\`]", " ");
            str = Regex.Replace(str, @"\'s", " \'s");
            return str;
        }
    }
}
