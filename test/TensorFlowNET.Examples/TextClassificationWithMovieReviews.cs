using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Tensorflow;
using NumSharp.Core;

namespace TensorFlowNET.Examples
{
    public class TextClassificationWithMovieReviews : Python, IExample
    {
        string dir = "text_classification_with_movie_reviews";
        string dataFile = "imdb.zip";

        public void Run()
        {
            PrepareData();
        }

        private void PrepareData()
        {
            Directory.CreateDirectory(dir);

            // get model file
            string url = $"https://github.com/SciSharp/TensorFlow.NET/raw/master/data/{dataFile}";

            string zipFile = Path.Join(dir, $"imdb.zip");
            Utility.Web.Download(url, zipFile);
            Utility.Compress.UnZip(zipFile, dir);

            // prepare training dataset
            NDArray x_train = File.ReadAllLines(Path.Join(dir, "x_train.txt"));
            NDArray labels_train = File.ReadAllLines(Path.Join(dir, "y_train.txt"));
            NDArray indices_train = File.ReadAllLines(Path.Join(dir, "indices_train.txt"));
            x_train = x_train[indices_train];
            labels_train = labels_train[indices_train];

            NDArray x_test = File.ReadAllLines(Path.Join(dir, "x_test.txt"));
            NDArray labels_test = File.ReadAllLines(Path.Join(dir, "y_test.txt"));
            NDArray indices_test = File.ReadAllLines(Path.Join(dir, "indices_test.txt"));
            x_test = x_test[indices_test];
            labels_test = labels_test[indices_test];
        }
    }
}
