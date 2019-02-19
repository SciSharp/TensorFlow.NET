using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.Examples
{
    public class TextClassificationWithMovieReviews : Python, IExample
    {
        string dir = "text_classification_with_movie_reviews";

        public void Run()
        {
            PrepareData();
        }

        private void PrepareData()
        {

            Directory.CreateDirectory(dir);

            // get model file
            string url = "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz";

            string zipFile = Path.Join(dir, $"imdb.zip");
            Utility.Web.Download(url, zipFile);

            if (!File.Exists(Path.Join(dir, zipFile)))
                Utility.Compress.ExtractTGZ(zipFile, dir);
        }
    }
}
