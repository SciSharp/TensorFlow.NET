using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Tensorflow;
using TensorFlowNET.Utility;

namespace TensorFlowNET.Examples.CnnTextClassification
{
    /// <summary>
    /// https://github.com/dongjun-Lee/text-classification-models-tf
    /// </summary>
    public class TextClassificationTrain : Python, IExample
    {
        private string dataDir = "text_classification";
        private string dataFileName = "dbpedia_csv.tar.gz";

        private const int CHAR_MAX_LEN = 1014;

        public void Run()
        {
            download_dbpedia();
            Console.WriteLine("Building dataset...");
            var (x, y, alphabet_size) = DataHelpers.build_char_dataset("train", "vdcnn", CHAR_MAX_LEN);
            var (train_x, valid_x, train_y, valid_y) = train_test_split(x, y, test_size: 0.15);
        }

        public void download_dbpedia()
        {
            string url = "https://github.com/le-scientifique/torchDatasets/raw/master/dbpedia_csv.tar.gz";
            Web.Download(url, dataDir, dataFileName);
            Compress.ExtractTGZ(Path.Join(dataDir, dataFileName), dataDir);
        }
    }
}
