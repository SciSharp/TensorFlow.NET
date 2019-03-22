using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Tensorflow;
using TensorFlowNET.Examples.TextClassification;
using TensorFlowNET.Examples.Utility;

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
        private const int NUM_CLASS = 2;

        public void Run()
        {
            PrepareData();
            Console.WriteLine("Building dataset...");
            var (x, y, alphabet_size) = DataHelpers.build_char_dataset("train", "vdcnn", CHAR_MAX_LEN);

            var (train_x, valid_x, train_y, valid_y) = train_test_split(x, y, test_size: 0.15f);

            with(tf.Session(), sess =>
            {
                new VdCnn(alphabet_size, CHAR_MAX_LEN, NUM_CLASS);
            });
        }

        private (int[][], int[][], int[], int[]) train_test_split(int[][] x, int[] y, float test_size = 0.3f)
        {
            int len = x.Length;
            int classes = y.Distinct().Count();
            int samples = len / classes;
            int train_size = int.Parse((samples * (1 - test_size)).ToString());

            var train_x = new List<int[]>();
            var valid_x = new List<int[]>();
            var train_y = new List<int>();
            var valid_y = new List<int>();

            for (int i = 0; i< classes; i++)
            {
                for (int j = 0; j < samples; j++)
                {
                    int idx = i * samples + j;
                    if (idx < train_size + samples * i)
                    {
                        train_x.Add(x[idx]);
                        train_y.Add(y[idx]);
                    }
                    else
                    {
                        valid_x.Add(x[idx]);
                        valid_y.Add(y[idx]);
                    }
                }
            }

            return (train_x.ToArray(), valid_x.ToArray(), train_y.ToArray(), valid_y.ToArray());
        }

        public void PrepareData()
        {
            string url = "https://github.com/le-scientifique/torchDatasets/raw/master/dbpedia_csv.tar.gz";
            Web.Download(url, dataDir, dataFileName);
            Compress.ExtractTGZ(Path.Join(dataDir, dataFileName), dataDir);
        }
    }
}
