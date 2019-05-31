using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Tensorflow;
using TensorFlowNET.Examples.Utility;
using static Tensorflow.Python;

namespace TensorFlowNET.Examples.ImageProcess
{
    /// <summary>
    /// This example removes the background from an input image.
    /// 
    /// https://github.com/susheelsk/image-background-removal
    /// </summary>
    public class ImageBackgroundRemoval : IExample
    {
        public bool Enabled { get; set; } = true;
        public bool IsImportingGraph { get; set; } = true;

        public string Name => "Image Background Removal";

        string dataDir = "deeplabv3";
        string modelDir = "deeplabv3_mnv2_pascal_train_aug";
        string modelName = "frozen_inference_graph.pb";

        public bool Run()
        {
            PrepareData();

            // import GraphDef from pb file
            var graph = new Graph().as_default();
            graph.Import(Path.Join(dataDir, modelDir, modelName));

            Tensor output = graph.OperationByName("SemanticPredictions");

            with(tf.Session(graph), sess =>
            {
                // Runs inference on a single image.
                sess.run(output, new FeedItem(output, "[np.asarray(resized_image)]"));
            });

            return false;
        }

        public void PrepareData()
        {
            // get mobile_net_model file
            string fileName = "deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz";
            string url = $"http://download.tensorflow.org/models/{fileName}";
            Web.Download(url, dataDir, fileName);
            Compress.ExtractTGZ(Path.Join(dataDir, fileName), dataDir);

            // xception_model, better accuracy
            /*fileName = "deeplabv3_pascal_train_aug_2018_01_04.tar.gz";
            url = $"http://download.tensorflow.org/models/{fileName}";
            Web.Download(url, modelDir, fileName);
            Compress.ExtractTGZ(Path.Join(modelDir, fileName), modelDir);*/
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
    }
}
