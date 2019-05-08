using System;
using System.Collections.Generic;
using System.Text;
using TensorFlowNET.Examples.Utility;

namespace TensorFlowNET.Examples.ImageProcess
{
    /// <summary>
    /// This example removes the background from an input image.
    /// 
    /// https://github.com/susheelsk/image-background-removal
    /// </summary>
    public class ImageBackgroundRemoval : IExample
    {
        public int Priority => 15;

        public bool Enabled { get; set; } = true;
        public bool ImportGraph { get; set; } = true;

        public string Name => "Image Background Removal";

        string modelDir = "deeplabv3";

        public bool Run()
        {
            return false;
        }

        public void PrepareData()
        {
            // get model file
            string url = "http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz";
            Web.Download(url, modelDir, "deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz");
        }
    }
}
