using Newtonsoft.Json;
using NumSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Tensorflow;
using TensorFlowNET.Examples.Utility;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;

namespace TensorFlowNET.Examples
{

    public class ObjectDetection : Python, IExample
    {
        public int Priority => 11;
        public bool Enabled { get; set; } = true;
        public string Name => "Object Detection";
        public float MIN_SCORE = 0.5f;

        string modelDir = "ssd_mobilenet_v1_coco_2018_01_28";
        string imageDir = "images";
        string pbFile = "frozen_inference_graph.pb";
        string labelFile = "mscoco_label_map.pbtxt";
        string picFile = "input.jpg";

        public bool Run()
        {
            //buildOutputImage(null);

            // read in the input image
            var imgArr = ReadTensorFromImageFile(Path.Join(imageDir, "input.jpg"));

            var graph = new Graph().as_default();          
            graph.Import(Path.Join(modelDir, pbFile));

            var tensorNum = graph.OperationByName("num_detections").outputs[0];
            var tensorBoxes = graph.OperationByName("detection_boxes").outputs[0];
            var tensorScores = graph.OperationByName("detection_scores").outputs[0];
            var tensorClasses = graph.OperationByName("detection_classes").outputs[0];

            var imgTensor = graph.OperationByName("image_tensor").outputs[0];

            

            Tensor[] outTensorArr = new Tensor[] { tensorNum, tensorBoxes, tensorScores, tensorClasses };

            with(tf.Session(graph), sess =>
            {
                var results = sess.run(outTensorArr, new FeedItem(imgTensor, imgArr));
                
                NDArray[] resultArr = results.Data<NDArray>();
                
                buildOutputImage(resultArr);
            });

            return true;
        }

        public void PrepareData()
        {
            if (!Directory.Exists(modelDir))
                Directory.CreateDirectory(modelDir);

            if (!File.Exists(Path.Join(modelDir, "ssd_mobilenet_v1_coco.tar.gz")))
            {
                // get model file
                string url = "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz";

                Utility.Web.Download(url, modelDir, "ssd_mobilenet_v1_coco.tar.gz");
            }

            if (!File.Exists(Path.Join(modelDir, "frozen_inference_graph.pb")))
            {
                Utility.Compress.ExtractTGZ(Path.Join(modelDir, "ssd_mobilenet_v1_coco.tar.gz"), "./");
            }


            // download sample picture
            if (!Directory.Exists(imageDir))
                Directory.CreateDirectory(imageDir);

            if (!File.Exists(Path.Join(imageDir, "input.jpg")))
            {
                string url = $"https://github.com/tensorflow/models/raw/master/research/object_detection/test_images/image2.jpg";
                Utility.Web.Download(url, imageDir, "input.jpg");
            }

            // download the pbtxt file
            if (!File.Exists(Path.Join(modelDir, "mscoco_label_map.pbtxt")))
            {
                string url = $"https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt";
                Utility.Web.Download(url, modelDir, "mscoco_label_map.pbtxt");
            }
        }

        private NDArray ReadTensorFromImageFile(string file_name)
        {
            return with(tf.Graph().as_default(), graph =>
            {
                var file_reader = tf.read_file(file_name, "file_reader");
                var decodeJpeg = tf.image.decode_jpeg(file_reader, channels: 3, name: "DecodeJpeg");
                var casted = tf.cast(decodeJpeg, TF_DataType.TF_UINT8);
                var dims_expander = tf.expand_dims(casted, 0);
                return with(tf.Session(graph), sess => sess.run(dims_expander));
            });
        }

        private void buildOutputImage(NDArray[] resultArr)
        {
            // get pbtxt items
            PbtxtItems pbTxtItems = PbtxtParser.ParsePbtxtFile(Path.Join(modelDir, "mscoco_label_map.pbtxt"));

            // get bitmap
            Bitmap bitmap = new Bitmap(Path.Join(imageDir, "input.jpg"));

            float[] scores = resultArr[2].Data<float>();

            for (int i=0; i<scores.Length; i++)
            {
                float score = scores[i];
                if (score > MIN_SCORE)
                {
                    //var boxes = resultArr[1].Data<float[,,]>();
                    float[] boxes = resultArr[1].Data<float>();
                    float top = boxes[i * 4] * bitmap.Height;
                    float left = boxes[i * 4 + 1] * bitmap.Width;
                    float bottom = boxes[i * 4 + 2] * bitmap.Height;
                    float right = boxes[i * 4 + 3] * bitmap.Width;

                    Rectangle rect = new Rectangle()
                    {
                        X = (int)left,
                        Y = (int)top,
                        Width = (int)(right - left),
                        Height = (int)(bottom - top)
                    };

                    float[] ids = resultArr[3].Data<float>();

                    string name = pbTxtItems.items.Where(w => w.id == (int)ids[i]).Select(s=>s.display_name).FirstOrDefault();

                    drawObjectOnBitmap(bitmap, rect, score, name);
                }
            }

            bitmap.Save(Path.Join(imageDir, "output.jpg"));
        }

        private void drawObjectOnBitmap(Bitmap bmp, Rectangle rect, float score, string name)
        {
            using (Graphics graphic = Graphics.FromImage(bmp))
            {
                graphic.SmoothingMode = SmoothingMode.AntiAlias;
                
                using (Pen pen = new Pen(Color.Red, 2))
                {
                    graphic.DrawRectangle(pen, rect);

                    Point p = new Point(rect.Right + 5, rect.Top + 5);
                    string text = string.Format("{0}:{1}%", name, (int)(score * 100));
                    graphic.DrawString(text, new Font("Verdana", 8), Brushes.Red, p);
                }
            }
        }
    }
}
