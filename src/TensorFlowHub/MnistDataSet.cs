using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using NumSharp;
using Tensorflow;

namespace Tensorflow.Hub
{
    public class MnistDataSet : DataSetBase
    {
        public int NumOfExamples { get; private set; }
        public int EpochsCompleted { get; private set; }
        public int IndexInEpoch { get; private set; }

        public MnistDataSet(NDArray images, NDArray labels, Type dataType, bool reshape)
        {
            EpochsCompleted = 0;
            IndexInEpoch = 0;

            NumOfExamples = images.shape[0];

            images = images.reshape(images.shape[0], images.shape[1] * images.shape[2]);
            images = images.astype(dataType);
            // for debug np.multiply performance
            var sw = new Stopwatch();
            sw.Start();
            images = np.multiply(images, 1.0f / 255.0f);
            sw.Stop();
            Console.WriteLine($"{sw.ElapsedMilliseconds}ms");
            Data = images;

            labels = labels.astype(dataType);
            Labels = labels;
        }

        public (NDArray, NDArray) GetNextBatch(int batch_size, bool fake_data = false, bool shuffle = true)
        {
            var start = IndexInEpoch;
            // Shuffle for the first epoch
            if(EpochsCompleted == 0 && start == 0 && shuffle)
            {
                var perm0 = np.arange(NumOfExamples);
                np.random.shuffle(perm0);
                Data = Data[perm0];
                Labels = Labels[perm0];
            }

            // Go to the next epoch
            if (start + batch_size > NumOfExamples)
            {
                // Finished epoch
                EpochsCompleted += 1;

                // Get the rest examples in this epoch
                var rest_num_examples = NumOfExamples - start;
                //var images_rest_part = _images[np.arange(start, _num_examples)];
                //var labels_rest_part = _labels[np.arange(start, _num_examples)];
                // Shuffle the data
                if (shuffle)
                {
                    var perm = np.arange(NumOfExamples);
                    np.random.shuffle(perm);
                    Data = Data[perm];
                    Labels = Labels[perm];
                }

                start = 0;
                IndexInEpoch = batch_size - rest_num_examples;
                var end = IndexInEpoch;
                var images_new_part = Data[np.arange(start, end)];
                var labels_new_part = Labels[np.arange(start, end)];

                /*return (np.concatenate(new float[][] { images_rest_part.Data<float>(), images_new_part.Data<float>() }, axis: 0),
                    np.concatenate(new float[][] { labels_rest_part.Data<float>(), labels_new_part.Data<float>() }, axis: 0));*/
                return (images_new_part, labels_new_part);
            }
            else
            {
                IndexInEpoch += batch_size;
                var end = IndexInEpoch;
                return (Data[np.arange(start, end)], Labels[np.arange(start, end)]);
            }
        }
    }
}
