using System;
using System.Collections.Generic;
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
            images.astype(dataType);
            images = np.multiply(images, 1.0f / 255.0f);
            Data = images;

            labels.astype(dataType);
            Labels = labels;
        }
    }
}
