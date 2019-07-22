/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using NumSharp;
using Tensorflow;

namespace TensorFlowNET.Examples.Utility
{
    public class DataSetMnist : IDataSet
    {
        public int num_examples { get; }

        public int epochs_completed { get; private set; }
        public int index_in_epoch { get; private set; }
        public NDArray data { get; private set; }
        public NDArray labels { get; private set; }

        public DataSetMnist(NDArray images, NDArray labels, TF_DataType dtype, bool reshape)
        {
            num_examples = images.shape[0];
            images = images.reshape(images.shape[0], images.shape[1] * images.shape[2]);
            images.astype(dtype.as_numpy_datatype());
            images = np.multiply(images, 1.0f / 255.0f);

            labels.astype(dtype.as_numpy_datatype());

            data = images;
            this.labels = labels;
            epochs_completed = 0;
            index_in_epoch = 0;
        }

        public (NDArray, NDArray) next_batch(int batch_size, bool fake_data = false, bool shuffle = true)
        {
            var start = index_in_epoch;
            // Shuffle for the first epoch
            if(epochs_completed == 0 && start == 0 && shuffle)
            {
                var perm0 = np.arange(num_examples);
                np.random.shuffle(perm0);
                data = data[perm0];
                labels = labels[perm0];
            }

            // Go to the next epoch
            if (start + batch_size > num_examples)
            {
                // Finished epoch
                epochs_completed += 1;

                // Get the rest examples in this epoch
                var rest_num_examples = num_examples - start;
                //var images_rest_part = _images[np.arange(start, _num_examples)];
                //var labels_rest_part = _labels[np.arange(start, _num_examples)];
                // Shuffle the data
                if (shuffle)
                {
                    var perm = np.arange(num_examples);
                    np.random.shuffle(perm);
                    data = data[perm];
                    labels = labels[perm];
                }

                start = 0;
                index_in_epoch = batch_size - rest_num_examples;
                var end = index_in_epoch;
                var images_new_part = data[np.arange(start, end)];
                var labels_new_part = labels[np.arange(start, end)];

                /*return (np.concatenate(new float[][] { images_rest_part.Data<float>(), images_new_part.Data<float>() }, axis: 0),
                    np.concatenate(new float[][] { labels_rest_part.Data<float>(), labels_new_part.Data<float>() }, axis: 0));*/
                return (images_new_part, labels_new_part);
            }
            else
            {
                index_in_epoch += batch_size;
                var end = index_in_epoch;
                return (data[np.arange(start, end)], labels[np.arange(start, end)]);
            }
        }
    }
}
