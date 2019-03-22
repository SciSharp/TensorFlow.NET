using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.Examples.Utility
{
    public class DataSet
    {
        private int _num_examples;
        public int num_examples => _num_examples;
        private int _epochs_completed;
        public int epochs_completed => _epochs_completed;
        private int _index_in_epoch;
        public int index_in_epoch => _index_in_epoch;
        private NDArray _images;
        public NDArray images => _images;
        private NDArray _labels;
        public NDArray labels => _labels;

        public DataSet(NDArray images, NDArray labels, TF_DataType dtype, bool reshape)
        {
            _num_examples = images.shape[0];
            images = images.reshape(images.shape[0], images.shape[1] * images.shape[2]);
            images.astype(dtype.as_numpy_datatype());
            images = np.multiply(images, 1.0f / 255.0f);

            labels.astype(dtype.as_numpy_datatype());

            _images = images;
            _labels = labels;
            _epochs_completed = 0;
            _index_in_epoch = 0;
        }

        public (NDArray, NDArray) next_batch(int batch_size, bool fake_data = false, bool shuffle = true)
        {
            var start = _index_in_epoch;
            // Shuffle for the first epoch
            if(_epochs_completed == 0 && start == 0 && shuffle)
            {
                var perm0 = np.arange(_num_examples);
                np.random.shuffle(perm0);
                _images = images[perm0];
                _labels = labels[perm0];
            }

            // Go to the next epoch
            if (start + batch_size > _num_examples)
            {
                // Finished epoch
                _epochs_completed += 1;

                // Get the rest examples in this epoch
                var rest_num_examples = _num_examples - start;
                var images_rest_part = _images[np.arange(start, _num_examples)];
                var labels_rest_part = _labels[np.arange(start, _num_examples)];
                // Shuffle the data
                if (shuffle)
                {
                    var perm = np.arange(_num_examples);
                    np.random.shuffle(perm);
                    _images = images[perm];
                    _labels = labels[perm];
                }

                start = 0;
                _index_in_epoch = batch_size - rest_num_examples;
                var end = _index_in_epoch;
                var images_new_part = _images[np.arange(start, end)];
                var labels_new_part = _labels[np.arange(start, end)];

                /*return (np.concatenate(new float[][] { images_rest_part.Data<float>(), images_new_part.Data<float>() }, axis: 0),
                    np.concatenate(new float[][] { labels_rest_part.Data<float>(), labels_new_part.Data<float>() }, axis: 0));*/
                return (images_new_part, labels_new_part);
            }
            else
            {
                _index_in_epoch += batch_size;
                var end = _index_in_epoch;
                return (_images[np.arange(start, end)], _labels[np.arange(start, end)]);
            }
        }
    }
}
