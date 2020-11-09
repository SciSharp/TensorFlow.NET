using System;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Preprocessings
{
    public partial class DatasetUtils
    {
        public IDatasetV2 labels_to_dataset(int[] labels, string label_mode, int num_classes)
        {
            var label_ds = tf.data.Dataset.from_tensor_slices(labels);
            if (label_mode == "binary")
                throw new NotImplementedException("");
            else if (label_mode == "categorical")
                throw new NotImplementedException("");
            return label_ds;
        }
    }
}
