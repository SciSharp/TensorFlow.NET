using System;
using System.Linq;

namespace Tensorflow.Keras.Preprocessings
{
    public partial class DatasetUtils
    {
        /// <summary>
        /// Potentially restict samples & labels to a training or validation split.
        /// </summary>
        /// <param name="samples"></param>
        /// <param name="labels"></param>
        /// <param name="validation_split"></param>
        /// <param name="subset"></param>
        /// <returns></returns>
        public (T1[], T2[]) get_training_or_validation_split<T1, T2>(T1[] samples,
            T2[] labels,
            float validation_split,
            string subset)
        {
            if (string.IsNullOrEmpty(subset))
                return (samples, labels);

            var num_val_samples = Convert.ToInt32(samples.Length * validation_split);
            if (subset == "training")
            {
                Binding.tf_output_redirect.WriteLine($"Using {samples.Length - num_val_samples} files for training.");
                samples = samples.Take(samples.Length - num_val_samples).ToArray();
                labels = labels.Take(labels.Length - num_val_samples).ToArray();
            }
            else if (subset == "validation")
            {
                Binding.tf_output_redirect.WriteLine($"Using {num_val_samples} files for validation.");
                samples = samples.Skip(samples.Length - num_val_samples).ToArray();
                labels = labels.Skip(labels.Length - num_val_samples).ToArray();
            }
            else
                throw new NotImplementedException("");

            return (samples, labels);
        }
    }
}
