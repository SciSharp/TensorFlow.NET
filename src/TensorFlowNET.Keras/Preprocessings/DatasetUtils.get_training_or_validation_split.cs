using System;

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
            var num_val_samples = Convert.ToInt32(samples.Length * validation_split);
            if (subset == "training")
            {
                Console.WriteLine($"Using {samples.Length - num_val_samples} files for training.");
                samples = samples[..^num_val_samples];
                labels = labels[..^num_val_samples];
            }
            else if (subset == "validation")
            {
                Console.WriteLine($"Using {num_val_samples} files for validation.");
                samples = samples[(samples.Length - num_val_samples)..];
                labels = labels[(labels.Length - num_val_samples)..];
            }
            else
                throw new NotImplementedException("");

            return (samples, labels);
        }
    }
}
