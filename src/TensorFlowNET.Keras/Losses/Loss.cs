using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Losses
{
    public abstract class Loss
    {
        public static Tensor mean_squared_error(Tensor y_true, Tensor y_pred) => throw new NotImplementedException();

        public static Tensor mean_absolute_error(Tensor y_true, Tensor y_pred) => throw new NotImplementedException();

        public static Tensor mean_absolute_percentage_error(Tensor y_true, Tensor y_pred) => throw new NotImplementedException();

        public static Tensor mean_squared_logarithmic_error(Tensor y_true, Tensor y_pred) => throw new NotImplementedException();

        public static Tensor _maybe_convert_labels(Tensor y_true) => throw new NotImplementedException();

        public static Tensor squared_hinge(Tensor y_true, Tensor y_pred) => throw new NotImplementedException();

        public static Tensor hinge(Tensor y_true, Tensor y_pred) => throw new NotImplementedException();

        public static Tensor categorical_hinge(Tensor y_true, Tensor y_pred) => throw new NotImplementedException();

        public static Tensor huber_loss(Tensor y_true, Tensor y_pred, float delta = 1) => throw new NotImplementedException();

        public static Tensor logcosh(Tensor y_true, Tensor y_pred) => throw new NotImplementedException();

        public static Tensor categorical_crossentropy(Tensor y_true, Tensor y_pred, bool from_logits = false, float label_smoothing = 0) => throw new NotImplementedException();

        public static Tensor sparse_categorical_crossentropy(Tensor y_true, Tensor y_pred, bool from_logits = false, float axis = -1) => throw new NotImplementedException();

        public static Tensor binary_crossentropy(Tensor y_true, Tensor y_pred, bool from_logits = false, float label_smoothing = 0) => throw new NotImplementedException();

        public static Tensor kullback_leibler_divergence(Tensor y_true, Tensor y_pred) => throw new NotImplementedException();

        public static Tensor poisson(Tensor y_true, Tensor y_pred) => throw new NotImplementedException();

        public static Tensor cosine_similarity(Tensor y_true, Tensor y_pred, int axis = -1) => throw new NotImplementedException();
    }
}
