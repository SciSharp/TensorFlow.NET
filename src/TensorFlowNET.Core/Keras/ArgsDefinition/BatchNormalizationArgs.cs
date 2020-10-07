using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class BatchNormalizationArgs : LayerArgs
    {
        public TensorShape Axis { get; set; } = -1;
        public float Momentum { get; set; } = 0.99f;
        public float Epsilon { get; set; } = 1e-3f;
        public bool Center { get; set; } = true;
        public bool Scale { get; set; } = true;
        public IInitializer BetaInitializer { get; set; } = tf.zeros_initializer;
        public IInitializer GammaInitializer { get; set; } = tf.ones_initializer;
        public IInitializer MovingMeanInitializer { get; set; } = tf.zeros_initializer;
        public IInitializer MovingVarianceInitializer { get; set; } = tf.ones_initializer;
        public IRegularizer BetaRegularizer { get; set; }
        public IRegularizer GammaRegularizer { get; set; }
        public bool Renorm { get; set; }
        public float RenormMomentum { get; set; } = 0.99f;
    }
}
