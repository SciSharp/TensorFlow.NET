using System;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class MultiHeadAttentionArgs : LayerArgs
    {
        public int NumHeads { get; set; }
        public int KeyDim { get; set; }
        public int? ValueDim { get; set; } = null;
        public float Dropout { get; set; } = 0f;
        public bool UseBias { get; set; } = true;
        public Shape OutputShape { get; set; } = null;
        public Shape AttentionAxis { get; set; } = null;
        public IInitializer KernelInitializer { get; set; } = tf.glorot_uniform_initializer;
        public IInitializer BiasInitializer { get; set; } = tf.zeros_initializer;
        public IRegularizer KernelRegularizer { get; set; } = null;
        public IRegularizer BiasRegularizer { get; set; } = null;
        public Action KernelConstraint { get; set; } = null;
        public Action BiasConstraint { get; set; } = null;
    }
}