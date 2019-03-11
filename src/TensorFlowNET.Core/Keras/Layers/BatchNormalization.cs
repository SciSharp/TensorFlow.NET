using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Layers;

namespace Tensorflow.Keras.Layers
{
    public class BatchNormalization : Layer
    {
        private bool _USE_V2_BEHAVIOR = true;
        private float momentum;
        private float epsilon;
        private bool center;
        private bool scale;
        private bool renorm;
        private bool fused;
        private bool _bessels_correction_test_only;
        private int[] axis;
        private string _data_format;
        private IInitializer beta_initializer;
        private IInitializer gamma_initializer;
        private IInitializer moving_mean_initializer;
        private IInitializer moving_variance_initializer;
        private RefVariable gamma;
        private RefVariable beta;
        private RefVariable moving_mean;

        public BatchNormalization(int axis = -1,
            float momentum = 0.99f,
            float epsilon = 0.001f,
            bool center = true,
            bool scale = true,
            IInitializer beta_initializer = null,
            IInitializer gamma_initializer = null,
            IInitializer moving_mean_initializer = null,
            IInitializer moving_variance_initializer = null,
            bool renorm = false,
            float renorm_momentum = 0.99f,
            bool trainable = true,
            string name = null) : base(trainable: trainable, 
                name: name)
        {
            this.axis = new int[] { axis };
            this.momentum = momentum;
            this.epsilon = epsilon;
            this.center = center;
            this.scale = scale;
            if (beta_initializer == null)
                beta_initializer = tf.zeros_initializer;
            if (gamma_initializer == null)
                gamma_initializer = tf.ones_initializer;
            if (moving_mean_initializer == null)
                moving_mean_initializer = tf.zeros_initializer;
            if (moving_variance_initializer == null)
                moving_variance_initializer = tf.ones_initializer;
            this.beta_initializer = beta_initializer;
            this.gamma_initializer = gamma_initializer;
            this.moving_mean_initializer = moving_mean_initializer;
            this.moving_variance_initializer = moving_variance_initializer;
            this.renorm = renorm;
            this.fused = true;
            this.supports_masking = true;
            this._bessels_correction_test_only = true;
        }

        protected override void build(TensorShape input_shape)
        {
            var ndims = input_shape.NDim;
            foreach (var (idx, x) in Python.enumerate(axis))
                if (x < 0)
                    axis[idx] = ndims + x;

            if (fused)
                if (Enumerable.SequenceEqual(axis, new int[] { 3 }))
                    _data_format = "NHWC";

            var param_dtype = _dtype == TF_DataType.DtInvalid ? TF_DataType.TF_FLOAT : _dtype;
            var param_shape = new int[] { input_shape.Dimensions[axis[0]] };

            if (scale)
                gamma = add_weight("gamma",
                    param_shape,
                    dtype: param_dtype,
                    initializer: gamma_initializer,
                    trainable: true);
            else
                throw new NotImplementedException("add_weight gamma");

            if (center)
                beta = add_weight("beta",
                    param_shape,
                    dtype: param_dtype,
                    initializer: beta_initializer,
                    trainable: true);
            else
                throw new NotImplementedException("add_weight beta");

            if(_scope != null)
            {
                
            }

            moving_mean = add_weight("moving_mean",
                param_shape,
                dtype: param_dtype);
        }
    }
}
