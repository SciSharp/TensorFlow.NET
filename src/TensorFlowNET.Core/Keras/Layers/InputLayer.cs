using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Layers
{
    /// <summary>
    /// Layer to be used as an entry point into a Network (a graph of layers).
    /// </summary>
    public class InputLayer : Layer
    {
        public bool sparse;
        public int? batch_size;

        public InputLayer(int[] input_shape = null,
            int? batch_size = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            string name = null,
            bool sparse = false,
            Tensor input_tensor = null)
        {
            built = true;
            this.sparse = sparse;
            this.batch_size = batch_size;
            this.supports_masking = true;

            if(input_tensor == null)
            {
                var batch_input_shape = new int[] { batch_size.HasValue ? batch_size.Value : -1, -1 };

                if (sparse)
                {
                    throw new NotImplementedException("InputLayer sparse is true");
                }
                else
                {
                    input_tensor = backend.placeholder(
                          shape: batch_input_shape,
                          dtype: dtype,
                          name: name);
                }
            }
        }
    }
}
