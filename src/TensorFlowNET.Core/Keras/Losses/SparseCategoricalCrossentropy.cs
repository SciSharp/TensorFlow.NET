using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Losses
{
    public class SparseCategoricalCrossentropy : LossFunctionWrapper, ILossFunc
    {
        public SparseCategoricalCrossentropy(bool from_logits = false,
            string reduction = ReductionV2.AUTO,
            string name = "sparse_categorical_crossentropy") : 
            base(sparse_categorical_crossentropy, 
                reduction: reduction, 
                name: name)
        {

        }

        static void sparse_categorical_crossentropy()
        {

        }
    }
}
