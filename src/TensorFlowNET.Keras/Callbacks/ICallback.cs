using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Callbacks
{
    public interface ICallback
    {
        void on_train_begin();
        void on_epoch_begin(int epoch);
        void on_train_batch_begin(long step);
        void on_train_batch_end(long end_step, Dictionary<string, float> logs);
        void on_epoch_end(int epoch, Dictionary<string, float> epoch_logs);
    }
}
