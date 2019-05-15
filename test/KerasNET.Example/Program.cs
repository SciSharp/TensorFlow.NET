using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using static Tensorflow.Python;
using static Keras.Keras;
using Keras.Layers;
using Keras;
using NumSharp;

namespace Keras.Example
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("================================== Keras ==================================");

            #region data
            var batch_size = 1000;
            var (X, Y) = XOR(batch_size);
            //var (X, Y, batch_size) = (np.array(new float[,]{{1, 0 },{1, 1 },{0, 0 },{0, 1 }}), np.array(new int[] { 0, 1, 1, 0 }), 4);
            #endregion

            #region features
            var (features, labels) = (new Tensor(X), new Tensor(Y));
            var num_steps = 10000;
            #endregion

            #region model
            var m = new Model();
            
            //m.Add(new Dense(8, name: "Hidden", activation: tf.nn.relu())).Add(new Dense(1, name:"Output"));

            m.Add(
                new ILayer[] {
                    new Dense(8, name: "Hidden_1", activation: tf.nn.relu()),
                    new Dense(1, name: "Output")
                });

            m.train(num_steps, (X, Y));
            #endregion

            Console.ReadKey();
        }
        static (NDArray, NDArray) XOR(int samples)
        {
            var X = new List<float[]>();
            var Y = new List<float>();
            var r = new Random();
            for (int i = 0; i < samples; i++)
            {
                var x1 = (float)r.Next(0, 2);
                var x2 = (float)r.Next(0, 2);
                var y = 0.0f;
                if (x1 == x2)
                    y = 1.0f;
                X.Add(new float[] { x1, x2 });
                Y.Add(y);
            }
            
            return (np.array(X.ToArray()), np.array(Y.ToArray()));
        }
    }
}
