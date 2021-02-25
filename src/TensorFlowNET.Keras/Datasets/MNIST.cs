/*****************************************************************************
   Copyright 2020 Haiping Chen. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using NumSharp;
using System;
using System.IO;
using Tensorflow.Keras.Utils;

namespace Tensorflow.Keras.Datasets
{
    public class Mnist
    {
        string origin_folder = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/";
        string file_name = "mnist.npz";

        /// <summary>
        /// Loads the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).
        /// </summary>
        /// <returns></returns>
        public DatasetPass load_data()
        {
            var file = Download();
            var bytes = File.ReadAllBytes(file);
            var datax = LoadX(bytes);
            var datay = LoadY(bytes);
            return new DatasetPass
            {
                Train = (datax.Item1, datay.Item1),
                Test = (datax.Item2, datay.Item2)
            };
        }

        (NDArray, NDArray) LoadX(byte[] bytes)
        {
            var x = np.Load_Npz<byte[,,]>(bytes);
            return (x["x_train.npy"], x["x_test.npy"]);
        }

        (NDArray, NDArray) LoadY(byte[] bytes)
        {
            var y = np.Load_Npz<byte[]>(bytes);
            return (y["y_train.npy"], y["y_test.npy"]);
        }

        string Download()
        {
            var fileSaveTo = Path.Combine(Path.GetTempPath(), file_name);

            if (File.Exists(fileSaveTo))
            {
                Binding.tf_output_redirect.WriteLine($"The file {fileSaveTo} already exists");
                return fileSaveTo;
            }

            Web.Download(origin_folder + file_name, Path.GetTempPath(), file_name);

            return fileSaveTo;
        }
    }
}
