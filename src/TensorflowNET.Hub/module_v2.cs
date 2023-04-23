using System.IO;
using Tensorflow.Train;

namespace Tensorflow.Hub
{
    internal static class module_v2
    {
        public static Trackable load(string handle, LoadOptions? options)
        {
            var module_path = resolve(handle);

            // TODO(Rinne): deal with is_hub_module_v1

            var saved_model_path = Path.Combine(module_path, Constants.SAVED_MODEL_FILENAME_PB);
            var saved_model_pb_txt_path = Path.Combine(module_path, Constants.SAVED_MODEL_FILENAME_PBTXT);
            if (!File.Exists(saved_model_path) && !Directory.Exists(saved_model_path) && !File.Exists(saved_model_pb_txt_path)
                && !Directory.Exists(saved_model_pb_txt_path))
            {
                throw new ValueError($"Trying to load a model of incompatible/unknown type. " +
                    $"'{module_path}' contains neither '{Constants.SAVED_MODEL_FILENAME_PB}' " +
                    $"nor '{Constants.SAVED_MODEL_FILENAME_PBTXT}'.");
            }

            var obj = Loader.load(module_path, options: options);
            return obj;
        }

        public static string resolve(string handle)
        {
            return MultiImplRegister.GetResolverRegister().Call(handle);
        }
    }
}
