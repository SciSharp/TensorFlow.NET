using System.IO;
using System.Security.Cryptography.X509Certificates;
using Tensorflow.Train;
using static Tensorflow.Binding;

namespace Tensorflow;

public static partial class SavedModelUtils
{
    /// <summary>
    /// Return variables sub-directory, or create one if it doesn't exist.
    /// </summary>
    /// <returns></returns>
    public static string get_or_create_variables_dir(string export_dir)
    {
        var variables_dir = get_variables_dir(export_dir);
        Directory.CreateDirectory(variables_dir);
        return variables_dir;
    }

    /// <summary>
    /// Return variables sub-directory in the SavedModel.
    /// </summary>
    /// <param name="export_dir"></param>
    /// <returns></returns>
    public static string get_variables_dir(string export_dir)
    {
        return Path.Combine(tf.compat.as_text(export_dir), tf.compat.as_text(Constants.VARIABLES_DIRECTORY));
    }

    public static string get_variables_path(string export_dir)
    {
        return Path.Combine(tf.compat.as_text(get_variables_dir(export_dir)), tf.compat.as_text(Constants.VARIABLES_FILENAME));
    }

    /// <summary>
    /// Return assets sub-directory, or create one if it doesn't exist.
    /// </summary>
    /// <param name="export_dir"></param>
    /// <returns></returns>
    public static string get_or_create_assets_dir(string export_dir)
    {
        var assets_destination_dir = get_assets_dir(export_dir);
        Directory.CreateDirectory(assets_destination_dir);
        return assets_destination_dir;
    }

    /// <summary>
    /// Return path to asset directory in the SavedModel.
    /// </summary>
    /// <param name="export_dir"></param>
    /// <returns></returns>
    public static string get_assets_dir(string export_dir)
    {
        return Path.Combine(tf.compat.as_text(export_dir), tf.compat.as_text(Constants.ASSETS_DIRECTORY));
    }
}
