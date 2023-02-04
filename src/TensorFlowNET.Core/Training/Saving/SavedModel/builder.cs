using System;
using System.Collections.Generic;
using static Tensorflow.Binding;

namespace Tensorflow;

public class BuilderUtils
{
    public static void copy_assets_to_destination_dir(IDictionary<AssetInfo, string> asset_filename_map,
        string destination_dir, HashSet<string>? saved_files = null)
    {
        if (saved_files is null) saved_files = new HashSet<string>();

        var asset_destination_dir = SavedModelUtils.get_or_create_assets_dir(destination_dir);

        // TODO: complete the implementation of this function.
        if (asset_filename_map is not null && asset_filename_map.Count > 0)
        {
            throw new NotImplementedException();
        }
    }
}
