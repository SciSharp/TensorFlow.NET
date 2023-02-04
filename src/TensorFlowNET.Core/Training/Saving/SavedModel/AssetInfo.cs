using System.Collections.Generic;

namespace Tensorflow;

public record class AssetInfo
(
    List<AssetFileDef> asset_defs,
    Dictionary<object, object> asset_initializers_by_resource,
    Dictionary<AssetInfo, string> asset_filename_map,
    Dictionary<object, object> asset_index
);
