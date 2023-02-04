namespace Tensorflow;

public static class Constants
{
    public static readonly string ASSETS_DIRECTORY = "assets";
    public static readonly string ASSETS_KEY = "saved_model_assets";

    public static readonly string DEBUG_DIRECTORY = "debug";

    public static readonly string DEBUG_INFO_FILENAME_PB = "saved_model_debug_info.pb";

    public static readonly string EXTRA_ASSETS_DIRECTORY = "assets.extra";

    public static readonly string FINGERPRINT_FILENAME = "fingerprint.pb";

    public static readonly string INIT_OP_SIGNATURE_KEY = "__saved_model_init_op";

    public static readonly string LEGACY_INIT_OP_KEY = "legacy_init_op";

    public static readonly string MAIN_OP_KEY = "saved_model_main_op";

    public static readonly string SAVED_MODEL_FILENAME_PB = "saved_model.pb";
    public static readonly string SAVED_MODEL_FILENAME_PBTXT = "saved_model.pbtxt";

    public static readonly int SAVED_MODEL_SCHEMA_VERSION = 1;

    public static readonly string TRAIN_OP_KEY = "saved_model_train_op";

    public static readonly string TRAIN_OP_SIGNATURE_KEY = "__saved_model_train_op";

    public static readonly string VARIABLES_DIRECTORY = "variables";
    public static readonly string VARIABLES_FILENAME = "variables";
}
