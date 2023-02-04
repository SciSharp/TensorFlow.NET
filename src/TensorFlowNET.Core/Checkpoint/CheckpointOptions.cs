namespace Tensorflow.Checkpoint;

public record class CheckpointOptions(
    string? experimental_io_device = null, 
    bool experimental_enable_async_checkpoint = false);
