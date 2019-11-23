# Chapter. Image Recognition

An example for using the [TensorFlow.NET](https://github.com/SciSharp/TensorFlow.NET) and [NumSharp](https://github.com/SciSharp/NumSharp) for image recognition, it will use a pre-trained inception model to predict a image which outputs the categories sorted by probability. The original paper is [here](https://arxiv.org/pdf/1512.00567.pdf). The Inception architecture of GoogLeNet was designed to perform well even under strict constraints on memory and computational budget. The computational cost of Inception is also much lower than other performing successors. This has made it feasible to utilize Inception networks in big-data scenarios, where huge amount of data needed to be processed at reasonable cost or scenarios where memory or computational capacity is inherently limited, for example in mobile vision settings.

The GoogLeNet architecture conforms to below design principles:

* Avoid representational bottlenecks, especially early in the network.
* Higher dimensional representations are easier to process locally within a network.
* Spatial aggregation can be done over lower dimensional embeddings without much or any loss in representational power.
* Balance the width and depth of the network.

#### Let's get started with real code.

##### 1. Prepare data

This example will download the dataset and uncompress it automatically. Some external paths are omitted, please refer to the source code for the real path.

```csharp
private void PrepareData()
{
    Directory.CreateDirectory(dir);

    // get model file
    string url = "models/inception_v3_2016_08_28_frozen.pb.tar.gz";

    string zipFile = Path.Join(dir, $"{pbFile}.tar.gz");
    Utility.Web.Download(url, zipFile);

    Utility.Compress.ExtractTGZ(zipFile, dir);

    // download sample picture
    string pic = "grace_hopper.jpg";
    Utility.Web.Download($"data/{pic}", Path.Join(dir, pic));
}
```

##### 2. Load image file and normalize

We need to load a sample image to test our pre-trained inception model. Convert it into tensor and normalized the input image. The pre-trained model takes input in the form of a 4-dimensional tensor with shape [BATCH_SIZE, INPUT_HEIGHT, INPUT_WEIGHT, 3] where:

- BATCH_SIZE allows for inference of multiple images in one pass through the graph
- INPUT_HEIGHT is the height of the images on which the model was trained
- INPUT_WEIGHT is the width of the images on which the model was trained
- 3 is the (R, G, B) values of the pixel colors represented as a float.

```csharp
private NDArray ReadTensorFromImageFile(string file_name,
                                int input_height = 299,
                                int input_width = 299,
                                int input_mean = 0,
                                int input_std = 255)
{
	return with<Graph, NDArray>(tf.Graph().as_default(), graph =>
    {
		var file_reader = tf.read_file(file_name, "file_reader");
        var image_reader = tf.image.decode_jpeg(file_reader, channels: 3, name: "jpeg_reader");
        var caster = tf.cast(image_reader, tf.float32);
        var dims_expander = tf.expand_dims(caster, 0);
        var resize = tf.constant(new int[] { input_height, input_width });
        var bilinear = tf.image.resize_bilinear(dims_expander, resize);
        var sub = tf.subtract(bilinear, new float[] { input_mean });
        var normalized = tf.divide(sub, new float[] { input_std });

		return with<Session, NDArray>(tf.Session(graph), sess => sess.run(normalized));
    });
}
```

##### 3. Load pre-trained model and predict

Load the pre-trained inception model which is saved as Google's protobuf file format. Construct a new graph then set input and output operations in a new session. After run the session, you will get a numpy-like ndarray which is provided by NumSharp. With NumSharp, you can easily perform various operations on multiple dimensional arrays in the .NET environment.

```csharp
public void Run()
{
	PrepareData();

	var labels = File.ReadAllLines(Path.Join(dir, labelFile));

    var nd = ReadTensorFromImageFile(Path.Join(dir, picFile),
        input_height: input_height,
        input_width: input_width,
        input_mean: input_mean,
        input_std: input_std);

    var graph = Graph.ImportFromPB(Path.Join(dir, pbFile));
    var input_operation = graph.get_operation_by_name(input_name);
    var output_operation = graph.get_operation_by_name(output_name);

    var results = with<Session, NDArray>(tf.Session(graph),
    	sess => sess.run(output_operation.outputs[0], 
        	new FeedItem(input_operation.outputs[0], nd)));

	results = np.squeeze(results);

    var argsort = results.argsort<float>();
    var top_k = argsort.Data<float>()
        .Skip(results.size - 5)
        .Reverse()
        .ToArray();

    foreach (float idx in top_k)
    	Console.WriteLine($"{picFile}: {idx} {labels[(int)idx]}, {results[(int)idx]}");
}
```

##### 4. Print the result

The best probability is `military uniform` which is 0.8343058. It's the correct classification.

```powershell
2/18/2019 3:56:18 AM Starting InceptionArchGoogLeNet
label_image_data\inception_v3_2016_08_28_frozen.pb.tar.gz already exists.
label_image_data\grace_hopper.jpg already exists.
2019-02-19 21:56:18.684463: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
create_op: Const 'file_reader/filename', inputs: empty, control_inputs: empty, outputs: file_reader/filename:0
create_op: ReadFile 'file_reader', inputs: file_reader/filename:0, control_inputs: empty, outputs: file_reader:0
create_op: DecodeJpeg 'jpeg_reader', inputs: file_reader:0, control_inputs: empty, outputs: jpeg_reader:0
create_op: Cast 'Cast/Cast', inputs: jpeg_reader:0, control_inputs: empty, outputs: Cast/Cast:0
create_op: Const 'ExpandDims/dim', inputs: empty, control_inputs: empty, outputs: ExpandDims/dim:0
create_op: ExpandDims 'ExpandDims', inputs: Cast/Cast:0, ExpandDims/dim:0, control_inputs: empty, outputs: ExpandDims:0
create_op: Const 'Const', inputs: empty, control_inputs: empty, outputs: Const:0
create_op: ResizeBilinear 'ResizeBilinear', inputs: ExpandDims:0, Const:0, control_inputs: empty, outputs: ResizeBilinear:0
create_op: Const 'y', inputs: empty, control_inputs: empty, outputs: y:0
create_op: Sub 'Sub', inputs: ResizeBilinear:0, y:0, control_inputs: empty, outputs: Sub:0
create_op: Const 'y_1', inputs: empty, control_inputs: empty, outputs: y_1:0
create_op: RealDiv 'truediv', inputs: Sub:0, y_1:0, control_inputs: empty, outputs: truediv:0
grace_hopper.jpg: 653 military uniform, 0.8343058
grace_hopper.jpg: 668 mortarboard, 0.02186947
grace_hopper.jpg: 401 academic gown, 0.01035806
grace_hopper.jpg: 716 pickelhaube, 0.008008132
grace_hopper.jpg: 466 bulletproof vest, 0.005350832
2/18/2019 3:56:25 AM Completed InceptionArchGoogLeNet
```

You can find the full source code from [github](https://github.com/SciSharp/TensorFlow.NET-Examples/tree/master/src/TensorFlowNET.Examples/ImageProcessing). 

