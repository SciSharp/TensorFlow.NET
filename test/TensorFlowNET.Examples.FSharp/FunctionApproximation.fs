module FunctionApproximation

//reduced example from https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Function%20Approximation%20by%20Neural%20Network/Function%20approximation%20by%20linear%20model%20and%20deep%20network.ipynb

open NumSharp
open Tensorflow
open System

let run()=
    
    let N_points = 75 // Number of points for constructing function
    let x_min = 1.0 // Min of the range of x (feature)
    let x_max = 15.0 // Max of the range of x (feature)
    let noise_mean = 0.0 // Mean of the Gaussian noise adder
    let noise_sd = 10.0 // Std.Dev of the Gaussian noise adder

    let linspace points = [| for i in 0 .. (points - 1) -> x_min + (x_max - x_min)/(float)points * (float)i |]

    let  func_trans(xAr:float []) = 
        xAr
        |>Array.map (fun (x:float) -> (20.0 * x+3.0 * System.Math.Pow(x,2.0)+0.1 * System.Math.Pow(x,3.0))*sin(x)*exp(-0.1*x))
        
    let X_raw = linspace N_points
    let Y_raw = func_trans(X_raw)
    let X_mtr = Array2D.init X_raw.Length 1 (fun i j -> X_raw.[i])
    let X = np.array(X_mtr)

    let noise_x = np.random.normal(noise_mean,noise_sd,N_points)
    let y =  np.array(Y_raw)+noise_x

    let X_train = X
    let y_train = y

    let learning_rate = 0.00001
    let training_epochs = 35000
    
    let n_input = 1  // Number of features
    let n_output = 1  // Regression output is a number only
    let n_hidden_layer_1 = 25 // Hidden layer 1
    let n_hidden_layer_2 = 25 // Hidden layer 2

    let tf = Python.New<tensorflow>()
    let x = tf.placeholder(tf.float64, new TensorShape(N_points,n_input))
    let y = tf.placeholder(tf.float64, new TensorShape(n_output))
    
   
    let weights = dict[
        "hidden_layer_1", tf.Variable(tf.random_normal([|n_input; n_hidden_layer_1|],dtype=tf.float64))
        "hidden_layer_2", tf.Variable(tf.random_normal([|n_hidden_layer_1; n_hidden_layer_2|],dtype=tf.float64))
        "out", tf.Variable(tf.random_normal([|n_hidden_layer_2; n_output|],dtype=tf.float64))
    ]
    let biases = dict[
        "hidden_layer_1", tf.Variable(tf.random_normal([|n_hidden_layer_1|],dtype=tf.float64))
        "hidden_layer_2", tf.Variable(tf.random_normal([|n_hidden_layer_2|],dtype=tf.float64))
        "out", tf.Variable(tf.random_normal([|n_output|],dtype=tf.float64))
    ]

 
    // Hidden layer with RELU activation

    let layer_1 = tf.add(tf.matmul(x, weights.["hidden_layer_1"]._AsTensor()),biases.["hidden_layer_1"])
    let layer_1 = tf.nn.relu(layer_1)

    let layer_2 = tf.add(tf.matmul(layer_1, weights.["hidden_layer_2"]._AsTensor()),biases.["hidden_layer_2"])
    let layer_2 = tf.nn.relu(layer_2)

    // Output layer with linear activation
    let ops = tf.add(tf.matmul(layer_2, weights.["out"]._AsTensor()), biases.["out"])
    
    // Define loss and optimizer
    let cost = tf.reduce_mean(tf.square(tf.squeeze(ops)-y))

    let gs = tf.Variable(1, trainable= false, name= "global_step")

    let optimizer = tf.train.GradientDescentOptimizer(learning_rate=(float32)learning_rate).minimize(cost,global_step = gs)

    let init = tf.global_variables_initializer()
    
    
    Tensorflow.Python.``tf_with``(tf.Session(), fun (sess:Session) ->
        sess.run(init)  |> ignore  
        // Loop over epochs
        for epoch in [0..training_epochs] do
            // Run optimization process (backprop) and cost function (to get loss value)

            let result=sess.run([|optimizer:>ITensorOrOperation; gs._AsTensor():>ITensorOrOperation; cost:>ITensorOrOperation|], new FeedItem(x, X_train), new FeedItem(y, y_train))


            let loss_value = (double) result.[2];

            let step = (int) result.[1];
            
            if epoch % 1000 = 0 then
                sprintf "Step %d loss: %f" step loss_value |> Console.WriteLine
        let w=sess.run(weights |> Array.ofSeq |> Array.map (fun pair -> pair.Value))
        let b = sess.run(biases |> Array.ofSeq |> Array.map (fun pair -> pair.Value))
        let yhat=sess.run([|ops:>ITensorOrOperation|],new FeedItem(x,X_train))
        for i in [0..(N_points-1)] do
            sprintf "pred %f real: %f" ((double)(yhat.[0].[i].[0])) ((double)Y_raw.[i]) |> Console.WriteLine
    )




