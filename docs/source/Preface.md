

# Preface

Why do I start the TensorFlow.NET project?

In a few days, it was Christmas in 2018. I watched my children grow up and be sensible every day, and I felt that time passed too fast. IT technology updates are faster than ever, and a variety of front-end technologies are emerging. Big data, Artificial Intelligence and Blockchain, Container technology and Microservices, Distributed Computing and Serverless technology are dazzling. The Amazon AI service interface claims that engineers who don't need any machine learning experience can use it, so that the idea of just calming down for two years and planning to switch to an AI architecture in the future is a splash of cold water.

再过几天就是2018年圣诞节，看着孩子一天天长大并懂事，感慨时间过得太快。IT技术更新换代比以往任何时候都更快，各种前后端技术纷纷涌现。大数据，人工智能和区块链，容器技术和微服务，分布式计算和无服务器技术，让人眼花缭乱。Amazon AI服务接口宣称不需要具有任何机器学习经验的工程师就能使用，让像我这样刚静下心来学习了两年并打算将来转行做AI架构的想法泼了一桶凉水。



TensorFlow is an open source project for machine learning especially for deep learning. It's used for both research and production at Google company. It's designed according to dataflow programming pattern across a range of tasks. TensorFlow is not just a deep learning library. As long as you can represent your calculation process as a data flow diagram, you can use TensorFlow for distributed computing. TensorFlow uses a computational graph to build a computing network while operating on the graph. Users can write their own upper-level models in Python based on TensorFlow, or extend the underlying C++ custom action code to TensorFlow.

TensorFlow是一个用于机器学习的开源项目，尤其适用于深度学习。 它最初是谷歌公司的用于内部研究和生产的工具，后来开源出来给社区使用。TensorFlow并不仅仅是一个深度学习库，只要可以把你的计算过程表示称一个数据流图的过程，就可以使用TensorFlow来进行分布式计算。TensorFlow用计算图的方式建立计算网络，同时对图进行操作。用户可以基于TensorFlow的基础上用python编写自己的上层模型，也可以扩展底层的C++自定义操作代码添加到TensorFlow中。



In order to avoid confusion, the unique classes defined in TensorFlow are not translated in this book. For example, Tensor, Graph, Shape will retain the English name.

为了避免混淆，本书中对TensorFlow中定义的特有类不进行翻译，比如Tensor, Graph, Shape这些词都会保留英文名称。



Terminology:

TF: Google TensorFlow

TF.NET: TensorFlow.NET