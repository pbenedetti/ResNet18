# ResNet18 + train

Start_block:
  7x7 convo, stride 2
  3x3 maxpool, stride 2

ResBlock (increasing number of filter)
  1x1 convo, padding valid ( only if stride 2 )
  3x3 convo
  3x3 convo
  
Class_block:
  avg_pool
  softmax

Batch normalizarion adopted right after each convolution and before activation
Drop_out 80%
