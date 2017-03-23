Optimizers - Engineers nightmare 

If you know how a Gradient Descent and Back propagation works, Its time to learn how different optimizers work. There is a lot of math behind each and every optimizer and I am not going to touch upon that.

I will be using a basic 2 layer Feed forward neural network to demonstrate how different optimizers work.

FC784 - FC256 - FC64 - FC10
We will be using relu's as activation functions for hidden layers and softmax as activation function for output layer.


**epochs** = **100**
**batch_size** = **100**

We will be performing 3 Iterations 
- Constant Learning rate:
   - High Value -0.1 
   - Low Value - 0.001
   - Very low Value - 0.00001

- Using exponential Learning rate decay

- Manually Reduce the learning rate.
