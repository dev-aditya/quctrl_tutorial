This folder contains a Julia reimplementation of the optimal control algorithms from the original code base.  
The functionality is equivalent, but several components are implemented differently.

Instead of using the original diagonalization method to compute the analytical gradient of the matrix exponential,  
this version uses the Fréchet derivative, which is conceptually clearer and results in more compact code.

You will also notice syntax and idiomatic differences that naturally arise from moving from the Python ecosystem to Julia.