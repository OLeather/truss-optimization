# Truss Optimization
This repo was made as an extension to a school project from MTE 119: Statics. The goal of the project was to design a low-cost truss bridge that would support a train of given mass without
exceeding given maximum tension and compression forces within each member.

## Methodology
Our group experimented with a variety of methodologies for this project and finally settled on a numeric optimization problem: We would input a rough un-optimized shape for the bridge and
the program would optimize the joint positions to minimize the cost while maintaining the constraints. Other ideas we explored involved neural networks to design unique shapes to be 
optimized, but we deemed it out of the scope of this project. Our final approach used the CasADI library to represent the cost function symbolically and mimimize through nonlinear optimization. 

### Truss Solver
The first step of optimizing the cost of a truss is to solve for the member forces and cost given an input configuration. We used the Method of Joints approach learned in MTE 119: Statics to algorithmically formulate the problem as a matrix system of equations that could be inverted and solved. The input is a vector of joint (X,Y) positions, and the output is a vector of forces for each truss member.

$X = A^{-1}Y$
where
$X = \begin{bmatrix} a_1 & a_2 & a_3 \\ b_1 & b_2 & b_3 \\ c_1 & c_2 & c_3 \end{bmatrix}  $

The euclidian distance between joints was used to compute the cost, along with a multiplier associated with the modulus of the solved force with the maximum force to represent how many times the member must be reinforced.

According to the assignment specifications, truss members could be reinforced by doubling or tripling the material to multiply the force constraint at the cost of multiplying the cost. Our solver factored in this multiplier to the cost.

(Insert image)

Since the cost function was formulated symbolically using CasADI, the computer algebra system was able to automatically derive the jacobian and hessian matrices for the cost function to facilitate better nonlinear optimization. However, we noticed the reinforcement multiplier created discontinuities in the cost function. To solve that, we derived a smoothed multiplier function based on the sigmoid function with gradients along transition points between multipliers. This not only made the cost function continuous, but also had the benefit of making the optimizer prioritize jumps from a higher to lower multiplier to better minimize the cost. The final multiplier function we used is show in this desmos graph: https://www.desmos.com/calculator/40bxe1edsw.      

![image](https://github.com/OLeather/truss-optimization/assets/43189206/3eaa443c-2837-451f-9ffd-7f25a61ee812)


## Results
