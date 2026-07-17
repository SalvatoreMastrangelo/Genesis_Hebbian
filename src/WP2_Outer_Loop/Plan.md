you have complete access to the computing power of this pc to run your code and deliver a fully working wp2 outer loop implementation. you can perform tests and understand from logs the situation at any time. consider that the computing power is limited, so try to send simulations with increasing time requirements carefully. you can run stuff using PYTHONPATH=src python

take inspiration from the current implementation of the WP2 outer loop and the random mutation resampling currently performed in the inner loop, so that you can decide whether it makes sense to create a more efficient and effective outer loop implementation taking inspiration from the inner loop or fix/edit the current one. Consider a suggested set of fixes in the outer loop folder but do not limit yourself to them. 

i want to have in the end the demostration of improving pareto fronts, using as objectives only the fitness and the cost of transport, using a plot, as well as a fitness and other metrics plot just like the current inner loop plots. 

if possible, for the tests you will run, try to stay at or below 4 urdfs. i know it's a very tight constraint, but it is important to keep the tests efficient, since each urdf refresh requires a lot of time.

if you have any questions ask now, then infer all the rest by yourself. block and ask again only if it is ABSOLUTELY necessary.