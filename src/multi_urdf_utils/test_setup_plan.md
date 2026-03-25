I want to make a test setup on multi-urdf scenes. let's define some namings before the project description.

NAMINGS
 [] drone: can be used interchangeably with urdf. there must be no difference and thus no confusion.
 [] WP: work package, the different stages of the project.
 [] environment: the single simulation instance. in the previous WP is the number under the name of envs, and was used for a single urdf simulation
 [] scene: set of multiple environments. in the previous WP a scene was the set of environments in which a single urdf was simulated. in the src/WP1/configs/foundation.yaml there are for example 96 urdfs and 33600 environments, thus each scene contains 33600/96 = 350 environments. 
 [] morphology: the specific urdf of a drone, thus the specific drone.
 
TEST SETUP
 I want to make a test setup in which each scene (grandtotal of S scenes) can be populated with N different drones (so a grandtotal of N*S drones), thus each of E environments contained in a specific scene will be populated with the N different drones related to that specific scene.  

REASONS AND MOTIVATIONS
 Currently at each generation, the WP2 compiles a bunch of genesis scenes each with only one drone and then simulates them in parallel on one GPU. this specific setup suffers from great simulation overhead due to recompilation at each generation in case of morphology evolution, sometimes overwhelming the simulation time. the proposed test setup will allow to simulate multiple drones in the same scene, with the aim to find a tradeoff between the number of drones (N) and the number of scenes (S), since lowering the number of scenes will lower the simulation overhead but also the GPU overall utilization due to lower parallelization. 

OUTCOME AND METHODS
 The outcome of this test setup must be a new python script (or a set of scripts if necessary) named WP2.5 that will hook to basecode, WP1 and WP2 in an incremental way, possibly without modifying the existing codebase, but without sacrificing nor modularity nor performance, thus rewriting entire functions if needed. this new code must of course either disable collisions at all in the simulator (since drone-cylinder collisons are computed geometrically and not by the physics engine) to reduce the collision checking overhead, or at least disable collisions between drones so that no drone is influenced by the presence of other drones. 

REPLICABILITY
 Just like the previous WPs, every simulation must be totally replicable by only having the codebase and the configuration yaml. The configuration yaml must be designed in a way to allow to easily change the number of drones (N), the number of scenes (S) and the number of environments (E) per scene, apart from obviously all the other simulation parameters included in the WP2 configuration yaml. 


If you have any questions on any of the above points feel free to ask. refer to this file for any clarification on the above points, apart from of course questions to myself.

Now i want to enter into plan mode in order to define the steps to implement the above test setup or any other relevant detail or doubts that may arise. 

Remember, make no errors.
