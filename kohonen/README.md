# Self Organizing Maps - Kohonen

Self Organizing Maps are implemented.
The code requires PIL for better visualization of network but if there is a problem delete PIL and Image related lines and directly draw h_grid.  
The animation code can be used to create animations and needs ImageMagick, without it wont work !!!.
In the gif the data is in three dimensional space. The neuron weights are marked with triangles and data points are circles.  
In the visualization, only the last activation (Winner neuron) in the set is show for each epoch.  
Kohonen is unsupervised learning method. The training done without class knowledge (update part). Then the neuron classes are added.

![Kohonen](kohonen.gif)

## To-Do List

* Make codes more readeble and fix issues  
* Documentation needs to be added
* Torch implementation  
