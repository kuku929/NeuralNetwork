# Simple Neural Network with GPU support
bare bones neural network library with CPU and GPU support.  
Only Linux is supported for now.

# Install

for both cpu and gpu, the following instructions apply   
assuming you are building for gpu: 
1. make a build directory inside gpu/
  ```bash
  mkdir gpu/build
  cd gpu/build
  ```
2. You can use whatever build system you want, for ninja:
```bash
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Debug
ninja
```
3. a main executable should now be in your build/ directory
