#pragma once
typedef float (*activation_funcptr)(float);
__device__ float sigmoid(float weighted_sum){
	float e_x=exp(weighted_sum);
	return e_x/(1.0f+e_x);
}

__device__ float ReLU(float weighted_sum){
	if(weighted_sum>0.0f)return weighted_sum;
	return 0.0f;
}

__device__ float Leaky(float weighted_sum){
	if(weighted_sum<0.0f)return weighted_sum*0.01f;
	return weighted_sum;
}

__device__ float Linear(float weighted_sum){
	return weighted_sum;
}

__device__ float sigmoid_der(float output){
	return output*(1-output);
}

__device__ float ReLU_der(float output){
	if(output>0.0f)return 1.0f;
	return 0.0f;
}

__device__ float Leaky_der(float output){
	if(output>0.0f)return 1.0f;
	return 0.01f;
}

__device__ float Linear_der(float weighted_sum){
	return 1.0f;
}
__device__ activation_funcptr activation_func_map[4] = {sigmoid, ReLU, Leaky, Linear};
__device__ activation_funcptr activation_func_der_map[4] = {sigmoid_der, ReLU_der, Leaky_der, Linear_der};
__device__ activation_funcptr null_funcptr = Linear;

