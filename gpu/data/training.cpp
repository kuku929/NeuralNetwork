#include <fstream>
#include <ostream>
#include <vector>

int main(){
	std::ofstream fout("data.txt");
	std::vector<std::vector<float>> v = {{1.0f,1.0f}, {1.0f,0.0f}, {0.0f,1.0f}, {0.0f,0.0f}};
	//std::vector<std::vector<float>> v = {{1.0f,1.0f}, {1.0f,0.0f}, {0.0f, 1.0f}};
	std::vector<float> a = {0.0f, 1.0f, 1.0f, 0.0f};
	for(int i=0; i < 1024; ++i){
		int index = rand()%4;
		fout << v[index][0] << ' ' << v[index][1] << ' ' << a[index] << '\n'; 	
	}
	fout.close();
}
