#include <iostream>
#include <numeric>
#include <math.h>
#include <vector>

using namespace std;

void print2d(vector<vector<int> > &vec){
	for (int i = 0; i < vec.size(); i++)
	{
		for (int j = 0; j < vec[i].size(); j++)
		{
			cout<<vec[i][j]<<'\t';
		}
		 cout<<endl;
	}
}

vector<vector<int> > get_offsets(const int dim){
	const int N = pow(3.,dim);
	vector< vector<int> > offsets(N,vector<int>(dim,0));
	for (int i = 1; i < N; i++)
	{
		bool flag = true;
		for (int d = dim-1; d >= 0; d--) {
			if (flag) {
				int temp = (offsets[i-1][d]+1) + 1;
				flag = temp >= 3;
				offsets[i][d] = temp % 3 - 1;
			}
			else {
				offsets[i][d] = offsets[i-1][d];
			}
		}
	}
	return offsets;
}

vector<vector<int> > get_neighbours(vector<int> p, vector<int> shape, bool exclude_p = true){
	const int dim = p.size();
	vector<vector<int> > offsets = get_offsets(dim);
	vector<vector<int> > neighbours;
	if (exclude_p) offsets.erase(offsets.begin());
	
	for (int i = 0; i < offsets.size(); i++)
	{
		vector<int> temp(dim, 0);
		bool flag = true;
		for (int j = 0; j < dim; j++) {
			temp[j] = offsets[i][j] + p[j];
			if (temp[j] < 0 || temp[j] >= shape[j]) {
				flag = false;
			}
		}
		if (flag) {
			neighbours.push_back(temp);
		}
	}
	return neighbours;
}

int main(int argc, char *argv[]) {
	int a[] = {3, 3, 4};
	vector<int> p;
	p.assign(a, a+3);
	int b[] = {4, 4, 5};	
	vector<int> shape;
	shape.assign(b, b+3);
	vector<vector<int> > v = get_neighbours(p, shape, true);
	//print2d(v);
	v = get_offsets(3);
	print2d(v);
}