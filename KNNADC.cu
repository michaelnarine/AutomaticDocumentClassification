#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
using namespace std;

__global__ void printArray(int *index, int outputsize) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < outputsize) {
		printf(" %d", index[i]);
	}
}

__global__ void printArray(double *index, int outputsize) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < outputsize) {
		printf(" %f", index[i]);
	}
}

__global__ void Count_number_of_term(int *A, int *Df) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int value = A[i] - 1;
	atomicAdd(&Df[value], 1);

}

__global__ void Kogge_Stone_scan_kernel(int *df, int *index, int InputSize, int thread_num, int *temp) {
	__shared__ int XY[100];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < InputSize) {
		XY[threadIdx.x] = df[i];
		__syncthreads();
	}

	for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
		__syncthreads();
		if (threadIdx.x >= stride) {
			XY[threadIdx.x] += XY[threadIdx.x - stride];
		}
	}
	if (i < thread_num-1) {
		index[i + 1] = XY[threadIdx.x];
	}
		__syncthreads();
}

__global__ void Create_InvertedIndexA (int *A, int *B, int *Df, int *Index ,int *InvertedIndexA) {
	int temp = Index[threadIdx.x] + Df[threadIdx.x];
	if (blockIdx.x == 0) {
		for (unsigned int i = Index[threadIdx.x]; i < temp; i++) {
			__syncthreads();
			InvertedIndexA[i] = threadIdx.x + 1;
		}

	}
}

__global__ void Create_InvertedIndexB(int *A, int *B, double *C, int *Df, int *Index, int *InvertedIndexB, double *InvertedIndexC) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	__syncthreads();
	int temp = 0;
	temp = Index[B[i]-1];
	__syncthreads();
	int value = B[i] - 1;
	int a = 0;
	a=atomicAdd(&Index[value], 1);
	InvertedIndexB[a] = A[i];
	InvertedIndexC[a] = C[i];
	__syncthreads();
}





// KNN Start

__global__ void knnTD(int *terms, double *qnorm, int *invertedIndex, double *norms, double *docs, int *index) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	double temp = 0;
	for (int j = index[terms[i]]; j < index[terms[i]+1]; j++) {
		docs[invertedIndex[j]]+=qnorm[i]*norms[j];
		__syncthreads();
	}
}

__global__ void knn(int* terms, double *qnorm, double *docs, double *docNorm, double queryNorm) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(docNorm[i] == 0 || queryNorm == 0) {
		docs[i] = 0;
	} else {
		docs[i] = docs[i]/(docNorm[i]*queryNorm);
	}
}

__global__ void printlist(int *terms, int *invertedIndex, int *index) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Term -> %d\n", index[terms[i]]);
	for (int j = index[terms[i]]; j < index[terms[i]+1]; j++) {
		printf("%d -> %d\n", terms[i], invertedIndex[j]);
	}
	printf("\n");
}

__global__ void getDocNorm(int *docs, int *terms, double *norms, double *dn, int num) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	double temp = 0.0;
	for (size_t j = 0; j < num; j++) {
		if(docs[j] == i) {
			temp+=(norms[j]*norms[j]);
		}
	}
	__syncthreads();
	dn[i] = sqrt(temp);
}

__global__ void oddEvenSort(double *data, int *dl, int num_elem) {
	int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
	int tid_idx;
	int offset = 0; //Start off with even, then odd
	int num_swaps;
	dl[tid] = tid+1;
	__syncthreads();
	//Calculation maximum index for a given block
	//Last block it is number of elements minus one
	//Other blocks to end of block minus one
	int tid_idx_max = min((((blockIdx.x + 1)*(blockDim.x * 2)) - 1), (num_elem - 1));

	do
	{
		//Reset number of swaps
		num_swaps = 0;

		//work out index of data
		tid_idx = (tid * 2) + offset;

		//If no array or block overrun
		if (tid_idx < tid_idx_max) {
			//Read values into registers
			double d0 = data[tid_idx];
			int db0 = dl[tid_idx];
			double d1 = data[tid_idx + 1];
			int db1 = dl[tid_idx + 1];

			//Compare registers
			if (d0 < d1) {
				//Swap values if needed
				data[tid_idx] = d1;
				dl[tid_idx] = db1;
				data[tid_idx + 1] = d0;
				dl[tid_idx + 1] = db0;

				//keep track that we did a swap
				num_swaps++;
			}
		}

		//Switch from even to off, or odd to even
		if (offset == 0) {
			offset = 1;
		}
		else {
			offset = 0;
		}
	} while (__syncthreads_count(num_swaps) != 0);
}


__global__ void classify(int *dl, double *dn, int k, double *dc, int *u) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	for (size_t j = 0; j < k; j++) {
		if(dl[j]/10 == u[i]) {
			dc[i] += dn[j];
		}
	}
}

// KNN end




int main(int argc, char **argv)
{
	int k;
	printf("%d\n", argc);
	if(argc < 2) {
		k = 3;
	} else {
		k = atoi(argv[1]);
	}

	// Read term-doc pairs
   ifstream ifs("result_norm.txt");
   string text;
   text.assign( (istreambuf_iterator<char>(ifs) ),
                     (istreambuf_iterator<char>()    ) );
   char arr[text.size()+1];
   strcpy(arr,text.c_str());

   vector<char*> v;
   vector<int> d1;
   vector<int> t1;
	vector<double> n1;
   char* chars_array = strtok(arr, "[");
   while(chars_array) {
         v.push_back(chars_array);
         chars_array = strtok(NULL, "[");
   }

	bool firstTerm = true, firstNorm = true;
   for(size_t n = 0; n < v.size(); ++n)
   {
        char* subchar_array = strtok(v[n], ",");
        while (subchar_array) {
            if (n == 0) {
               d1.push_back(atoi(subchar_array));
				} else if (n == 1) {
					if (firstTerm){
						d1.pop_back();
						firstTerm = false;
					}
               t1.push_back(atoi(subchar_array));
				} else if (n == 2) {
					if (firstNorm){
						t1.pop_back();
						firstNorm = false;
					}
					if (n1.size() == d1.size())
						break;
					n1.push_back(atof(subchar_array));
				}
            subchar_array = strtok(NULL, ",");
        }
   }

   int d[d1.size()];
   int t[t1.size()];
	double n[n1.size()];
   copy(d1.begin(), d1.end(), d);
   copy(t1.begin(), t1.end(), t);
	copy(n1.begin(), n1.end(), n);
	/*
	for (size_t i = 0; i < t1.size(); i++) {
      printf("%d -> [%d,%d,%f]\n",i,d[i],t[i],n[i]);
   }
	*/

	// Begin InvertedIndex algorithm

	int numDocs = d[d1.size()-1];

	const int arraySize = sizeof(d)/sizeof(int);
	printf("ArraySize: %d\n", arraySize);
	const int number_term = 7;
   int Df[number_term] = { 0 };
	int Index[number_term] = { 0 };
   vector<int> IA(arraySize,0);
   vector<int> IB(arraySize,0);
	vector<double> IC(arraySize,0);
	int InvertedIndexA[arraySize];//output
	int InvertedIndexB[arraySize];//output
	double InvertedIndexC[arraySize];//output
   copy(IA.begin(),IA.end(),InvertedIndexA);
   copy(IB.begin(),IB.end(),InvertedIndexB);
	copy(IC.begin(),IC.end(),InvertedIndexC);

	printf("A: %d\n", sizeof(InvertedIndexA)/sizeof(int));

	int thread_num = d[arraySize - 1];
	int blocks = (arraySize / thread_num) + (arraySize % thread_num != 0 ? 1 : 0);
	printf("blocks = %d\n", blocks);

	int *a, *b, *df, *index, *invertedIndexA, *invertedIndexB;
	double *c, *invertedIndexC, *dn;
	double docNorms[numDocs];
	cudaMallocManaged(&a, sizeof(d));
	cudaMallocManaged(&b, sizeof(t));
	cudaMallocManaged(&c, sizeof(n));
	cudaMallocManaged(&df, sizeof(Df));
	cudaMallocManaged(&index, sizeof(Index));
	cudaMallocManaged(&invertedIndexA, sizeof(InvertedIndexA));
	cudaMallocManaged(&invertedIndexB, sizeof(InvertedIndexB));
	cudaMallocManaged(&invertedIndexC, sizeof(InvertedIndexC));
	cudaMallocManaged(&dn,sizeof(docNorms));
	cudaMemcpy(a, d, sizeof(d), cudaMemcpyHostToDevice);
	cudaMemcpy(b, t, sizeof(t), cudaMemcpyHostToDevice);
	cudaMemcpy(c, n, sizeof(n), cudaMemcpyHostToDevice);
	cudaMemcpy(df, Df, sizeof(Df), cudaMemcpyHostToDevice);
	cudaMemcpy(index, Index, sizeof(Index), cudaMemcpyHostToDevice);
	cudaMemcpy(invertedIndexA, InvertedIndexA, sizeof(InvertedIndexA), cudaMemcpyHostToDevice);
	cudaMemcpy(invertedIndexB, InvertedIndexB, sizeof(InvertedIndexB), cudaMemcpyHostToDevice);
	cudaMemcpy(invertedIndexC, InvertedIndexC, sizeof(InvertedIndexC), cudaMemcpyHostToDevice);
	cudaMemcpy(dn,docNorms,sizeof(docNorms),cudaMemcpyHostToDevice);

	int Temp[number_term] = { 0 };int *temp;
	cudaMallocManaged(&temp, sizeof(Temp));
	cudaMemcpy(temp, Temp, sizeof(Temp), cudaMemcpyHostToDevice);

	printf("Initial Array:\n");
	printf("d:");
	printArray <<<1, arraySize>>> (a, sizeof(d) / sizeof(int));
	cudaDeviceSynchronize();
	printf("\n");

	printf("t:");
	printArray <<<1, arraySize >>> (b, sizeof(t) / sizeof(int));
	cudaDeviceSynchronize();
	printf("\n");

	printf("Count_number_of_term: \n");
	Count_number_of_term <<<blocks, thread_num>>> (b,df);
	cudaDeviceSynchronize();
	printArray <<<1, thread_num>>> (df, sizeof(Df) / sizeof(int));
	cudaDeviceSynchronize();
	printf("\n");

	printf("Execute the prefix sum by Kogge Stone:\n");
	Kogge_Stone_scan_kernel <<<blocks, thread_num>>> (df, index, arraySize, thread_num, temp);
	cudaDeviceSynchronize();

	//printf("Input count number array to the Kogge Stone:\n");
	printArray <<<1, arraySize >>> (index, sizeof(Index) / sizeof(int));
	cudaDeviceSynchronize();
	printf("\n");


	printf("InvertedIndex Array:\n");
	Create_InvertedIndexA <<<1, thread_num >>> (a, b, df, index, invertedIndexA);
	cudaDeviceSynchronize();
	printf("Terms: \n");
	for (size_t j = 0; j < arraySize; j++) {
		printf(" %d", invertedIndexA[j]);
	}
	printf("\n\n");

	printf("Documents: \n");
	Create_InvertedIndexB <<<blocks, thread_num >>> (a, b, c, df, index, invertedIndexB, invertedIndexC);
	cudaDeviceSynchronize();
	for (size_t j = 0; j < arraySize; j++) {
		printf(" %d", invertedIndexB[j]);
	}
	printf("\n\n");

	printf("Norms: \n");
	for (size_t j = 0; j < arraySize; j++) {
		printf(" %d", invertedIndexB[j]);
	}
	printf("\n\n");

	getDocNorm<<<1,numDocs>>>(a,b,c,dn,d1.size());
	cudaDeviceSynchronize();


   //Start Querying

   ifstream ifq("querydoc.txt");
   string qur;
   qur.assign( (istreambuf_iterator<char>(ifq) ),
                     (istreambuf_iterator<char>()    ) );
   char qarr[qur.size()+1];
   strcpy(qarr,qur.c_str());

   vector<char*> vq;
   vector<int> tq;
   vector<double> tf;
   char* query_array = strtok(qarr, "[");
   while(query_array) {
         vq.push_back(query_array);
         query_array = strtok(NULL, "[");
   }

   for(size_t n = 0; n < vq.size(); ++n)
   {
        char* subchar_array = strtok(vq[n], ",");
        while (subchar_array) {
            if (n == 0)
               tq.push_back(atoi(subchar_array));
            else if (n == 1)
               tf.push_back(atof(subchar_array));
            subchar_array = strtok(NULL, ",");
        }
   }

	int q_size = tq.size();
   int qterm[q_size];
	double sum[q_size];
   double qtermfreq[tf.size()];
   copy(tq.begin(), tq.end(), qterm);
   copy(tf.begin(), tf.end(), qtermfreq);


	int *qtptr;
	double *qfptr, *ds;
	double docSums[numDocs];


	cudaMallocManaged(&qtptr,q_size*sizeof(int));
	cudaMallocManaged(&qfptr,q_size*sizeof(double));
	cudaMallocManaged(&ds,sizeof(docSums));
	cudaMemcpy(qtptr,qterm,q_size *sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(qfptr,qtermfreq,q_size *sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(ds,docSums,sizeof(docSums),cudaMemcpyHostToDevice);

	double q_norm = 0;
	for (size_t j = 0; j < q_size; j++) {
		q_norm+=(qtermfreq[j]*qtermfreq[j]);
	}

	q_norm = sqrt(q_norm);

	knnTD<<<1,q_size>>>(qtptr,qfptr,invertedIndexB,invertedIndexC,ds,index);
	cudaDeviceSynchronize();
	knn<<<1,numDocs>>>(qtptr,qfptr,ds,dn,q_norm);
	cudaDeviceSynchronize();


/*
	printf("\n\nDoc Distances:\n");

	for (size_t j = 0; j < numDocs; j++) {
		printf("   %d -> %f\n",j+1,ds[j]);
	}
*/

	int docLabel[numDocs];
	int *dl;
	cudaMallocManaged(&dl,sizeof(docLabel));
	cudaMemcpy(dl,docLabel,sizeof(docLabel),cudaMemcpyHostToDevice);

	oddEvenSort<<<1,numDocs>>>(ds,dl,numDocs);
	cudaDeviceSynchronize();

	vector<int> nn;
	printf("\nK Nearest Neighbors (k=%d): \n", k);
	for (size_t j = 0; j < k; j++) {
		if (find(nn.begin(), nn.end(), dl[j]) != nn.end()) {

		} else {
			nn.push_back(dl[j]/10);
		}
		printf("   %d -> %f -> label = %d\n", dl[j],ds[j],dl[j]/10);
	}
	int uniqueN[nn.size()];
	copy(nn.begin(), nn.end(), uniqueN);


	double kCount[nn.size()];
	double *dc;
	int *u;
	cudaMallocManaged(&dc,sizeof(kCount));
	cudaMallocManaged(&u,sizeof(uniqueN));
	cudaMemcpy(dc,kCount,sizeof(kCount),cudaMemcpyHostToDevice);
	cudaMemcpy(u,uniqueN,sizeof(uniqueN),cudaMemcpyHostToDevice);
	classify<<<1,nn.size()>>>(dl,ds,k,dc,u);
	cudaDeviceSynchronize();

	double max = 0;
	int max_i = 0;
	for (size_t j = 0; j < nn.size(); j++) {
		if(dc[j] > max) {
			max = dc[j];
			max_i = j;
		}
	}
	printf("\nQuery Document is labelled = %d\n", u[max_i]);



	printf("\n");



	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
	cudaFree(df);
	cudaFree(index);
	cudaFree(invertedIndexA);
	cudaFree(invertedIndexB);
	cudaFree(invertedIndexC);
	cudaFree(qtptr);
	cudaFree(qfptr);
	cudaFree(ds);
	cudaFree(dn);
	cudaFree(dl);
	cudaFree(dc);

	return 0;
}
