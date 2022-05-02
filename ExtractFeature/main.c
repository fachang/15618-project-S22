#include <stdlib.h>
#include <stdio.h>
// #include <iostream>

unsigned char input[4*3*32*32];
unsigned char output[4*10];

unsigned char conv1_w[4*128*3*3*3];
unsigned char conv1_b[4*128];
unsigned char conv2_w[4*128*128*3*3];
unsigned char conv2_b[4*128];
unsigned char conv3_w[4*256*128*3*3];
unsigned char conv3_b[4*256];
unsigned char conv4_w[4*256*256*3*3];
unsigned char conv4_b[4*256];
unsigned char conv5_w[4*512*256*3*3];
unsigned char conv5_b[4*512];
unsigned char conv6_w[4*512*512*3*3];
unsigned char conv6_b[4*512];


unsigned char linear1_w[4*4096*8192];
unsigned char linear1_b[4*4096];
unsigned char linear2_w[4*4096*4096];
unsigned char linear2_b[4*4096];
unsigned char linear3_w[4*10*4096];
unsigned char linear3_b[4*10];
// using namspace std;

void read_binary(const char* filename, void* buffer, int length) {
	FILE *ptr;
	// printf("%d\n", length);
	ptr = fopen(filename,"rb");
	if(!ptr) {
		printf("file not found\n");
		return;
	}
	fread(buffer,length,1,ptr);
}

void load_input_output() {
	read_binary("input.bin", (void*)input, sizeof(input));
	float* fbuffer = (float*) input;
	printf("%lf\t", fbuffer[0]);
	printf("%lf\t", fbuffer[1*32*32]);
	printf("%lf\n", fbuffer[2*32*32]);
	// for(int i = 0; i < sizeof(input)/4; i++) {
	// 	printf("%lf\t", fbuffer[i]);
	// 	if(i != 0 && i % 32 == 0) {
	// 		printf("\n");
	// 	}
	// }
	read_binary("output.bin", (void*)output, sizeof(output));
	fbuffer = (float*) output;
	for(int i = 0; i < sizeof(output)/4; i++) {
		printf("%lf\t", fbuffer[i]);
	}
}

void load_wand() {
	float* fbuffer;
	read_binary("./model/conv1_w_np.bin", (void*)conv1_w, sizeof(conv1_w));

	read_binary("./model/conv1_b_np.bin", (void*)conv1_b, sizeof(conv1_b));
	read_binary("./model/conv2_w_np.bin", (void*)conv2_w, sizeof(conv2_w));
	read_binary("./model/conv2_b_np.bin", (void*)conv2_b, sizeof(conv2_b));
	read_binary("./model/conv3_w_np.bin", (void*)conv3_w, sizeof(conv3_w));
	read_binary("./model/conv3_b_np.bin", (void*)conv3_b, sizeof(conv3_b));
	read_binary("./model/conv4_w_np.bin", (void*)conv4_w, sizeof(conv4_w));
	read_binary("./model/conv4_b_np.bin", (void*)conv4_b, sizeof(conv4_b));
	read_binary("./model/conv5_w_np.bin", (void*)conv5_w, sizeof(conv5_w));
	read_binary("./model/conv5_b_np.bin", (void*)conv5_b, sizeof(conv5_b));
	read_binary("./model/conv6_w_np.bin", (void*)conv6_w, sizeof(conv6_w));
	read_binary("./model/conv6_b_np.bin", (void*)conv6_b, sizeof(conv6_b));

	read_binary("./model/linear1_w_np.bin", (void*)linear1_w, sizeof(linear1_w));
	read_binary("./model/linear1_b_np.bin", (void*)linear1_b, sizeof(linear1_b));
	read_binary("./model/linear2_w_np.bin", (void*)linear2_w, sizeof(linear2_w));
	read_binary("./model/linear2_b_np.bin", (void*)linear2_b, sizeof(linear2_b));
	read_binary("./model/linear3_w_np.bin", (void*)linear3_w, sizeof(linear3_w));
	read_binary("./model/linear3_b_np.bin", (void*)linear3_b, sizeof(linear3_b));
	
	fbuffer = (float*) conv1_w;
	printf("\nconv1_w\n");
	for(int i = 0; i < sizeof(conv1_w)/4; i++) {
		printf("%lf\t", fbuffer[i]);
	}
	printf("\nconv1_b\n");
	fbuffer = (float*) conv1_b;
	for(int i = 0; i < sizeof(conv1_b)/4; i++) {
		printf("%lf\t", fbuffer[i]);
	}
	printf("\nlinear3_w\n");
	fbuffer = (float*) linear3_w;
	for(int i = 0; i < sizeof(linear3_w)/4; i++) {
		printf("%lf\t", fbuffer[i]);
	}
	printf("\nlinear3_b\n");
	fbuffer = (float*) linear3_b;
	for(int i = 0; i < sizeof(linear3_b)/4; i++) {
		printf("%lf\t", fbuffer[i]);
	}
}
int main() {

	load_wand();
}
