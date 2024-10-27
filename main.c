#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#define IMAGE_WIDTH 128
#define IMAGE_HEIGHT 128
#define TRAIN_COUNT 100

float cat_train[TRAIN_COUNT][IMAGE_WIDTH * IMAGE_HEIGHT * 3];
float dog_train[TRAIN_COUNT][IMAGE_WIDTH * IMAGE_HEIGHT * 3];

void load_image(const char *filename, float *image){
  int width, height, channels;

  float * data = stbi_loadf(filename, &width, &height, &channels, 3);

  stbir_resize_float_linear(data, width, height, 0, image, IMAGE_WIDTH, IMAGE_HEIGHT, 0, STBIR_RGB);

  stbi_image_free(data);
}

void load_dataset(const char *path, const char *label, int count, float train[][IMAGE_WIDTH*IMAGE_HEIGHT*3]){
  for (int i = 0; i < count; i++){
    char filename[128] = {0};
    sprintf(filename, "%s/%s.%d.jpg", path, label, i);
    load_image(filename, train[i]);
  } 
}

int main(){
  //float image[IMAGE_WIDTH * IMAGE_HEIGHT * 3] = {0};
  //load_image("./data/train/cat.0.jpg", image);

  load_dataset("./data/train", "cat", TRAIN_COUNT, cat_train);
  load_dataset("./data/train", "dog", TRAIN_COUNT, dog_train);

  for (int j = 0; j < TRAIN_COUNT; j++){
    unsigned char some_data[IMAGE_WIDTH * IMAGE_HEIGHT * 3];
    for (int i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT * 3; i++){
      some_data[i] = (unsigned char)(dog_train[j][i] * 255);
    }
    char filename[128] = {0};
    sprintf(filename,"examples/output%d.jpg", j);
    stbi_write_jpg(filename, IMAGE_WIDTH, IMAGE_HEIGHT, 3, some_data, 100);
  }
 }
