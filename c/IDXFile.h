#ifndef IDXFILE_H
#define IDXFILE_H

#include <stddef.h>
#include <stdint.h>

/*
 * https://www.fon.hum.uva.nl/praat/manual/IDX_file_format.html
 ***/
typedef struct {
    uint8_t dtype;
    uint8_t rank;
    uint32_t* shape;
    uint8_t* data;
} IDXFile;


IDXFile IDXFile_ctor(const char* filepath);
void IDXFile_dtor(IDXFile idxfile);

void IDXFile_print(IDXFile idxfile);

#endif
