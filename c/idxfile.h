#ifndef IDXFILE_H
#define IDXFILE_H

#include <stddef.h>
#include <stdint.h>

/*
 * https://www.fon.hum.uva.nl/praat/manual/IDX_file_format.html
 ***/
typedef struct {
    /*
    * 0x08: unsigned byte
    * 0x09: signed byte
    * 0x0B: short (2 bytes)
    * 0x0C: int (4 bytes)
    * 0x0D: float (4 bytes)
    * 0x0E: double (8 bytes)
    */
    uint8_t dtype;
    // number of dimensions of the vector/matrix: 1 for vectors, 2 for matrices....
    uint8_t rank;
    // the size of each dimension; 4-byte integers (big-endian, like in most non-Intel processors).
    uint32_t* shape;
    // The data is stored like in a C array, i.e. the index in the last dimension changes the fastest.
    // Both labels and images are 0x08 -> unsigned byte type
    uint8_t* data;
} IDXFile;


IDXFile IDXFile_ctor(const char* filepath);
void IDXFile_dtor(IDXFile idxfile);

void IDXFile_print(IDXFile idxfile);

#endif
