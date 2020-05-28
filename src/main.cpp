#include "test.h"

#include <stdlib.h>
#include <cstring>

int main(int argc, char* argv[]){
    int imageNum = 0;
    if (argc > 1) {
        for (int i = 1; i < argc; i++) {
            if(strcmp(argv[i], "-n") == 0)
            {
                imageNum = (int)atoi(argv[++i]);
            }
        }
    }
    test(imageNum);
    return 0;
}

