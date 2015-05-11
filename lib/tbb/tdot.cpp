template<typename T>
void tensdot(T alpha, T *A, int na, int *sa, int *pa,
                      T *B, int nb, int *sb, int *pb,
                      T *C, int n) {
    if(na > 10 || nb > 10 || n > na+nb || na < 1 || nb < 1 || n < 0) {
        printf("whoa there!\n");
        return;
    }

    T acc;
