__kernel void bitonic_sort(__global long* data, const int size, const int stage, const int step) {
    int i = get_global_id(0);
    int ixj = i ^ step;

    if (ixj > i) {
        if ((i & stage) == 0) {
            if (data[i] > data[ixj]) {
                long temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        } else {
            if (data[i] < data[ixj]) {
                long temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        }
    }
}
