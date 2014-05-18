#include <iostream>
#include "ANN.h"

using namespace std;

int main()
{
    cout << "Hello world!" << endl;
    ANN net (4);
    net.display_connections();
    return 0;
}
