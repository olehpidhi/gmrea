#include <iostream>
#include "/usr/local/include/armadillo"
#include <thread>
#include <array>
using namespace std;
using namespace arma;
mat parallel_multiply(mat A, mat B)
{
    return mat(1,1);
}

vec DoParallel1(mat A, vec b, vec x0, double eps = 0.0001)
{
    int m = b.n_elem / 2 + 1;
    int p = m;
    //TODO: implement parallel
    vec r0 = b - A*x0;
    double beta = norm(r0);
    std::vector<vec> v(m+1, 0);
    mat H(m + 1, m);
    mat G(2, m);
    vec g(m + 1);
    do
    {
        H.zeros();
        g = vec(m + 1);
        g(m) = 1;
        g *= beta;
        v[0] = r0 / beta;
        vec w;
        for (int i = 0; i < m; ++i)
        {
            w = A * v[i];
            for (int k = 0; k < i + 1; ++k)
            {
                H(k, i) = dot(w, v[k]);
                w = w - H(k, i) * v[k];
            }
            H(i + 1, i) = norm(w);
            v[i + 1] = w / H(i + 1, i);
            for (int k = 0; k < i; ++k)
            {
                double hki = G(0, k) * H(k, i) + G(1, k) * H(k + 1, i);
                H(k + 1, i) = -G(1, k) * H(k, i) + G(0, k) * H(k + 1, i);
                H(k, i) = hki;
            }
            double hk = sqrt(pow(H(i, i), 2) + pow(H(i + 1, i), 2));
            G(0, i) = H(i, i) / hk;
            G(1, i) = H(i + 1, i) / hk;


            if (i != m - 1)
            {
                double hii = G(0, i) * H(i, i) + G(1, i) * H(i + 1, i);
                H(i + 1, i) = -G(1, i) * H(i, i) + G(0, i) * H(i + 1, i);
                H(i, i) = hii;
            }

            //g=Gi*g
            double gi = G(0, i) * g(i) + G(1, i) * g(i + 1);
            g(i + 1) = -G(1, i) * g(i) + G(0, i) * g(i + 1);
            g(i) = gi;
            if (abs(g(i + 1)) < eps)
            {
                p = (int)i + 1;
                break;
            }
        }
        //need to solve treangular SLAU
        vec y =  solve(H, g, "std");
        for (int i = 0; i < p; ++i)
        {
            x0 = x0 + y(i) * v[i];
        }
        if (p < m)
        {
            break;
        }
        vec tempvec = A * x0;
        r0 = b - tempvec;
        beta = norm(r0);
    }
    while (beta > eps);
    return x0;
}

int main()
{
    mat randomMatrix(5,5);
    randomMatrix.randu();
    vec randomVector(5);
    randomVector.randu();
    vec randomSolution(5);
    randomSolution.randu();
    vec solution = DoParallel1(randomMatrix,randomVector, randomSolution);
    cout << solution;
    return 0;
}

