/** @file main.cpp
 * Just a simple hello world using libfmt
 */
// The previous block is needed in every file for which you want to generate documentation

#include <fmt/format.h>
#include "Helpers/matrix.cpp"

// This should be in the headers

/**
 * @brief A function that does nothing but generate documentation
 * @return The answer to life, the universe and everything
 */
int foo();

int main()
{
    // Test matrix operations
    Matrix m1 = {{1, 2}, 
                 {3, 4}};
    Matrix m2 = {{5, 6}, 
                 {7, 8}};
    Matrix m3 = {{-1, 4, -2}, 
                 {-4, 6, 1},
                 {-6, -6, -2}};

    fmt::print("Add:\n");
    printMatrix(addMatrix(m1, m2));

    fmt::print("Multiply:\n");
    printMatrix(multiplyMatrix(m1, m2));

    fmt::print("REF of m3:\n");
    EliminationResult m3_elim = forwardElimination(m3);
    printMatrix(m3_elim.matrix);

    fmt::print("RREF of m3:\n"); // Prints diagonal matrix with 1s and 0s duh idk what im doing
    printMatrix(backwardElimination(m3_elim.matrix).matrix);

    fmt::print("Inverse of m3:\n");
    printMatrix(inverse(m3));

    fmt::print("Determinant m3:\n");
    fmt::print("{}\n", determinant(m3));

    return 0;
}

// Implementation
int foo() { return 42; }
