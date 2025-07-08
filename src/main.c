#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>

#include "Mat.h"

void gauss_jordan(Mat A, Mat b)
{
    size_t n = A.cols;
    Mat aug = Mat_alloc(n, A.cols + 1);

    // Create augmented matrix [A | b]
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < A.cols; j++)
            MAT_AT(aug, i, j) = MAT_AT(A, i, j);

        MAT_AT(aug, i, A.cols) = MAT_AT(b, i, 0);
    }

    for (size_t col = 0; col < n; col++)
    {
        size_t pivot_row = col;
        float max_val = fabsf(MAT_AT(aug, pivot_row, col));

        for (size_t row = col + 1; row < n; row++)
        {
            float val = fabsf(MAT_AT(aug, row, col));
            
            if (val > max_val)
            {
                max_val = val;
                pivot_row = row;
            }
        }

        if (fabsf(max_val) <= FLT_EPSILON)
        {
            fprintf(stderr, "Error: Singular matrix detected. Cannot proceed with Gauss-Jordan elimination (pivot too small or zero at column %zu).", col);
            Mat_free(A);
            Mat_free(b);            
            Mat_free(aug);
            
            exit(EXIT_FAILURE);
        }

        if (pivot_row != col)
        {
            for (size_t j = 0; j < aug.cols; j++)
            {
                float tmp = MAT_AT(aug, col, j);
                MAT_AT(aug, col, j) = MAT_AT(aug, pivot_row, j);
                MAT_AT(aug, pivot_row, j) = tmp;
            }
        }

        float pivot = MAT_AT(aug, col, col);

        for (size_t j = 0; j < aug.cols; j++)
            MAT_AT(aug, col, j) /= pivot;

        for (size_t row = 0; row < n; row++)
        {
            if (row == col) continue;

            float factor = MAT_AT(aug, row, col);
            for (size_t j = 0; j < aug.cols; j++)
                MAT_AT(aug, row, j) -= factor * MAT_AT(aug, col, j);
        }
    }

    for (size_t i = 0; i < n; i++)
        MAT_AT(b, i, 0) = MAT_AT(aug, i, n);

    Mat_free(aug);
}

int main(void)
{
    system("cls");

    int choice;

    while (1)
    {
        printf("Choose input method:\n");
        printf("1) Manual input\n");
        printf("2) Load from file\n");
        printf("3) Exit\n");
        printf("> ");
        scanf("%d", &choice);

        while (getchar() != '\n');

        if (choice != 1 && choice != 2 && choice != 3)
        {
            system("cls");
            fprintf(stderr, "Error: Invalid choice.\n\n");
        }
        else
            break;
    }    
    
    system("cls");

    Mat A, b;

    if (choice == 1)
    {
        char line[256];
        size_t rows, cols;
        
        while (1)
        {
            printf("Enter number of rows and columns (space-separated):\n> ");

            if (!fgets(line, sizeof(line), stdin))
            {
                system("cls");
                fprintf(stderr, "Error: Invalid input.\n\n");

                continue;
            }

            if (sscanf(line, "%zu %zu", &rows, &cols) == 2)
                break;

            system("cls");
            printf("Error: Invalid input. Please enter two integers.\n\n");
        }

        A = Mat_alloc(rows, cols);
        b = Mat_alloc(rows, 1);

        if (A.es == NULL || b.es == NULL)
        {
            system("cls");
            fprintf(stderr, "Error: Failed to allocate memory for matrix A or vector b.");
            
            Mat_free(A);
            Mat_free(b);

            return EXIT_FAILURE;
        }

        system("cls");

        for (size_t i = 0; i < rows; i++)
        {
            while (1)
            {
                printf("Enter elements of row %zu (space-separated, %zu values):\n> ", i + 1, cols);
                
                if (!fgets(line, sizeof(line), stdin))
                {
                    system("cls");
                    fprintf(stderr, "Error: Invalid input.\n\n");
                    
                    continue;
                }
            
                // Try to parse expected number of floats
                size_t count = 0;
                char *ptr = line;

                for (size_t j = 0; j < cols; j++)
                {
                    float val;
                    int n;

                    if (sscanf(ptr, "%f%n", &val, &n) != 1) break;
                    
                    MAT_AT(A, i, j) = val;
                    
                    ptr += n;
                    count++;
                }
            
                if (count == cols) break;

                system("cls");
                printf("Error: Invalid input. Please enter %zu float values\n\n", cols);
            }
        }

        system("cls");

        for (size_t i = 0; i < rows; i++)
        {
            while (1)
            {
                printf("Enter right-hand side value for row %zu:\n> ", i + 1);

                if (!fgets(line, sizeof(line), stdin))
                {
                    system("cls");
                    fprintf(stderr, "Error: Invalid input.\n\n");

                    continue;
                }

                float val;

                if (sscanf(line, "%f", &val) == 1)
                {
                    MAT_AT(b, i, 0) = val;
                
                    break;
                }

                system("cls");
                printf("Error: Invalid input. Please enter a float value.\n\n");
            }
        }
    }
    else if (choice == 2)
    {
        char path[256];
        FILE *file;

        while (1)
        {
            printf("Enter file path:\n> ");
            fgets(path, sizeof(path), stdin);

            path[strcspn(path, "\n")] = '\0';

            file = fopen(path, "r");
            
            system("cls");
            
            if (!file)
                fprintf(stderr, "Error: Cannot open file '%s'\n\n", path);
            else
                break;
        }

        size_t rows, cols;

        if (fscanf(file, "%zu %zu", &rows, &cols) != 2)
        {
            fprintf(stderr, "Error: Failed to read matrix dimensions from file.");
            fclose(file);

            return EXIT_FAILURE;
        }

        A = Mat_alloc(rows, cols);
        b = Mat_alloc(rows, 1);
        
        if (A.es == NULL || b.es == NULL)
        {
            fprintf(stderr, "Error: Failed to allocate memory for matrix A or vector b.");
            
            Mat_free(A);
            Mat_free(b);

            return EXIT_FAILURE;
        }

        int read_success = 1;

        for (size_t i = 0; i < rows; i++)
        {
            for (size_t j = 0; j < cols; j++)
            {
                if (fscanf(file, "%f", &MAT_AT(A, i, j)) != 1)
                {
                    fprintf(stderr, "Error: Not enough matrix elements in file (A[%zu][%zu]).\n", i, j);
                    read_success = 0;
                    
                    break;
                }
            }

            if (!read_success) break;
        
            if (fscanf(file, "%f", &MAT_AT(b, i, 0)) != 1)
            {
                fprintf(stderr, "Error: Missing right-hand side value for row %zu.\n", i);
                read_success = 0;

                break;
            }
        }

        if (!read_success)
        {
            fclose(file);

            Mat_free(A);
            Mat_free(b);

            return EXIT_FAILURE;
        }
    }
    if (choice == 3)
        return EXIT_SUCCESS;

    // Check for underdetermined or overdetermined system
    if (A.rows < A.cols)
    {
        fprintf(stderr, "Error: Too few equations. The system is underdetermined.");

        Mat_free(A);
        Mat_free(b);

        return EXIT_FAILURE;
    }

    if (A.rows > A.cols)
    {
        printf("Warning: More equations than unknowns. Proceeding anyway.\n\n");
    }

    // Move rows starting with 0 to bottom
    for (size_t i = 0; i < A.rows - 1; i++)
    {
        if (fabsf(MAT_AT(A, i, 0)) < FLT_EPSILON)
        {
            for (size_t k = i + 1; k < A.rows; k++)
            {
                if (fabsf(MAT_AT(A, k, 0)) > FLT_EPSILON)
                {
                    // Swap rows in A
                    for (size_t j = 0; j < A.cols; j++)
                    {
                        float tmp = MAT_AT(A, i, j);
                        MAT_AT(A, i, j) = MAT_AT(A, k, j);
                        MAT_AT(A, k, j) = tmp;
                    }

                    // Swap in b
                    float tmp_b = MAT_AT(b, i, 0);
                    MAT_AT(b, i, 0) = MAT_AT(b, k, 0);
                    MAT_AT(b, k, 0) = tmp_b;
                    
                    break;
                }
            }
        }
    }

    // Save original A and b for later verification
    Mat A_og = Mat_alloc(A.rows, A.cols);
    Mat b_og = Mat_alloc(b.rows, b.cols);
    
    if (A_og.es == NULL || b_og.es == NULL)
    {
        fprintf(stderr, "Error: Failed to allocate memory for matrix copies.");

        Mat_free(A);
        Mat_free(b);
        Mat_free(A_og);
        Mat_free(b_og);

        return EXIT_FAILURE;
    }
    
    Mat_copy(A_og, A);
    Mat_copy(b_og, b);

    printf("Solving system using Gauss-Jordan elimination\n\n");
    
    Mat solution = Mat_alloc(A.cols, b.cols);
    
    if (solution.es == NULL)
    {
        fprintf(stderr, "Error: Failed to allocate memory for matrix copies.");
        
        Mat_free(A);
        Mat_free(b);
        Mat_free(A_og);
        Mat_free(b_og);
        Mat_free(solution);
        
        return EXIT_FAILURE;
    }
    
    gauss_jordan(A, b);

    for (size_t i = 0; i < solution.rows; i++)
        for (size_t ii = 0; ii < solution.cols; ii++)
            MAT_AT(solution, i, ii) = MAT_AT(b, i, ii);

    printf("Computed solution vector:");
    MAT_PRINT(solution);

    // Verify solution
    Mat check = Mat_alloc(A_og.rows, 1);
    
    if (check.es == NULL) {
        fprintf(stderr, "\nError: Failed to allocate memory for verification vector.");
        
        Mat_free(A);
        Mat_free(b);
        Mat_free(A_og);
        Mat_free(b_og);
        Mat_free(solution);
        
        return EXIT_FAILURE;
    }
    
    Mat_dot(check, A_og, solution);
    
    printf("\nVerifying solution");

    int valid = 1;
    for (size_t i = 0; i < check.rows; i++)
    {
        if (fabsf(MAT_AT(check, i, 0) - MAT_AT(b_og, i, 0)) > 1e-3f)
        {
            printf("\nWarning: Equation %zu does not match the solution.", i + 1);

            valid = 0;
        }
    }

    if (valid)
        printf("\nThe solution is valid.");
    else
        printf("\nThe solution is invalid or inconsistent.");

    // Free memory
    Mat_free(A);
    Mat_free(b);
    Mat_free(A_og);
    Mat_free(b_og);
    Mat_free(solution);
    Mat_free(check);

    return EXIT_SUCCESS;
}