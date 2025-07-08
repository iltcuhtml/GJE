#ifndef MAT_H
#define MAT_H

#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#define ARRAY_LEN(xs) sizeof((xs)) / sizeof((xs)[0])

typedef struct
{
    size_t rows;
    size_t cols;

    size_t stride;

    float *es;
} Mat;

#define MAT_AT(m, row, col) (m).es[(row) * (m).stride + (col)]

Mat Mat_alloc(size_t rows, size_t cols);

void Mat_print(Mat m, const char *name, size_t padding);
#define MAT_PRINT(m) Mat_print(m, #m, 0)

void Mat_copy(Mat dst, Mat src);

void Mat_dot(Mat dst, Mat m1, Mat m2);

void Mat_free(Mat m);

Mat Mat_alloc(size_t rows, size_t cols)
{
    Mat m;

    m.rows = rows;
    m.cols = cols;

    m.stride = cols;

    m.es = (float*) malloc(sizeof(*m.es) * rows * cols);

    assert(m.es != NULL);

    return m;
}

void Mat_print(Mat m, const char *name, size_t padding)
{
    printf("\n%*s%s = [\n", (int) padding, "", name);

    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t ii = 0; ii < m.cols; ii++)
            printf("%*s    %f", (int) padding, "", MAT_AT(m, i, ii));

        printf("\n");
    }

    printf("%*s]\n", (int) padding, "");
}

void Mat_copy(Mat dst, Mat src)
{
    assert(dst.rows == src.rows);
    assert(dst.cols == src.cols);

    for (size_t i = 0; i < dst.rows; i++)
        for (size_t ii = 0; ii < dst.cols; ii++)
            MAT_AT(dst, i, ii) = MAT_AT(src, i, ii);
}

void Mat_dot(Mat dst, Mat m1, Mat m2)
{
    assert(m1.cols == m2.rows);
    assert(dst.rows == m1.rows);
    assert(dst.cols == m2.cols);

    for (size_t i = 0; i < dst.rows; i++)
        for (size_t ii = 0; ii < dst.cols; ii++)
        {
            MAT_AT(dst, i, ii) = 0;

            for (size_t iii = 0; iii < m1.cols; iii++)
                MAT_AT(dst, i, ii) += MAT_AT(m1, i, iii) * MAT_AT(m2, iii, ii);
        }
}

void Mat_free(Mat m)
{
    if (m.es != NULL)
        free(m.es);
}

#endif // MAT_H