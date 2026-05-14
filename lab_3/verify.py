import numpy as np


def read_matrix_from_file(filename):
    """Чтение матрицы из файла"""

    with open(filename, 'r') as f:
        lines = f.readlines()
    
    size = int(lines[0].strip())
    
    matrix = []
    for i in range(1, size + 1):
        row = list(map(int, lines[i].strip().split()))
        matrix.append(row)
    
    return np.array(matrix), size


def print_matrix_preview(name, matrix, size):
    """Вывод первых строк матрицы"""

    print(f"\n{name} (первые {min(5, size)}x{min(5, size)} элементов):")
    preview_size = min(5, size)
    for i in range(preview_size):
        row_str = ' '.join(f"{matrix[i][j]:6d}" for j in range(preview_size))
        print(row_str)
    if size > 5:
        print("...")


def verify_multiplication():
    """Верификация результатов вычислений"""

    A, sizeA = read_matrix_from_file('data/matrix_a.txt')
    B, sizeB = read_matrix_from_file('data/matrix_b.txt')
    
    C_result, sizeC = read_matrix_from_file('data/result.txt')

    print_matrix_preview("result.txt", C_result, sizeC)
    
    C_correct = np.dot(A, B)
    print_matrix_preview("Результат (NumPy)", C_correct, sizeA)
    
    print(f"\nРазмер матриц: {sizeA}x{sizeA}")
    if np.array_equal(C_result, C_correct):
        print("Результаты совпадают")
        return True
    else:
        print("Результаты не совпадают")
        return False


def main():
    verify_multiplication()

if __name__ == "__main__":
    main()
