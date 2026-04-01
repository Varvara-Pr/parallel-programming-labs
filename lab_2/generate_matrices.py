import random
import os
import argparse


def generate_matrices(size, output_dir='data'):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(52)
    
    A = [[random.randint(-100, 100) for _ in range(size)] for _ in range(size)]
    B = [[random.randint(-100, 100) for _ in range(size)] for _ in range(size)]
    
    with open(f'{output_dir}/matrix_a.txt', 'w') as f:
        f.write(str(size) + '\n')
        for row in A:
            f.write(' '.join([str(x) for x in row]) + '\n')
    
    with open(f'{output_dir}/matrix_b.txt', 'w') as f:
        f.write(str(size) + '\n')
        for row in B:
            f.write(' '.join([str(x) for x in row]) + '\n')
    
    print(f'{size}x{size} матрицы созданы')


def main():
    parser = argparse.ArgumentParser(description='Генерация матриц')
    parser.add_argument('size', type=int, help='Размер матрицы')
    
    args = parser.parse_args()
    generate_matrices(args.size)

if __name__ == '__main__':
    main()