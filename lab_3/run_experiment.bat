@echo off

set sizes=200 400 800 1200 1600 2000
set processes=1 2 4 8

for %%s in (%sizes%) do (
    echo.
    echo ===== Генерация матриц %%s x %%s =====
    python generate_matrices.py %%s
    
    for %%p in (%processes%) do (
        echo.
        echo ----- Размер: %%s, Процессов: %%p -----
        mpiexec -np %%p lab_3.exe
        echo.
    )
)

echo.
pause