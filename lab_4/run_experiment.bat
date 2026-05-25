@echo off
cd C:\Users\Varya\Desktop\lab_4

for %%s in (200 400 800 1200 1600 2000) do (
    echo ===== Размер %%s =====
    python generate_matrices.py %%s
    build\Release\lab_4.exe
    echo.
)

echo ===== Верификация =====
python verify.py
pause