# 血细胞检测系统 - PyInstaller 打包脚本
# 用法: .\build_exe.ps1

$ErrorActionPreference = "Stop"

Write-Host "=== 血细胞检测系统 EXE 打包 ===" -ForegroundColor Cyan

# 检查 PyInstaller
$pi = Get-Command pyinstaller -ErrorAction SilentlyContinue
if (-not $pi) {
    Write-Host "PyInstaller 未安装，正在安装..." -ForegroundColor Yellow
    pip install pyinstaller
}

# 清理旧构建
$dirs = @("build", "dist")
foreach ($d in $dirs) {
    if (Test-Path $d) {
        Remove-Item -Recurse -Force $d
        Write-Host "已清理 $d"
    }
}
if (Test-Path "血细胞检测系统.spec") {
    Remove-Item -Force "血细胞检测系统.spec"
}

Write-Host "开始打包..." -ForegroundColor Green

pyinstaller --onefile --windowed --name "血细胞检测系统" `
    --add-data "ultralytics;ultralytics" `
    --hidden-import "ultralytics" `
    --hidden-import "ultralytics.nn.modules" `
    --hidden-import "ultralytics.nn.tasks" `
    --hidden-import "ultralytics.utils" `
    --hidden-import "torch" `
    --hidden-import "torchvision" `
    --hidden-import "cv2" `
    --hidden-import "numpy" `
    --hidden-import "PyQt5" `
    --collect-all "ultralytics" `
    --collect-all "torch" `
    gui.py

Write-Host ""
Write-Host "打包完成！" -ForegroundColor Green
Write-Host "EXE 文件位置: dist\血细胞检测系统.exe" -ForegroundColor Yellow
Write-Host ""
Write-Host "注意：将 best.pt 模型放在与 EXE 同目录的 runs/detect/train7/weights/ 下" -ForegroundColor Yellow
Write-Host "或在程序启动后通过 '加载模型' 按钮手动选择模型文件" -ForegroundColor Yellow
