# cv_pic_2_coverage
cv识别图像中的苔藓覆盖率

1.处理原始图像，提取样框内的图像
2.cv识别图像内的苔藓，计算覆盖率
3.统计结果输出(image_name, coverage)


在Linux/macos上运行
1. python -m venv test_env
2. source ./test_env/bin/activate
3. pip install -r requirements.txt
4. 准备好调查样方文件夹
5. python ./run_light.py

