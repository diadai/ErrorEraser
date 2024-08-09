import pandas as pd

# 解析txt文件内容为DataFrame
def parse_txt_to_df(file_path):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.strip():
                try:
                    parts = line.split()
                    row = {
                        "Forget Rate": float(parts[0].split(':')[1]),
                        "Epoch": int(parts[1].split(':')[1]),
                        "Forget Epoch": int(parts[2].split(':')[1]),
                        "Percent": float(parts[3].split(':')[1]),
                        "Average Accuracy": float(parts[5].split(':')[1]),
                        "Average Forget": float(parts[7].split(':')[1]),
                        "Average Each Accuracy": float(parts[10].split(':')[1]),
                        "Comprehensive Indicators": float(parts[12].split(':')[1])
                    }
                    data.append(row)
                except (IndexError, ValueError) as e:
                    print(f"Error parsing line: {line}")
                    print(f"Error: {e}")
    return pd.DataFrame(data)

# 转换并保存为Excel文件
def convert_to_excel(file_paths, output_paths):
    for file_path, output_path in zip(file_paths, output_paths):
        df = parse_txt_to_df(file_path)
        df.to_excel(output_path, index=False)

# 转换并保存为Excel文件
def convert_to_excel(file_paths, output_paths):
    for file_path, output_path in zip(file_paths, output_paths):
        df = parse_txt_to_df(file_path)
        df.to_excel(output_path, index=False)

# 文件路径
file_paths = [
    'results_LwF_MNIST.txt',
    'results_DERPP_MNIST.txt',
    'results_HAT_MNIST.txt',
    'results_MAS_MNIST.txt',
    'results_NF_EWC_0504_MNIST.txt',
]

# 输出Excel文件路径
output_paths = [
    'results_LwF_MNIST.xlsx',
    'results_DERPP_MNIST.xlsx',
    'results_HAT_MNIST.xlsx',
    'results_MAS_MNIST.xlsx',
    'results_NF_EWC_0504_MNIST.xlsx',
]

convert_to_excel(file_paths, output_paths)
