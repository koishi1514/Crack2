import os
from PIL import Image
import math

def split_single_image_into_patches_by_count(input_image_path, output_sub_dir, num_patches_x, num_patches_y):
    """
    辅助函数：将一张大图切分成指定行数和列数的patch，并保存到指定的子目录。

    Args:
        input_image_path (str): 输入大图的完整路径。
        output_sub_dir (str): 保存patch的子目录。
        num_patches_x (int): 水平方向（列）的patch数量。
        num_patches_y (int): 垂直方向（行）的patch数量。
    Returns:
        int: 成功保存的patch数量。
    """
    try:
        img = Image.open(input_image_path)
        img_width, img_height = img.size
    except Exception as e:
        print(f"  错误：无法打开或读取图片文件 {input_image_path} - {e}")
        return 0

    # 确保输出子目录存在
    os.makedirs(output_sub_dir, exist_ok=True)

    # 获取原始图片的文件名和扩展名
    base_name = os.path.splitext(os.path.basename(input_image_path))[0]
    file_extension = os.path.splitext(input_image_path)[1]

    print(f"  图片尺寸：{img_width}x{img_height}，将切分为 {num_patches_y} 行 x {num_patches_x} 列。")

    patch_count = 0
    for y_idx in range(num_patches_y):
        for x_idx in range(num_patches_x):
            # 计算当前patch的左上角坐标
            # 使用 int() 进行截断，确保是整数像素坐标
            left = int(x_idx * img_width / num_patches_x)
            upper = int(y_idx * img_height / num_patches_y)

            # 计算当前patch的右下角坐标
            # 同样使用 int() 截断
            right = int((x_idx + 1) * img_width / num_patches_x)
            lower = int((y_idx + 1) * img_height / num_patches_y)

            # 确保最后一个patch覆盖到图片边缘
            right = min(right, img_width)
            lower = min(lower, img_height)

            # 裁剪patch
            patch = img.crop((left, upper, right, lower))

            # 构造输出文件名
            output_filename = f"{base_name}_r{y_idx}_c{x_idx}{file_extension}"
            output_path = os.path.join(output_sub_dir, output_filename)

            try:
                patch.save(output_path)
                patch_count += 1
            except Exception as e:
                print(f"  错误：无法保存patch {output_path} - {e}")
    return patch_count

if __name__ == "__main__":
    # 1. 定义输入图片路径
    # 请将 'path/to/your/large_image.png' 替换为你的大图的实际路径
    # 建议先放一张测试图片，例如一张1000x800的图片
    ori_test_data_path = '../../dataset/for_test_20250704/KNet-result'

    # 2. 定义输出目录
    # 所有切分出来的patch将保存到这个目录
    output_directory = '../../dataset/for_test_cropped_20250704/binary_labels'

    # 3. 定义每个patch的尺寸
    # 例如，如果你想把图片切成 256x256 的小块
    patch_num_x = 10
    patch_num_y = 8

    # --- 创建一个示例大图用于测试 (可选，如果你已经有大图可以跳过这步) ---
    # 这段代码会生成一个 1000x800 的白色图片，你可以用它来测试

    for filename in os.listdir(ori_test_data_path):
        if os.path.isfile(os.path.join(ori_test_data_path, filename)):
            input_image_path = os.path.join(ori_test_data_path, filename)
            split_single_image_into_patches_by_count(input_image_path, output_directory, patch_num_x, patch_num_y)

