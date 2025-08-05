import os
import shutil
from PIL import Image
import re
from collections import defaultdict

def replace_and_reconstruct_patches_v2(
        dir_labeled_patches,
        dir_unlabeled_patches,
        output_images_dir,
        num_rows=8, # 假设每个原始图像被分割成固定的行数
        num_cols=10,  # 假设每个原始图像被分割成固定的列数
        patch_filename_pattern=r"^(.*)_r(\d+)_c(\d+)\.jpg$" # 正则表达式模式
):
    """
    将有标注的图像补丁替换到无标注的目录中，然后按原始图像名称重组为完整的图像。

    Args:
        dir_labeled_patches (str): 存储有标注补丁的目录路径 (目录a)。
        dir_unlabeled_patches (str): 存储无标注补丁的目录路径 (目录b)。
        output_images_dir (str): 重组后完整图像的保存目录。
        num_rows (int): 每个原始图像被分割成的行数 (例如 10)。
        num_cols (int): 每个原始图像被分割成的列数 (例如 8)。
        patch_filename_pattern (str): 补丁文件名的正则表达式模式。
                                      期望捕获组：1-原始图像名，2-行索引，3-列索引。
                                      例如: r"^(.*)_r(\d+)_c(\d+)\.png$"
    """

    print(f"有标注补丁目录: {dir_labeled_patches}")
    print(f"无标注补丁目录 (将被修改): {dir_unlabeled_patches}")
    print(f"输出完整图像目录: {output_images_dir}")
    print(f"每个图像的补丁网格: {num_rows}x{num_cols} = {num_rows * num_cols} 个补丁")

    # 编译正则表达式
    regex = re.compile(patch_filename_pattern)

    for filename in os.listdir(dir_unlabeled_patches):
        unlabeled_patch_path = os.path.join(dir_unlabeled_patches, filename)
        temp_img = Image.open(unlabeled_patch_path)
        target_patch_size = temp_img.size
        temp_img.close() # 关闭图像文件句柄
        print(f"检测到无标注补丁尺寸: {target_patch_size[0]}x{target_patch_size[1]}")
        break # 找到一个补丁尺寸即可


    labeled_files_count = 0
    for filename in os.listdir(dir_labeled_patches):
        if filename.endswith('.png'): # 只处理符合命名模式的文件
            src_path = os.path.join(dir_labeled_patches, filename)
            dst_path = os.path.join(dir_unlabeled_patches, filename)

            labeled_img = Image.open(src_path)
            labeled_img = labeled_img.resize(target_patch_size, Image.Resampling.LANCZOS)
            dst_path = dst_path[:-4]+'.jpg'
            labeled_img.save(dst_path[:-4]+'.jpg')
            labeled_img.close()

            labeled_files_count += 1

    print(f"已从 '{dir_labeled_patches}' 替换 {labeled_files_count} 个符合模式的补丁到 '{dir_unlabeled_patches}'。")

    # --- 步骤 2: 重组图像 ---
    print("\n--- 步骤 2: 重组补丁为完整图像 ---")
    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)

    # 存储按原始图像名称分组的补丁
    # { original_image_name: { (row, col): Image_object } }
    image_patches_map = defaultdict(dict)


    # 遍历无标注目录中的所有补丁文件
    for filename in os.listdir(dir_unlabeled_patches):
        match = regex.match(filename)
        if match:
            original_image_name = match.group(1)

            row_idx = int(match.group(2))
            col_idx = int(match.group(3))

            patch_path = os.path.join(dir_unlabeled_patches, filename)

            img = Image.open(patch_path)

            image_patches_map[original_image_name][(row_idx, col_idx)] = img

        else:
            print(f"警告: 文件 '{filename}' 不符合预期的补丁命名模式，跳过。")


    # 遍历每个原始图像，进行重组
    for original_image_name, patches_dict in image_patches_map.items():
        print(f"\n--- 正在重组图像: {original_image_name} ---")
        missing_patches_coords = []
        for r in range(num_rows):
            for c in range(num_cols):
                if (r, c) not in patches_dict:
                    missing_patches_coords.append((r, c))

        if missing_patches_coords:
            # 尝试从已存在的补丁中获取模式和尺寸，否则使用默认值
            sample_patch_mode = list(patches_dict.values())[0].mode if patches_dict else 'RGB'
            # 缺失补丁的尺寸应该与目标尺寸一致 (因为有标注的补丁已经调整过)
            dummy_patch = Image.new(sample_patch_mode, target_patch_size, color='black')
            print(f"警告: 图像 '{original_image_name}' 缺少 {len(missing_patches_coords)} 个补丁，将用黑色填充。")
            for r_m, c_m in missing_patches_coords:
                patches_dict[(r_m, c_m)] = dummy_patch

        actual_col_widths = [0] * num_cols
        actual_row_heights = [0] * num_rows

        for r in range(num_rows):
            for c in range(num_cols):
                patch_img = patches_dict[(r, c)] # 此时 guaranteed to exist
                patch_w, patch_h = patch_img.size
                actual_col_widths[c] = max(actual_col_widths[c], patch_w) # 更新该列的最大宽度
                actual_row_heights[r] = max(actual_row_heights[r], patch_h) # 更新该行的最大高度

        reconstructed_width = sum(actual_col_widths)
        reconstructed_height = sum(actual_row_heights)


        # 创建一个新的空白图像
        # 确保使用正确的模式 (例如 'RGB', 'L')
        # 从一个已加载的补丁中获取模式
        sample_patch = list(patches_dict.values())[0]
        reconstructed_image = Image.new(sample_patch.mode, (reconstructed_width, reconstructed_height))

        # 将补丁粘贴到正确的位置
        current_y_offset = 0
        for r in range(num_rows):
            current_x_offset = 0
            for c in range(num_cols):
                patch_img = patches_dict[(r, c)]
                reconstructed_image.paste(patch_img, (current_x_offset, current_y_offset))
                current_x_offset += actual_col_widths[c] # 累加该列的最大宽度
            current_y_offset += actual_row_heights[r] # 累加该行的最大高度

        # 保存重组后的图像
        output_path = os.path.join(output_images_dir, f"{original_image_name}.jpg")

        try:
            reconstructed_image.save(output_path)
            print(f"成功重组图像并保存到: {output_path}")
        except Exception as e:
            print(f"错误: 保存重组图像 '{original_image_name}' 失败: {e}")

    print("\n--- 处理完成 ---")

# --- 示例用法 ---
if __name__ == "__main__":
    # --- 1. 定义目录和文件路径 ---
    DIR_LABELED_PATCHES = "../../dataset/测试样本结果/for_test_cropped_20250704_checked_post"
    DIR_UNLABELED_PATCHES = "../../dataset/测试样本结果/images"
    OUTPUT_IMAGES_DIR = "../../dataset/测试样本结果/outputs/nnunet_post"

    NUM_ROWS = 8
    NUM_COLS = 10

    # --- 2. (可选) 创建一些虚拟数据用于测试 ---
    # print("--- 正在生成虚拟测试数据 (实际使用时请注释掉此部分) ---")
    # if os.path.exists(DIR_LABELED_PATCHES): shutil.rmtree(DIR_LABELED_PATCHES)
    # if os.path.exists(DIR_UNLABELED_PATCHES): shutil.rmtree(DIR_UNLABELED_PATCHES)
    # if os.path.exists(OUTPUT_IMAGES_DIR): shutil.rmtree(OUTPUT_IMAGES_DIR)

    # os.makedirs(DIR_LABELED_PATCHES)
    # os.makedirs(DIR_UNLABELED_PATCHES)
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)


    # --- 虚拟数据生成结束 ---

    # --- 3. 调用主函数执行操作 ---
    replace_and_reconstruct_patches_v2(
        dir_labeled_patches=DIR_LABELED_PATCHES,
        dir_unlabeled_patches=DIR_UNLABELED_PATCHES,
        output_images_dir=OUTPUT_IMAGES_DIR,
        num_rows=NUM_ROWS,
        num_cols=NUM_COLS
    )

    # --- 4. (可选) 清理虚拟数据 ---
    # print("\n--- 清理虚拟数据 ---")
    # shutil.rmtree(DIR_LABELED_PATCHES)
    # shutil.rmtree(DIR_UNLABELED_PATCHES)
    # shutil.rmtree(OUTPUT_IMAGES_DIR) # 清理输出目录
    # print("--- 清理完成 ---")
