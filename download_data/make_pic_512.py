from PIL import Image, ImageDraw
import random
import os
from tqdm import tqdm


def generate_random_line_image(output_path, size=(512, 512), num_points=50, num_lines=100, background_color=0, line_color=255):
    """
    ランダムな点と線を生成してモノクロ画像を生成する関数

    Args:
        output_path (str): 出力画像ファイルのパス
        size (tuple): 画像のサイズ (幅, 高さ)
        num_points (int): 画像内に配置するランダムな点の数
        num_lines (int): 点同士を結ぶ線の数
        background_color (int): 背景色 (0:黒, 255:白)
        line_color (int): 点と線の色 (0:黒, 255:白)
    """
    img = Image.new('L', size, background_color)
    draw = ImageDraw.Draw(img)

    width, height = size

    # ランダムな点を生成
    points = []
    for _ in range(num_points):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        points.append((x, y))

    # 生成した点の中からランダムに2つを選び、線を描画
    if num_points > 1:
        for _ in range(num_lines):
            # ランダムに2つの点を選ぶ
            if len(points) >= 2:
                p1, p2 = random.sample(points, 2)
            else:
                continue

            draw.line([p1, p2], fill=line_color, width=5)

    img_monochrome = img.point(lambda x: 0 if x < 128 else 255, '1')
    img_monochrome.save(output_path)


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), os.pardir, "data", "pic_512")
    partition_file = os.path.join(output_dir, "partition_file.csv")
    num_images_to_generate = 100000

    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not os.path.exists(os.path.join(output_dir, "image")):
        os.makedirs(os.path.join(output_dir, "image"))

    
    with open(partition_file, 'w') as f:
        f.write("file_name,type\n")

        # tqdmでループをラップすることでプログレスバーが表示される
        for i in tqdm(range(num_images_to_generate), desc="Generating Images"):
            # ファイル名
            output_filename = f'random_lines_{i:07d}.png'

            # 分割番号の設定
            if i < 0.7 * num_images_to_generate:
                partition_type = 0
            elif 0.7 * num_images_to_generate <= i and i < 0.9 * num_images_to_generate:
                partition_type = 1
            else:
                partition_type = 2

            # 分割ファイルへの保存
            f.write(f"{output_filename},{partition_type}\n")
            output_path = os.path.join(output_dir, "image", output_filename)

            generate_random_line_image(
                output_path,
                num_points=random.randint(3, 10),
                num_lines=random.randint(3, 10),
                background_color=255,
                line_color=0
            )

    print(f"\n{num_images_to_generate}枚の画像生成が完了しました。")
    print(f"保存先: {output_dir}")
