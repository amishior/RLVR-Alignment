import pandas as pd
from pathlib import Path
from typing import Union

def jsonl_to_parquet(
    input_path: Union[str, Path],
    output_path: Union[str, Path] = None,
    compression: str = "snappy",
    chunk_size: int = None
) -> None:
    """
    将JSONL文件转换为Parquet格式
    
    参数:
        input_path: 输入JSONL文件路径
        output_path: 输出Parquet文件路径（默认为输入文件名改扩展名）
        compression: 压缩算法 ('snappy', 'gzip', 'brotli', 'none')
        chunk_size: 分块读取行数（用于大文件），默认None表示一次性读取
    """
    input_path = Path(input_path)
    
    # 自动生成输出路径
    if output_path is None:
        output_path = input_path.with_suffix(".parquet")
    else:
        output_path = Path(output_path)
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")
    
    print(f"开始转换: {input_path}")
    print(f"输出路径: {output_path}")
    
    if chunk_size:
        # 分块处理大文件
        reader = pd.read_json(input_path, lines=True, chunksize=chunk_size)
        
        for idx, chunk in enumerate(reader, 1):
            part_path = output_path.with_stem(f"{output_path.stem}_part{idx:03d}")
            chunk.to_parquet(part_path, compression=compression, index=False)
            print(f"  写入分块 {idx}: {part_path.name} ({len(chunk):,}行)")
    else:
        # 一次性读取转换
        df = pd.read_json(input_path, lines=True)
        
        # 写入Parquet文件
        df.to_parquet(output_path, compression=compression, index=False)
        
        # 打印统计信息
        print(f"\n✅ 转换成功!")
        print(f"数据量: {len(df):,} 行 × {len(df.columns)} 列")
        print(f"压缩方式: {compression}")
        print(f"\n列名预览: {list(df.columns)}")


# 使用示例
if __name__ == "__main__":
    # 示例1: 基本用法（推荐）
    jsonl_to_parquet(
        input_path="datasets-insurance-regulate-alpaca-0.7k.jsonl",
        output_path="datasets-insurance-regulate-alpaca-0.7k.parquet"
    )
    
    # 示例2: 处理大文件（每1万行一个文件）
    # jsonl_to_parquet(
    #     input_path="large_dataset.jsonl",
    #     output_path="output/dataset.parquet",
    #     chunk_size=10000
    # )