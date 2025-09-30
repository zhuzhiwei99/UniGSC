import os
import shutil
import argparse
import fnmatch
from typing import List, Union
from tqdm import tqdm
from datetime import datetime

def search_files(root_dir: str, patterns: List[str]) -> List[str]:
    """
    在 root_dir 目录下递归搜索匹配 patterns 的文件

    Args:
        root_dir (str): 根目录
        patterns (List[str]): 文件名或通配符列表

    Returns:
        List[str]: 匹配的文件路径列表
    """
    matches = []
    for dirpath, _, filenames in os.walk(root_dir):
        for pattern in patterns:
            for filename in fnmatch.filter(filenames, pattern):
                matches.append(os.path.join(dirpath, filename))
    return matches

def handle_match_files(
    match_files: List[tuple],
    mode: str,
    dry_run: bool,
    overwrite: bool,
    write_log,
):
    """
    处理源文件和目标文件对的通用函数

    Args:
        match_files (List[tuple]): 包含 (src_file, dst_dir, dst_file) 的列表
        mode (str): 操作模式 'copy' | 'move' | 'symlink'
        dry_run (bool): 是否只打印不执行
        overwrite (bool): 是否覆盖已存在文件
        write_log (function): 用于写日志的函数
    """
    if dry_run:
        print("Dry-run mode: the following operations would be performed:")
        for src_file, dst_dir, dst_file in match_files:
            action = "Overwrite" if (overwrite and os.path.exists(dst_file)) else mode.capitalize()
            msg = f"{action}: {src_file} -> {dst_file}"
            print(msg)
            write_log(msg)
        return

    for src_file, dst_dir, dst_file in tqdm(match_files, desc=f"{mode.capitalize()}ing files"):
        os.makedirs(dst_dir, exist_ok=True)

        if os.path.exists(dst_file) and not overwrite:
            msg = f"Skipped (exists): {dst_file}"
            write_log(msg)
            continue

        if mode == "copy":
            shutil.copy2(src_file, dst_file)
        elif mode == "move":
            shutil.move(src_file, dst_file)
        elif mode == "link":
            if os.path.exists(dst_file):
                os.remove(dst_file)
            os.symlink(src_file, dst_file)

        msg = f"{mode.capitalize()}: {src_file} -> {dst_file}"
        write_log(msg)


def safe_flatten_name(root_dir: str, file_path: str) -> str:
    """
    将文件的相对路径扁平化为安全的文件名
    e.g.  scene1/config.json  ->  scene1_config.json
    """
    rel_path = os.path.relpath(file_path, root_dir)
    # 用下划线替换目录分隔符
    return rel_path.replace(os.sep, "_")

def collect_target_files(
    root_dir: str,
    output_dir: str,
    target_files: Union[str, List[str]],
    flat: bool = False,
    mode: str = "copy",
    dry_run: bool = False,
    overwrite: bool = False,
    log: bool = True,
):
    """
    遍历 root_dir，找到所有匹配 target_files 的文件，
    保持目录结构，将其处理到 output_dir，并保存日志。

    Args:
        root_dir (str): 输入根目录
        output_dir (str): 输出根目录
        target_files (Union[str, List[str]]): 要查找的文件名或通配符，可为单个字符串或字符串列表
        flat (bool): 是否将文件扁平化存放到以文件名命名的目录下，默认 False
        mode (str): 操作模式，可选 'copy', 'move', 'link'，默认 'copy'
        dry_run (bool): 若为 True，则只打印将要执行的操作，不实际执行
        overwrite (bool): 是否允许覆盖已存在文件，默认 False
        log (bool): 是否保存日志，默认 True
    """
    assert mode in ("copy", "move", "link"), f"Unsupported mode: {mode}"

    if isinstance(target_files, str):
        target_files = [target_files]

    root_dir = os.path.abspath(root_dir)
    output_dir = os.path.abspath(output_dir)

    # 收集所有待处理文件
    all_matches = []
    if flat:
        for dirpath, _, filenames in os.walk(root_dir):
            for pattern in target_files:
                for filename in fnmatch.filter(filenames, pattern):
                    src_file = os.path.join(dirpath, filename)

                    # 目标目录：使用输出根目录+当前文件名
                    dst_dir = os.path.join(output_dir, filename)
            
                    # 扁平化文件名
                    flat_name = safe_flatten_name(root_dir, dirpath)
                    dst_file = os.path.join(dst_dir, flat_name + os.path.splitext(filename)[1])

                    all_matches.append((src_file, dst_dir, dst_file))
    else:
        for dirpath, _, filenames in os.walk(root_dir):
            for pattern in target_files:
                for filename in fnmatch.filter(filenames, pattern):
                    src_file = os.path.join(dirpath, filename)
                    rel_path = os.path.relpath(dirpath, root_dir)
                    dst_dir = os.path.join(output_dir, rel_path)
                    dst_file = os.path.join(dst_dir, filename)
                    all_matches.append((src_file, dst_dir, dst_file))

    # 准备日志文件
    log_file = None
    log_dir = os.path.join(output_dir, "logs")
    if log:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"file_ops_{timestamp}.log")

    def write_log(line: str):
        if log_file:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    handle_match_files(
        match_files=all_matches,
        mode=mode,
        dry_run=dry_run,
        overwrite=overwrite,
        write_log=write_log,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Collect specific files from root_dir to output_dir while preserving structure."
    )
    parser.add_argument("root_dir", help="输入根目录")
    parser.add_argument("output_dir", help="输出根目录")
    parser.add_argument(
        "target_files",
        nargs="+",
        help="目标文件名或通配符（可多个），例如: config.json *.yaml"
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="启用后将文件扁平化存放到以文件名命名的目录下"
    ) 
    
    parser.add_argument(
        "--mode",
        choices=["copy", "move", "link"],
        default="copy",
        help="操作模式，默认是 copy"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="启用后只打印将执行的操作，不实际执行"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="是否允许覆盖已存在文件，默认不覆盖"
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="禁用日志保存（默认保存到 output_dir/file_ops_xxx.log）"
    )

    args = parser.parse_args()

    collect_target_files(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        target_files=args.target_files,
        flat=args.flat,
        mode=args.mode,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        log=not args.no_log,
    )


if __name__ == "__main__":
    main()
