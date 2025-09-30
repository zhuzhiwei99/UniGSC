import json
import os
import time
import tyro
import yaml
from gsplat.distributed import cli
from gsc import deep_update_object
from gsc.runner import Runner, Config, VgscCodecConfig, GpccCodecConfig

def render(runner: Runner):  
    runner.logger.info("Starting rendering process...")
    render_stage_list = []
    if runner.cfg.ply_dir is not None:
        # Load PLY sequences from the specified directory
        runner.load_ply_sequences(runner.cfg.ply_dir)
        runner.eval(render_stage="val")
        render_stage_list.append("val")
    if runner.cfg.compressed_ply_dir is not None:
        # Load compressed PLY sequences from the specified directory
        runner.load_ply_sequences(runner.cfg.compressed_ply_dir)
        runner.eval(render_stage="compress")
        render_stage_list.append("compress")

def eval(runner: Runner, compress_stats=None):  
    runner.logger.info("Starting evaluation process...")
    render_stats = None

    try:
        runner.logger.info(f"Loading render stats from {runner.cfg.ori_render_dir}/stats/val.json")
        render_stats_json = f"{runner.cfg.ori_render_dir}/stats/val.json"
        with open(render_stats_json, "r") as f:
            render_stats = json.load(f)
    except FileNotFoundError:
        runner.logger.warning(f"Render stats not found: {render_stats_json}. Rendering uncompressed PLYs for evaluation...")
        runner.logger.info(f"Evaluating uncompressed PLYs from {runner.cfg.ply_dir}")
        runner.load_ply_sequences(runner.cfg.ply_dir)
        render_stats = runner.eval(render_stage="val")

    if compress_stats is not None:
        runner.logger.info("Using provided compressed stats for evaluation.")
    elif os.path.exists(f"{runner.stats_dir}/compress.json"):
        # load compress stats from existing rendered results
        runner.logger.info(f"Loading compress stats from {runner.stats_dir}/compress.json")
        compress_stats_json = f"{runner.stats_dir}/compress.json"
        with open(compress_stats_json, "r") as f:
            compress_stats = json.load(f)
    else:
        if runner.cfg.compressed_ply_dir is not None:
            runner.load_ply_sequences(runner.cfg.compressed_ply_dir)
            runner.logger.info(f"Evaluating compressed PLYs from {runner.cfg.compressed_ply_dir}")
        else:
            try:
                runner.load_ply_sequences(runner.reconstructed_dir)
            except FileNotFoundError: 
                runner.logger.error(f"Compressed PLY not found: {runner.reconstructed_dir}. Maybe you need to run the 'decode' process first and set 'save_rec_ply' to True in the codec config.")
            runner.logger.info(f"Evaluating compressed PLYs from {runner.reconstructed_dir}") 

        compress_stats = runner.eval(render_stage="compress")
    runner.compare_render_stats(render_stats, compress_stats, name1="Original", name2="Compressed")
    
    runner.eval_pngs_with_gsc_ctc_metrics(ref_prefix="val", test_prefix="compress")
    runner.summary()


def encode(runner: Runner):
    runner.logger.info("Starting encoding process...")  
    load_quantized = runner.load_quant_ply_sequences()
    if not load_quantized:
        runner.load_ply_sequences(runner.cfg.ply_dir) 
        runner.preprocess()
        runner.quantize()
          
    if isinstance(runner.cfg.codec, VgscCodecConfig):
        runner.vgsc_encode()
    elif isinstance(runner.cfg.codec, GpccCodecConfig):
        runner.gpcc_encode()
    else:
        raise NotImplementedError(f"{type(runner.cfg.codec).__name__} has not been implemented.")
    
    
def decode(runner: Runner):
    runner.logger.info("Starting decoding process...")
    
    if isinstance(runner.cfg.codec, VgscCodecConfig):
        runner.vgsc_decode()
    elif isinstance(runner.cfg.codec, GpccCodecConfig):
        runner.gpcc_decode()
    else:
        raise NotImplementedError(f"{type(runner.cfg.codec).__name__} has not been implemented.")
        
    runner.dequantize()  
    runner.postprocess()
    
    if runner.cfg.codec.save_rec_ply:
        runner.save_ply()
    else:
        runner.logger.warning("Reconstructed PLY not saved. Set 'save_rec_ply' to True in the codec config to save it.")
    
def preprocess(runner: Runner):
    runner.logger.info("Test preprocessing...")
    runner.load_ply_sequences(runner.cfg.ply_dir)
    ori_stats = runner.eval(render_stage="val")
    runner.preprocess()
    runner.postprocess()
    process_stats = runner.eval(render_stage="processed")
    runner.compare_render_stats(ori_stats, process_stats, name1="Original", name2="Processed")
    
def quantize(runner: Runner):
    runner.logger.info("Test quantization...")
    runner.load_ply_sequences(runner.cfg.ply_dir)
    ori_stats = runner.eval(render_stage="val")
    runner.quantize()
    runner.dequantize()
    quant_stats = runner.eval(render_stage="quantized")
    runner.compare_render_stats(ori_stats, quant_stats, name1="Original", name2="Quantized")
    

def main(local_rank: int, world_rank: int, world_size: int, cfg: Config):  
    runner = Runner(local_rank, world_rank, world_size, cfg)
    if cfg.pipe_stage == "render":
        render(runner)
    elif cfg.pipe_stage == "eval":
        eval(runner)
    elif cfg.pipe_stage == "encode":
        encode(runner)
    elif cfg.pipe_stage == "decode":
        decode(runner)
    elif cfg.pipe_stage == "decode_eval":
        decode(runner)
        compress_stats = runner.eval(render_stage="compress")
        eval(runner, compress_stats)
    elif cfg.pipe_stage == "codec":
        encode(runner)
        decode(runner)
    elif cfg.pipe_stage == "benchmark":
        encode(runner)
        decode(runner)   
        compress_stats = runner.eval(render_stage="compress")
        eval(runner, compress_stats)
    elif cfg.pipe_stage == "preprocess":
        preprocess(runner)
    elif cfg.pipe_stage == "quantize":
        quantize(runner)
    else:
        raise ValueError(f"Unknown pipeline stage: {cfg.pipe_stage}.")
        

if __name__ == "__main__":
    import argparse
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument("-c", "--config", type=str, help="Path to a YAML config file.")
    
    args, remaining_argv = conf_parser.parse_known_args()
    
    # Init the base configuration based on the codec type
    if remaining_argv[0] == "vgsc":
        base_cfg = Config(codec=VgscCodecConfig())
    elif remaining_argv[0] == "gpcc":
        base_cfg = Config(codec=GpccCodecConfig())
    else:
        raise ValueError(f"Unknown codec type: {remaining_argv[0]}. Please use 'vgsc' or 'gpcc'.")
    # Load the YAML configuration if provided
    if args.config is not None:
        print(f"Loading configuration from: {args.config}")
        with open(args.config, "r") as f:
            yaml_dict = yaml.safe_load(f)
        base_cfg = deep_update_object(base_cfg, yaml_dict)
        
    # Create a dictionary of configurations for the CLI
    configs = {}
    configs[remaining_argv[0]] = (
        f"Configuration for {remaining_argv[0]} codec.",
        base_cfg
    )
    # Parse the command line arguments using Tyro    
    cfg = tyro.extras.overridable_config_cli(configs=configs, args=remaining_argv)
        
    cli(main, cfg, verbose=True)
