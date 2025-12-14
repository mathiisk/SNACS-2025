import torch
from config import Config
from helpers.generate_sample import create_data_masks, subsample_train
from helpers.reproducibility import set_seed
from helpers.process_data import write_to_csv, summarize_results, load_dataset, edge_homophily, average_results
from helpers.train import train
import time


def run_single_experiment(cfg, model_name, dataset_name, sampling_ratio, sampler_name, seed, verbose=True):
    set_seed(seed)
    dataset, data = load_dataset(dataset_name)

    # fixed split on full graph
    data = create_data_masks(data, cfg)

    model = cfg.models[model_name](
        in_dim=dataset.num_node_features,
        out_dim=dataset.num_classes,
        params=cfg
    )

    if sampling_ratio == 1.0:
        sampled_data = data
        sampler_label = "full_graph"
    else:
        sampler = cfg.samplers[sampler_name](params=cfg)
        sampled_data = subsample_train(data, sampler, sampling_ratio)
        sampler_label = sampler_name

    print(
        f"Sample |V'|={sampled_data.num_nodes}, edges={sampled_data.edge_index.size(1)} -> "
        f"sampled_train_nodes={int(sampled_data.train_mask.sum())} || "
        f"val={int(sampled_data.val_mask.sum())} || test={int(sampled_data.test_mask.sum())} || "
        
    )


    num_sampled_nodes = sampled_data.num_nodes
    num_sampled_edges = sampled_data.edge_index.size(1)

    optimizer = cfg.optimizer(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=cfg.lr_schedule_patience,
        factor=cfg.lr_reduce_factor, min_lr=cfg.min_lr
    )

    final_scores = train(model, sampled_data, optimizer, scheduler, cfg, verbose=verbose)

    return {
        "dataset": dataset_name,
        "model": model_name,
        "sampler": sampler_label,
        "ratio": sampling_ratio,
        "train_f1": final_scores["train"]["f1"],
        "val_f1": final_scores["val"]["f1"],
        "test_f1": final_scores["test"]["f1"],
        "num_nodes": num_sampled_nodes,
        "num_edges": num_sampled_edges,
    }


def run_experiments(verbose=True):
    cfg = Config()
    start_time = time.time()

    for dataset_name in cfg.datasets:
        raw_results = []  # reset per dataset

        print("\n" + "=" * 80)
        dataset, data = load_dataset(dataset_name)
        homophily = edge_homophily(data.edge_index, data.y)
        print(f"Dataset: {dataset_name} -> homophily: {homophily:.3f}")

        for model_name in cfg.models:
            print("\nTesting model:", model_name)

            for ratio in cfg.sampling_ratios:
                print("Ratio:", ratio)
                sampler_names = ["full_graph"] if ratio == 1.0 else list(cfg.samplers.keys())

                for sampler_name in sampler_names:
                    print("Sampler:", sampler_name)

                    for seed in cfg.seeds:
                        print("Seed:", seed)
                        res = run_single_experiment(
                            cfg, model_name, dataset_name, ratio, sampler_name, seed, verbose=verbose
                        )
                        raw_results.append(res)

        bad = [r for r in raw_results if "train_f1" not in r]
        if bad:
            raise RuntimeError(f"Malformed result (missing train_f1). Keys: {list(bad[0].keys())}")

        summary = average_results(raw_results)
        summarize_results(cfg, summary)
        write_to_csv(summary, f"results_{dataset_name}.csv")

    print(time.time() - start_time)


if __name__ == "__main__":

    run_experiments()
