eval_dataset, eval_examples, eval_features = load_and_cache_examples_fn(
    args,
    tokenizer,
    tag_encoder,
    mode='eval',
)

eval_output_comp = OutputComposer(
    eval_examples,
    eval_features,
    output_transform_fn=tag_encoder.convert_ids_to_tags)

eval_sampler = SequentialSampler(eval_dataset)


eval_dataloader = DataLoader(eval_dataset,
                                sampler=eval_sampler,
                                batch_size=args.per_gpu_eval_batch_size,
                                num_workers=os.cpu_count())

eval_metrics = get_eval_metrics_fn(tag_encoder)

metrics = evaluate(
    args,
    model,
    tqdm(eval_dataloader, desc="Evaluation"),
    eval_output_comp,
    eval_metrics,
    reset=False,
)

metrics_values = []
for metric_name in ('f1_score', 'precision', 'recall'):
    metric_value = metrics[metric_name]
    metrics_values.append(metric_value)

with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as fd:
    fd.write(','.join(map(str, metrics_values)))

conll_file = os.path.join(args.output_dir, 'predictions_conll.txt')

y_pred = eval_output_comp.get_outputs()
y_pred_filt = [tag_encoder.decode_valid(preds) for preds in y_pred]

