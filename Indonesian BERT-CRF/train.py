train_dataset, train_examples, train_features = load_and_cache_examples_fn(
    args,
    tokenizer,
    tag_encoder,
    mode='train',
)

train_output_comp = OutputComposer(
    train_examples,
    train_features,
    output_transform_fn=tag_encoder.convert_ids_to_tags)

if args.valid_file:
    valid_dataset, valid_examples, valid_features = load_and_cache_examples_fn(
        args,
        tokenizer,
        tag_encoder,
        mode='valid',
    )

    valid_output_comp = OutputComposer(
        valid_examples,
        valid_features,
        output_transform_fn=tag_encoder.convert_ids_to_tags)

if args.freeze_bert:
    model.freeze_bert()
    assert model.frozen_bert

    train_dataset = get_bert_encoded_dataset(
        model, train_dataset, args.per_gpu_train_batch_size,
        args.device)
    if valid_dataset:
        valid_dataset = get_bert_encoded_dataset(
            model, valid_dataset, args.per_gpu_train_batch_size,
            args.device)

train_metrics = get_train_metrics_fn(tag_encoder)

if args.valid_file:
    valid_metrics = get_valid_metrics_fn(tag_encoder)

train(
    args,
    model,
    train_dataset,
    train_metrics=train_metrics,
    train_output_composer=train_output_comp,
    valid_dataset=valid_dataset,
    valid_metrics=valid_metrics,
    valid_output_composer=valid_output_comp,
)

model = load_model(args, model_path=args.output_dir, training=False)
model.to(device)
