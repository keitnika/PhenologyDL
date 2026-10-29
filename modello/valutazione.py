from modello.supporto import choose_device, file_hash, project_path, read_json, write_json
from modello.architettura import load_checkpoint
"""Metriche e valutazione interna: il test reale non si avvia automaticamente."""
import numpy as np
import torch
from modello.dati import ImageDataset, make_loader, make_evaluation_transform, load_splits
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def predict_frame(model, frame, image_root, config, device, batch_size, workers, seed=42):
    transform = make_evaluation_transform(config)
    loader = make_loader(ImageDataset(frame, image_root, transform), batch_size, workers, False, seed)
    model.eval()
    probabilities, bbch_predictions_raw = [], []
    with torch.inference_mode():
        for images, _, _ in loader:
            logits, bbch_estimate = model(images.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            bbch_predictions_raw.append(bbch_estimate.cpu().numpy())
    probabilities = np.concatenate(probabilities)
    bbch_predictions_raw = np.concatenate(bbch_predictions_raw)
    # Keep the raw regression estimate: clipping can turn a wrong class prediction
    # into a very misleading BBCH number (e.g. true 12 -> forced to 51).
    return probabilities, bbch_predictions_raw

def metrics(true, predicted, class_count=12):
    true, predicted = (np.asarray(true), np.asarray(predicted))
    if len(true) == 0 or len(true) != len(predicted):
        raise ValueError('Etichette e predizioni vuote o di lunghezze diverse')
    report = classification_report(true, predicted, labels=list(range(class_count)), output_dict=True, zero_division=0)
    delta = np.abs(true - predicted)
    values = {'accuracy': float(np.mean(true == predicted)), 'macro_f1': report['macro avg']['f1-score'], 'weighted_f1': report['weighted avg']['f1-score'], 'macro_precision': report['macro avg']['precision'], 'macro_recall': report['macro avg']['recall'], 'weighted_precision': report['weighted avg']['precision'], 'weighted_recall': report['weighted avg']['recall'], 'accuracy_within_1': float(np.mean(delta <= 1)), 'ordinal_mae': float(delta.mean())}
    return (values, report)

def bootstrap(true, predicted, groups, class_count, repetitions, seed):
    groups = np.asarray(groups)
    ids = np.unique(groups)
    indices = [np.flatnonzero(groups == group) for group in ids]
    generator = np.random.default_rng(seed)
    samples, class_samples = ({}, [])
    for _ in range(repetitions):
        chosen = generator.integers(0, len(ids), size=len(ids))
        selection = np.concatenate([indices[index] for index in chosen])
        values, report = metrics(true[selection], predicted[selection], class_count)
        for key, value in values.items():
            samples.setdefault(key, []).append(value)
        class_samples.append([report[str(i)]['f1-score'] for i in range(class_count)])
    intervals = {key: np.percentile(values, [2.5, 97.5]).tolist() for key, values in samples.items()}
    class_intervals = np.percentile(class_samples, [2.5, 97.5], axis=0).T
    return (intervals, class_intervals)

def save_evaluation(frame, probabilities, bbch_predictions, config, destination, bootstrap_config=None):
    destination.mkdir(parents=True, exist_ok=True)
    names = [item['name'] for item in config['classes']]
    true = frame.class_id.to_numpy(int)
    predicted = probabilities.argmax(axis=1)
    values, report = metrics(true, predicted, len(names))
    result = dict(values)
    true_bbch = frame.BBCH.to_numpy(dtype=float)
    result['bbch_mae_raw'] = float(np.mean(np.abs(true_bbch - bbch_predictions)))
    result['bbch_rmse_raw'] = float(np.sqrt(np.mean((true_bbch - bbch_predictions) ** 2)))
    intervals = None
    if bootstrap_config and bootstrap_config['bootstrap_repetitions'] > 0:
        result['confidence_intervals_95'], intervals = bootstrap(true, predicted, frame.group_id.to_numpy(), len(names), bootstrap_config['bootstrap_repetitions'], bootstrap_config['bootstrap_seed'])
        result['bootstrap'] = {k: bootstrap_config[k] for k in ('bootstrap_repetitions', 'bootstrap_seed')}
    rows = []
    interpretable = []
    for index, name in enumerate(names):
        subset = frame[frame.class_id == index]
        image_count, group_count = (len(subset), subset.group_id.nunique())
        min_images = (bootstrap_config or {}).get('minimum_images', 10)
        min_groups = (bootstrap_config or {}).get('minimum_groups', 3)
        reasons = []
        if image_count < min_images:
            reasons.append(f'immagini < {min_images}')
        if group_count < min_groups:
            reasons.append(f'gruppi < {min_groups}')
        if not reasons:
            interpretable.append(index)
        row = {'class_id': index, 'class_name': name, **report[str(index)], 'image_count': image_count, 'group_count': group_count, 'interpretability': 'NON INTERPRETABILE: ' + '; '.join(reasons) if reasons else 'interpretabile'}
        if intervals is not None:
            row.update(f1_lower_95=float(intervals[index, 0]), f1_upper_95=float(intervals[index, 1]))
        rows.append(row)
    result['interpretable_classes'] = [names[i] for i in interpretable]
    result['macro_f1_interpretable_classes'] = float(np.mean([report[str(i)]['f1-score'] for i in interpretable])) if interpretable else None
    errors = np.abs(true - predicted)
    result['adjacent_fraction_of_errors'] = float(np.sum(errors == 1) / np.sum(errors != 0)) if np.any(errors) else None
    pd.DataFrame(rows).to_csv(destination / 'class_report.csv', index=False)
    predictions = frame[['image_name', 'group_id', 'class_id']].copy()
    predictions['predicted_class_id'] = predicted
    predictions['predicted_class_name'] = [names[int(i)] for i in predicted]
    predictions['confidence_score'] = probabilities.max(axis=1)
    predictions['confidence_percent'] = probabilities.max(axis=1) * 100.0
    predictions['true_BBCH'] = true_bbch
    predictions['predicted_BBCH_float_raw'] = bbch_predictions
    predictions['predicted_BBCH_float'] = np.round(bbch_predictions, 3)
    predictions['BBCH_absolute_error'] = np.abs(true_bbch - bbch_predictions)
    for index, name in enumerate(names):
        predictions[f'probability_{name}'] = probabilities[:, index]
    predictions.to_csv(destination / 'predictions.csv', index=False, float_format='%.3f')
    matrix = confusion_matrix(true, predicted, labels=range(len(names)))
    normalized = np.divide(matrix, matrix.sum(axis=1, keepdims=True), out=np.zeros_like(matrix, dtype=float), where=matrix.sum(axis=1, keepdims=True) != 0)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for label, data in (('counts', matrix), ('normalized', normalized)):
        pd.DataFrame(data, index=names, columns=names).to_csv(destination / f'confusion_{label}.csv')
        fig, axis = plt.subplots(figsize=(11, 9))
        plot = axis.imshow(data, cmap='Blues')
        axis.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names, yticklabels=names, xlabel='Classe predetta', ylabel='Classe vera')
        plt.setp(axis.get_xticklabels(), rotation=70, ha='right')
        for i in range(len(names)):
            for j in range(len(names)):
                axis.text(j, i, f'{data[i, j]:.2f}' if label == 'normalized' else str(data[i, j]), ha='center', va='center', fontsize=7)
        fig.colorbar(plot, ax=axis)
        fig.tight_layout()
        fig.savefig(destination / f'confusion_{label}.png', dpi=140)
        plt.close(fig)
    from modello.supporto import write_json
    write_json(destination / 'metrics.json', result)
    return result


def valuta_test(cartella_addestramento, cartella_risultati=None):
    """Solo per il data scientist: mai richiamata dai due comandi pubblici.

    Il test si apre una sola volta per ciascun addestramento/elenco di test.
    Non usare il modello corrente copiato nella radice outputs: indicare la
    cartella versionata che conserva configurazione e conferma di completamento.
    """
    run = project_path(cartella_addestramento)
    runtime = read_json(run / 'config.json')['training']
    settings = read_json(project_path('configs/model.json'))['evaluation_defaults'].copy()
    best_checkpoint = run / 'best_validation.pt'
    if not best_checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint best_validation.pt non trovato in {run}')
    settings.update(checkpoint=str(best_checkpoint), splits_dir=runtime['splits_dir'],
        image_root=runtime['image_root'], dinov2_repo=runtime['dinov2_repo'],
        output_dir=str(project_path(cartella_risultati) if cartella_risultati else run / 'test'),
        device=runtime['device'])
    return _esegui_test_protetto(settings)


def _esegui_test_protetto(settings):
    if not settings['image_root'] or not settings['dinov2_repo']:
        raise ValueError('Specificare image_root e dinov2_repo')
    checkpoint = project_path(settings['checkpoint'])
    parts = load_splits(project_path(settings['splits_dir']))
    output = project_path(settings['output_dir'])
    if output.exists():
        raise FileExistsError('Output test già esistente: nessuna sovrascrittura')
    complete = checkpoint.parent / 'complete.json'
    if not complete.is_file() or not read_json(complete).get('completed'):
        raise RuntimeError('Training non completato: il test resta chiuso')
    identity = file_hash(checkpoint)
    test_identity = file_hash(project_path(settings['splits_dir']) / 'test.csv')
    marker = checkpoint.parent / f'test_access_{test_identity}.json'
    if marker.exists():
        raise RuntimeError('Test già aperto per questo addestramento e questo elenco di test')
    device = choose_device(settings['device'])
    model, config = load_checkpoint(checkpoint, project_path(settings['dinov2_repo']), device)
    with marker.open('x', encoding='utf-8') as stream:
        import json
        json.dump({'status': 'started', 'checkpoint_sha256': identity, 'test_manifest_sha256': test_identity}, stream)
    probabilities, bbch_predictions = predict_frame(model, parts['test'], project_path(settings['image_root']), config, device, settings['batch_size'], settings['workers'], settings['bootstrap_seed'])
    result = save_evaluation(parts['test'], probabilities, bbch_predictions, config, output, settings)
    write_json(marker, {'status': 'completed', 'checkpoint_sha256': identity, 'test_manifest_sha256': test_identity})
    write_json(output / 'config.json', settings)
    return result
