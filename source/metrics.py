
from seqeval import scheme
from seqeval import metrics

def entity_f1_score(
    annos, preds
    , criterion_ignored_la, tag_names
    , average
):
    active_indices = annos != criterion_ignored_la
    annos, preds = annos[active_indices].tolist(), preds[active_indices].tolist()
    annos, preds = [tag_names[tag] for tag in annos], [tag_names[tag] for tag in preds]

    f1_score = metrics.f1_score(
        [annos], [preds]
        , mode="strict", scheme=scheme.IOB2
        , average=average
    )

    return f1_score

def entity_classification_report(
    annos, preds
    , criterion_ignored_la, tag_names
):
    active_indices = annos != criterion_ignored_la
    annos, preds = annos[active_indices].tolist(), preds[active_indices].tolist()
    annos, preds = [tag_names[tag] for tag in annos], [tag_names[tag] for tag in preds]

    classification_report = metrics.classification_report(
        [annos], [preds]
        , mode="strict", scheme=scheme.IOB2
        , digits=3
    )

    return classification_report